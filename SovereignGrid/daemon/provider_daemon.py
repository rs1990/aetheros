"""SovereignGrid Provider Daemon — the local bridge.

Connects the provider's local inference engine (Ollama / vLLM) to the central
orchestrator over a persistent WebSocket. Authenticates with a wallet
signature (BYOK — the private key never leaves this machine), broadcasts the
capability handshake, then serves job assignments until disconnected.

Run:
    GRID_WALLET_KEY=0x... python provider_daemon.py \
        --gateway ws://grid.example.org:8080/ws/provider \
        --engine ollama --max-concurrency 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys

import httpx
import websockets
from eth_account import Account
from eth_account.messages import encode_defunct

logging.basicConfig(level=logging.INFO, format="%(asctime)s daemon %(message)s")
log = logging.getLogger("grid.daemon")

ENGINE_DEFAULTS = {
    "ollama": "http://localhost:11434",
    "vllm": "http://localhost:8000",
    "aphrodite": "http://localhost:2242",
}


class ProviderDaemon:
    def __init__(self, gateway: str, engine: str, engine_url: str,
                 private_key: str, max_concurrency: int, display_name: str | None):
        self.gateway = gateway
        self.engine = engine
        self.engine_url = engine_url.rstrip("/")
        self.account = Account.from_key(private_key)
        self.max_concurrency = max_concurrency
        self.display_name = display_name
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.http = httpx.AsyncClient(timeout=600)
        self._stop = asyncio.Event()

    # ── local engine ──────────────────────────────────────────────────────────

    async def list_models(self) -> list[str]:
        """Ask the local engine which models are loadable right now."""
        try:
            if self.engine == "ollama":
                r = await self.http.get(f"{self.engine_url}/api/tags")
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
            # vLLM / aphrodite expose OpenAI-compatible /v1/models.
            r = await self.http.get(f"{self.engine_url}/v1/models")
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except Exception as exc:
            log.error("cannot list local models (%s at %s): %s",
                      self.engine, self.engine_url, exc)
            return []

    async def run_inference(self, model: str, payload: dict) -> tuple[dict, int, int]:
        """Execute a chat completion against the strictly-local engine.
        Returns (output, tokens_in, tokens_out)."""
        messages = payload.get("messages", [])
        if self.engine == "ollama":
            r = await self.http.post(f"{self.engine_url}/api/chat", json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": payload.get("max_tokens", 1024),
                    "temperature": payload.get("temperature", 0.7),
                },
            })
            r.raise_for_status()
            data = r.json()
            output = {
                "model": model,
                "message": data.get("message", {}),
            }
            return output, data.get("prompt_eval_count", 0), data.get("eval_count", 0)

        # OpenAI-compatible engines (vLLM, aphrodite)
        r = await self.http.post(f"{self.engine_url}/v1/chat/completions", json={
            "model": model,
            "messages": messages,
            "max_tokens": payload.get("max_tokens", 1024),
            "temperature": payload.get("temperature", 0.7),
        })
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        output = {
            "model": model,
            "message": data.get("choices", [{}])[0].get("message", {}),
        }
        return output, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    # ── gateway protocol ──────────────────────────────────────────────────────

    def sign_challenge(self, nonce: str) -> str:
        signed = Account.sign_message(
            encode_defunct(text=f"grid-auth:{nonce}"), private_key=self.account.key
        )
        return signed.signature.hex()

    async def handle_job(self, ws, msg: dict) -> None:
        """Run one assignment under the concurrency semaphore and report back."""
        job_id = msg["job_id"]
        async with self.semaphore:
            log.info("job %s started (%s)", job_id[:8], msg["model"])
            try:
                output, tin, tout = await asyncio.wait_for(
                    self.run_inference(msg["model"], msg["payload"]),
                    timeout=msg.get("deadline_seconds", 300),
                )
                frame = {"type": "result", "job_id": job_id, "ok": True,
                         "output": output, "tokens_in": tin, "tokens_out": tout}
                log.info("job %s done (%d tokens out)", job_id[:8], tout)
            except Exception as exc:
                frame = {"type": "result", "job_id": job_id, "ok": False,
                         "error": str(exc), "tokens_in": 0, "tokens_out": 0}
                log.warning("job %s failed: %s", job_id[:8], exc)
            try:
                await ws.send(json.dumps(frame))
            except Exception:
                # Socket gone — the orchestrator's disconnect handler requeues.
                log.warning("could not deliver result for %s (socket closed)", job_id[:8])

    async def session(self) -> None:
        """One connected session: challenge → handshake → serve jobs."""
        async with websockets.connect(self.gateway, ping_interval=20) as ws:
            challenge = json.loads(await ws.recv())
            if challenge.get("type") != "challenge":
                raise RuntimeError(f"expected challenge, got {challenge}")

            models = await self.list_models()
            if not models:
                raise RuntimeError("local engine reports no models — refusing to register")

            await ws.send(json.dumps({
                "type": "handshake",
                "wallet_address": self.account.address,
                "signature": self.sign_challenge(challenge["nonce"]),
                "models": models,
                "max_concurrency": self.max_concurrency,
                "engine": self.engine,
                "display_name": self.display_name,
            }))

            welcome = json.loads(await ws.recv())
            if welcome.get("type") != "welcome":
                raise RuntimeError(f"handshake rejected: {welcome}")
            log.info("registered as %s serving %s", welcome["did"], models)

            heartbeat = asyncio.create_task(self._heartbeat_loop(ws))
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "job":
                        asyncio.create_task(self.handle_job(ws, msg))
            finally:
                heartbeat.cancel()

    async def _heartbeat_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                await ws.send(json.dumps({"type": "heartbeat", "running_jobs": []}))
            except Exception:
                return

    async def run_forever(self) -> None:
        """Reconnect loop with exponential backoff."""
        backoff = 1
        while not self._stop.is_set():
            try:
                await self.session()
                backoff = 1
            except Exception as exc:
                log.warning("session ended: %s — reconnecting in %ds", exc, backoff)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60)

    def stop(self) -> None:
        self._stop.set()


def main() -> None:
    p = argparse.ArgumentParser(description="SovereignGrid provider daemon")
    p.add_argument("--gateway", required=True, help="ws(s)://host:port/ws/provider")
    p.add_argument("--engine", choices=ENGINE_DEFAULTS, default="ollama")
    p.add_argument("--engine-url", default=None,
                   help="local engine base URL (default depends on --engine)")
    p.add_argument("--max-concurrency", type=int, default=1)
    p.add_argument("--name", default=None, help="human-readable node label")
    args = p.parse_args()

    key = os.getenv("GRID_WALLET_KEY")
    if not key:
        sys.exit("GRID_WALLET_KEY env var required (never pass keys as CLI args)")

    daemon = ProviderDaemon(
        gateway=args.gateway,
        engine=args.engine,
        engine_url=args.engine_url or ENGINE_DEFAULTS[args.engine],
        private_key=key,
        max_concurrency=args.max_concurrency,
        display_name=args.name,
    )

    loop = asyncio.new_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, daemon.stop)
    try:
        loop.run_until_complete(daemon.run_forever())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
