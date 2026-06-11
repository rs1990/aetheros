"""SovereignGrid orchestrator — FastAPI gateway.

Endpoints:
  WS  /ws/provider          provider daemons connect, handshake, receive jobs
  POST /v1/chat/completions consumer inference (synchronous wait, with failover)
  POST /v1/jobs             consumer inference (async: returns job_id)
  GET  /v1/jobs/{job_id}    async job status/result
  GET  /v1/models           models currently served by online nodes
  GET  /health              liveness + grid summary
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

import auth
import dispatch
import scheduler
from config import settings
from db import store
from models import (
    CompletionRequest,
    CompletionResponse,
    Handshake,
    Heartbeat,
    JobResult,
    JobStatus,
)
from registry import NodeConnection, registry
from sanitize import SanitizationError, sanitize_payload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("grid.main")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(title="SovereignGrid Orchestrator", lifespan=lifespan)


# ─── Provider WebSocket ───────────────────────────────────────────────────────

@app.websocket("/ws/provider")
async def provider_socket(ws: WebSocket) -> None:
    await ws.accept()

    # 1. Challenge: send a one-time nonce the daemon must sign.
    nonce = auth.issue_nonce()
    await ws.send_text(json.dumps({"type": "challenge", "nonce": nonce}))

    # 2. Handshake within the nonce TTL.
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=settings.NONCE_TTL_SECONDS)
        hs = Handshake.model_validate_json(raw)
    except (asyncio.TimeoutError, ValidationError) as exc:
        await ws.close(code=4400, reason=f"bad handshake: {exc}")
        return

    ok, reason = auth.verify_handshake(nonce, hs.wallet_address, hs.signature)
    if not ok:
        log.warning("handshake rejected for %s: %s", hs.wallet_address, reason)
        await ws.close(code=4401, reason=reason)
        return

    node_row = store.upsert_node(
        hs.wallet_address,
        models=hs.models,
        max_concurrency=hs.max_concurrency,
        engine=hs.engine,
        display_name=hs.display_name,
        status="online",
    )
    conn = NodeConnection(
        node_id=node_row["id"],
        did=node_row["did"],
        wallet=hs.wallet_address,
        ws=ws,
        models=set(hs.models),
        max_concurrency=hs.max_concurrency,
        engine=hs.engine,
    )
    await registry.add(conn)
    await ws.send_text(json.dumps({"type": "welcome", "did": conn.did}))

    # New capacity online — drain anything waiting for these models.
    asyncio.create_task(dispatch.drain_queue())

    # 3. Receive loop: results and heartbeats until disconnect.
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "result":
                dispatch.handle_result(conn, JobResult.model_validate(msg))
            elif mtype == "heartbeat":
                Heartbeat.model_validate(msg)  # shape check; lease renewal hook
            else:
                log.warning("unknown frame type %r from %s", mtype, conn.did)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning("socket error for %s: %v", conn.did, exc)
    finally:
        # Immediate failover: requeue all in-flight jobs on this socket.
        await registry.remove(conn.node_id)
        dispatch.fail_node_jobs(conn, f"node {conn.did} disconnected")
        store.set_node_status(conn.node_id, "offline")
        asyncio.create_task(dispatch.drain_queue())


# ─── Consumer API ─────────────────────────────────────────────────────────────

def _consumer_key(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(401, "missing Authorization header")
    return hashlib.sha256(authorization.encode()).hexdigest()[:16]


def _create_sanitized_job(req: CompletionRequest, consumer: str) -> dict:
    try:
        payload = sanitize_payload(req.model_dump())
    except SanitizationError as exc:
        raise HTTPException(422, str(exc)) from exc
    return store.create_job(consumer, req.model, payload)


@app.post("/v1/chat/completions", response_model=CompletionResponse)
async def chat_completions(
    req: CompletionRequest, authorization: str | None = Header(default=None)
):
    """Synchronous inference: dispatch, wait, fail over across nodes."""
    consumer = _consumer_key(authorization)
    job = _create_sanitized_job(req, consumer)

    last_error = "no node serves this model"
    for _ in range(settings.JOB_MAX_ATTEMPTS):
        pair = await dispatch.dispatch_job(job)
        if pair is None:
            raise HTTPException(503, f"no online node serves model {req.model!r}")
        fut, conn = pair
        try:
            result = await dispatch.await_result(job, fut)
            return CompletionResponse(
                job_id=job["id"],
                model=req.model,
                output=result.output or {},
                served_by=conn.did,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
            )
        except Exception as exc:
            last_error = str(exc)
            current = store.get_job(job["id"])
            if current is None or current["status"] != "queued":
                break  # dead (attempts exhausted) or finished elsewhere
    raise HTTPException(502, f"inference failed after retries: {last_error}")


@app.post("/v1/jobs", response_model=JobStatus, status_code=202)
async def submit_job(
    req: CompletionRequest, authorization: str | None = Header(default=None)
):
    """Async inference for heavy background agent workloads: enqueue and return."""
    consumer = _consumer_key(authorization)
    job = _create_sanitized_job(req, consumer)
    asyncio.create_task(dispatch.drain_queue())
    return JobStatus(job_id=job["id"], status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
async def job_status(job_id: str):
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return JobStatus(
        job_id=job_id, status=job["status"], result=job.get("result"), error=job.get("error")
    )


@app.get("/v1/models")
async def models_available():
    return {"models": registry.models_available()}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "nodes_online": len(registry.snapshot()),
        "nodes": registry.snapshot(),
    }
