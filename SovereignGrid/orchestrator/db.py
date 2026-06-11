"""State machine access layer.

Supabase (PostgreSQL) is the source of truth in production. When SUPABASE_URL
is unset the layer degrades to an in-memory store so the grid can be developed
and demoed without a database. The interface is identical either way.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from config import settings

log = logging.getLogger("grid.db")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryStore:
    """Dev-only fallback. Single-process, no durability."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.guild_admins: dict[str, dict] = {}
        self.guild_members: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}
        self.ledger: list[dict] = []
        self.model_rates: dict[str, dict] = {
            "llama3.1:70b": {"param_b": 70, "credit_per_1k": 10.0},
            "llama3.1:8b": {"param_b": 8, "credit_per_1k": 1.5},
            "mistral:7b": {"param_b": 7, "credit_per_1k": 1.2},
            "phi3": {"param_b": 4, "credit_per_1k": 0.8},
        }

    # ── guild ──
    def is_admitted(self, wallet: str) -> Optional[dict]:
        m = self.guild_members.get(wallet.lower())
        if m and m.get("revoked_at") is None:
            return m
        return None

    def admin_exists(self, wallet: str) -> bool:
        return wallet.lower() in self.guild_admins

    # ── nodes ──
    def upsert_node(self, wallet: str, **fields: Any) -> dict:
        node = self.nodes.get(wallet.lower())
        if node is None:
            node = {
                "id": str(uuid.uuid4()),
                "did": f"did:grid:{wallet.lower()}",
                "wallet_address": wallet,
                "balance_credits": 0.0,
                "first_seen_at": _now(),
            }
            self.nodes[wallet.lower()] = node
        node.update(fields)
        node["last_seen_at"] = _now()
        return node

    def set_node_status(self, node_id: str, status: str) -> None:
        for n in self.nodes.values():
            if n["id"] == node_id:
                n["status"] = status

    # ── jobs ──
    def create_job(self, consumer_key: str, model: str, payload: dict) -> dict:
        job = {
            "id": str(uuid.uuid4()),
            "consumer_key": consumer_key,
            "model": model,
            "payload": payload,
            "status": "queued",
            "assigned_node": None,
            "attempts": 0,
            "max_attempts": settings.JOB_MAX_ATTEMPTS,
            "lease_expires_at": None,
            "result": None,
            "error": None,
            "created_at": _now(),
        }
        self.jobs[job["id"]] = job
        return job

    def mark_running(self, job_id: str, node_id: str) -> None:
        job = self.jobs[job_id]
        job.update(
            status="running",
            assigned_node=node_id,
            attempts=job["attempts"] + 1,
            lease_expires_at=_now() + timedelta(seconds=settings.JOB_LEASE_SECONDS),
        )

    def mark_done(self, job_id: str, ok: bool, result: Optional[dict], error: Optional[str]) -> None:
        job = self.jobs[job_id]
        job.update(status="succeeded" if ok else "failed", result=result, error=error)

    def requeue(self, job_id: str, reason: str) -> bool:
        """Return job to the queue; returns False when attempts are exhausted."""
        job = self.jobs[job_id]
        if job["attempts"] >= job["max_attempts"]:
            job.update(status="dead", error=f"max attempts exhausted ({reason})")
            return False
        job.update(status="queued", assigned_node=None, lease_expires_at=None,
                   error=f"requeued: {reason}")
        return True

    def get_job(self, job_id: str) -> Optional[dict]:
        return self.jobs.get(job_id)

    def stale_running_jobs(self) -> list[dict]:
        now = _now()
        return [j for j in self.jobs.values()
                if j["status"] == "running"
                and j["lease_expires_at"] and j["lease_expires_at"] < now]

    def queued_jobs(self, model: Optional[str] = None) -> list[dict]:
        jobs = [j for j in self.jobs.values() if j["status"] == "queued"]
        if model:
            jobs = [j for j in jobs if j["model"] == model]
        return sorted(jobs, key=lambda j: j["created_at"])

    # ── ledger ──
    def rate_for(self, model: str) -> dict:
        for pattern, rate in self.model_rates.items():
            if model.startswith(pattern):
                return rate
        return {"param_b": 7, "credit_per_1k": 1.0}

    def record_inference(self, job_id: str, node_id: str, model: str,
                         tokens_in: int, tokens_out: int) -> float:
        rate = self.rate_for(model)
        credits = round(tokens_out / 1000.0 * rate["credit_per_1k"], 6)
        self.ledger.append({
            "job_id": job_id, "node_id": node_id, "model": model,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "credits": credits, "settled": False, "created_at": _now(),
        })
        return credits

    def settle(self) -> dict[str, float]:
        """Aggregate unsettled rows into node balances. Returns node_id → credited."""
        credited: dict[str, float] = {}
        for row in self.ledger:
            if row["settled"]:
                continue
            credited[row["node_id"]] = credited.get(row["node_id"], 0.0) + row["credits"]
            row["settled"] = True
        for node in self.nodes.values():
            if node["id"] in credited:
                node["balance_credits"] += credited[node["id"]]
        return credited


class SupabaseStore:
    """Production store backed by Supabase. Mirrors MemoryStore's interface."""

    def __init__(self) -> None:
        from supabase import create_client  # lazy import — optional dependency
        self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)

    def is_admitted(self, wallet: str) -> Optional[dict]:
        rows = (self.client.table("guild_members").select("*")
                .eq("wallet_address", wallet).is_("revoked_at", "null")
                .limit(1).execute().data)
        return rows[0] if rows else None

    def admin_exists(self, wallet: str) -> bool:
        rows = (self.client.table("guild_admins").select("wallet_address")
                .eq("wallet_address", wallet).limit(1).execute().data)
        return bool(rows)

    def upsert_node(self, wallet: str, **fields: Any) -> dict:
        payload = {
            "did": f"did:grid:{wallet.lower()}",
            "wallet_address": wallet,
            "last_seen_at": _now().isoformat(),
            **fields,
        }
        rows = (self.client.table("nodes")
                .upsert(payload, on_conflict="wallet_address")
                .execute().data)
        return rows[0]

    def set_node_status(self, node_id: str, status: str) -> None:
        self.client.table("nodes").update({"status": status}).eq("id", node_id).execute()

    def create_job(self, consumer_key: str, model: str, payload: dict) -> dict:
        rows = (self.client.table("job_queue").insert({
            "consumer_key": consumer_key, "model": model, "payload": payload,
            "max_attempts": settings.JOB_MAX_ATTEMPTS,
        }).execute().data)
        return rows[0]

    def mark_running(self, job_id: str, node_id: str) -> None:
        lease = (_now() + timedelta(seconds=settings.JOB_LEASE_SECONDS)).isoformat()
        job = self.client.table("job_queue").select("attempts").eq("id", job_id).execute().data[0]
        self.client.table("job_queue").update({
            "status": "running", "assigned_node": node_id,
            "attempts": job["attempts"] + 1, "lease_expires_at": lease,
        }).eq("id", job_id).execute()

    def mark_done(self, job_id: str, ok: bool, result: Optional[dict], error: Optional[str]) -> None:
        self.client.table("job_queue").update({
            "status": "succeeded" if ok else "failed",
            "result": result, "error": error,
        }).eq("id", job_id).execute()

    def requeue(self, job_id: str, reason: str) -> bool:
        job = (self.client.table("job_queue").select("attempts,max_attempts")
               .eq("id", job_id).execute().data[0])
        if job["attempts"] >= job["max_attempts"]:
            self.client.table("job_queue").update({
                "status": "dead", "error": f"max attempts exhausted ({reason})",
            }).eq("id", job_id).execute()
            return False
        self.client.table("job_queue").update({
            "status": "queued", "assigned_node": None,
            "lease_expires_at": None, "error": f"requeued: {reason}",
        }).eq("id", job_id).execute()
        return True

    def get_job(self, job_id: str) -> Optional[dict]:
        rows = self.client.table("job_queue").select("*").eq("id", job_id).execute().data
        return rows[0] if rows else None

    def stale_running_jobs(self) -> list[dict]:
        return (self.client.table("job_queue").select("*")
                .eq("status", "running")
                .lt("lease_expires_at", _now().isoformat())
                .execute().data)

    def queued_jobs(self, model: Optional[str] = None) -> list[dict]:
        q = self.client.table("job_queue").select("*").eq("status", "queued")
        if model:
            q = q.eq("model", model)
        return q.order("created_at").execute().data

    def rate_for(self, model: str) -> dict:
        rows = self.client.table("model_rates").select("*").execute().data
        for r in sorted(rows, key=lambda r: -len(r["model_pattern"])):
            if model.startswith(r["model_pattern"]):
                return r
        return {"param_b": 7, "credit_per_1k": 1.0}

    def record_inference(self, job_id: str, node_id: str, model: str,
                         tokens_in: int, tokens_out: int) -> float:
        rate = self.rate_for(model)
        credits = round(tokens_out / 1000.0 * float(rate["credit_per_1k"]), 6)
        self.client.table("inference_ledger").insert({
            "job_id": job_id, "node_id": node_id, "model": model,
            "tokens_in": tokens_in, "tokens_out": tokens_out, "credits": credits,
        }).execute()
        return credits

    def settle(self) -> dict[str, float]:
        rows = self.client.rpc("settle_ledger").execute().data or []
        credited: dict[str, float] = {}
        for r in rows:
            credited[r["node_id"]] = credited.get(r["node_id"], 0.0) + float(r["credited"])
        return credited


def build_store() -> MemoryStore | SupabaseStore:
    if settings.supabase_enabled:
        log.info("state store: Supabase (%s)", settings.SUPABASE_URL)
        return SupabaseStore()
    log.warning("SUPABASE_URL unset — using in-memory store (dev only, no durability)")
    return MemoryStore()


store = build_store()
