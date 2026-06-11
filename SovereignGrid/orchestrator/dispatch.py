"""Smart dispatch: route queued jobs to live provider sockets.

A dispatched job owns an asyncio.Future resolved when the daemon returns a
result frame. Disconnection, timeout, or an error frame all funnel into the
same requeue path, so a consumer request survives any single node failure
(up to max_attempts).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from config import settings
from db import store
from ledger import credit_inference
from models import JobAssignment, JobResult
from registry import NodeConnection, registry

log = logging.getLogger("grid.dispatch")


async def dispatch_job(job: dict) -> Optional[tuple[asyncio.Future, NodeConnection]]:
    """Send one queued job to the best node. Returns (result future, node), or
    None when no node currently serves the model (job stays queued)."""
    conn = await registry.pick(job["model"])
    if conn is None:
        return None

    fut: asyncio.Future = asyncio.get_running_loop().create_future()
    conn.in_flight[job["id"]] = fut
    store.mark_running(job["id"], conn.node_id)

    assignment = JobAssignment(
        job_id=job["id"],
        model=job["model"],
        payload=job["payload"],
        deadline_seconds=settings.JOB_LEASE_SECONDS,
    )
    try:
        await conn.ws.send_text(assignment.model_dump_json())
    except Exception as exc:
        conn.in_flight.pop(job["id"], None)
        _requeue_or_kill(job["id"], f"send failed: {exc}")
        if not fut.done():
            fut.set_exception(ConnectionError(str(exc)))
        return fut, conn

    log.info("job %s → %s (%s)", job["id"][:8], conn.did, job["model"])
    return fut, conn


async def await_result(job: dict, fut: asyncio.Future) -> JobResult:
    """Wait for the daemon's result with a deadline; requeue on any failure."""
    try:
        result: JobResult = await asyncio.wait_for(
            fut, timeout=settings.DISPATCH_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        _requeue_or_kill(job["id"], "dispatch timeout")
        raise
    except ConnectionError:
        # Socket died mid-inference — handle_result already requeued via
        # fail_node_jobs; surface to the caller for retry.
        raise

    if not result.ok:
        _requeue_or_kill(job["id"], result.error or "provider error")
        raise RuntimeError(result.error or "provider error")

    return result


def handle_result(conn: NodeConnection, result: JobResult) -> None:
    """Called by the WS receive loop when a daemon reports completion."""
    fut = conn.in_flight.pop(result.job_id, None)
    if fut is None or fut.done():
        log.warning("orphan result for job %s from %s", result.job_id[:8], conn.did)
        return

    if result.ok:
        store.mark_done(result.job_id, True, result.output, None)
        credit_inference(result, conn)
    # Failure bookkeeping happens in await_result via the requeue path.
    fut.set_result(result)


def fail_node_jobs(conn: NodeConnection, reason: str) -> None:
    """On disconnect: every in-flight job on this socket is immediately
    requeued and its waiter unblocked, so consumers fail over without polling."""
    for job_id, fut in list(conn.in_flight.items()):
        _requeue_or_kill(job_id, reason)
        if not fut.done():
            fut.set_exception(ConnectionError(reason))
    conn.in_flight.clear()


def _requeue_or_kill(job_id: str, reason: str) -> None:
    if store.requeue(job_id, reason):
        log.info("job %s requeued (%s)", job_id[:8], reason)
    else:
        log.warning("job %s dead: %s", job_id[:8], reason)


async def drain_queue() -> int:
    """Try to dispatch every queued job that now has a serving node.
    Called by the scheduler and after each new node handshake."""
    dispatched = 0
    for job in store.queued_jobs():
        pair = await dispatch_job(job)
        if pair is not None:
            dispatched += 1
            # Fire-and-forget watcher: jobs drained from the queue have no
            # waiting HTTP consumer attached here; results land in job_queue
            # and are fetched via GET /v1/jobs/{id}.
            asyncio.create_task(_watch(job, pair[0]))
    return dispatched


async def _watch(job: dict, fut: asyncio.Future) -> None:
    try:
        await await_result(job, fut)
    except Exception:
        pass  # requeue path already handled bookkeeping
