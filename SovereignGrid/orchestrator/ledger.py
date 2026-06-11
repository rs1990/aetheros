"""Compute Proof: append-only inference ledger and periodic settlement.

Every completed inference writes one ledger row priced by the model-size rate
table (a 70B model pays out more per generated token than an 8B). Balances on
the nodes table are only touched by the hourly settle batch — never on the
request hot path — so high-throughput routing causes no write contention.
"""

from __future__ import annotations

import logging

from db import store
from models import JobResult
from registry import NodeConnection

log = logging.getLogger("grid.ledger")


def credit_inference(result: JobResult, conn: NodeConnection) -> None:
    credits = store.record_inference(
        job_id=result.job_id,
        node_id=conn.node_id,
        model=_job_model(result, conn),
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
    )
    log.info("ledger: %s earned %.4f credits (%d tokens out)",
             conn.did, credits, result.tokens_out)


def _job_model(result: JobResult, conn: NodeConnection) -> str:
    # The daemon echoes usage; model identity comes from the assignment, which
    # the store recorded on mark_running. Fallback: first model the node serves.
    if result.output and isinstance(result.output, dict):
        model = result.output.get("model")
        if isinstance(model, str) and model:
            return model
    return next(iter(conn.models), "unknown")


def settle() -> None:
    credited = store.settle()
    if credited:
        total = sum(credited.values())
        log.info("settlement: %.4f credits across %d nodes", total, len(credited))
    else:
        log.info("settlement: nothing to settle")
