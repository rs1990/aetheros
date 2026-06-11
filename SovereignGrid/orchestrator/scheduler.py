"""APScheduler wiring: stale-job requeue loop and hourly ledger settlement.

The requeue loop is the safety net behind WebSocket-level failover: if a
socket drop is missed (process kill, network partition where the TCP close
never arrives), the lease on the running job expires and the job returns to
the queue on the next sweep.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import dispatch
import ledger
from config import settings
from db import store

log = logging.getLogger("grid.scheduler")

scheduler = AsyncIOScheduler()


async def requeue_stale_jobs() -> None:
    stale = store.stale_running_jobs()
    for job in stale:
        if store.requeue(job["id"], "lease expired"):
            log.warning("job %s lease expired — requeued", job["id"][:8])
    # Opportunistically push the queue at every sweep so requeued (and
    # previously unroutable) jobs go out as soon as capacity appears.
    dispatched = await dispatch.drain_queue()
    if stale or dispatched:
        log.info("sweep: %d stale requeued, %d dispatched", len(stale), dispatched)


async def settle_ledger() -> None:
    ledger.settle()


def start() -> None:
    scheduler.add_job(
        requeue_stale_jobs,
        IntervalTrigger(seconds=settings.REQUEUE_INTERVAL_SECONDS),
        id="requeue-stale",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        settle_ledger,
        CronTrigger(minute=settings.SETTLE_CRON_MINUTE),
        id="settle-ledger",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    log.info("scheduler started: requeue every %ds, settle hourly at :%02d",
             settings.REQUEUE_INTERVAL_SECONDS, settings.SETTLE_CRON_MINUTE)


def stop() -> None:
    scheduler.shutdown(wait=False)
