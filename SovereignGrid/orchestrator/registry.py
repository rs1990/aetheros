"""Dynamic hardware registry: live WebSocket connections and model routing index.

The registry is in-process state (which sockets are alive right now); Supabase
holds the durable mirror (node rows, status). On disconnect every in-flight
job owned by that socket is failed over to the queue.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from fastapi import WebSocket

log = logging.getLogger("grid.registry")


@dataclass
class NodeConnection:
    node_id: str                       # Supabase nodes.id
    did: str
    wallet: str
    ws: WebSocket
    models: set[str]
    max_concurrency: int
    engine: str
    semaphore: asyncio.Semaphore = field(init=False)
    in_flight: dict[str, asyncio.Future] = field(default_factory=dict)  # job_id → result future

    def __post_init__(self) -> None:
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

    @property
    def load(self) -> int:
        return len(self.in_flight)

    @property
    def has_capacity(self) -> bool:
        return self.load < self.max_concurrency


class Registry:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeConnection] = {}     # node_id → conn
        self._by_model: dict[str, set[str]] = {}        # model → node_ids
        self._lock = asyncio.Lock()

    async def add(self, conn: NodeConnection) -> None:
        async with self._lock:
            # A reconnect replaces the old socket for the same node.
            old = self._nodes.pop(conn.node_id, None)
            if old is not None:
                self._unindex(old)
            self._nodes[conn.node_id] = conn
            for m in conn.models:
                self._by_model.setdefault(m, set()).add(conn.node_id)
        log.info("node online: %s models=%s cap=%d", conn.did, sorted(conn.models), conn.max_concurrency)

    def _unindex(self, conn: NodeConnection) -> None:
        for m in conn.models:
            peers = self._by_model.get(m)
            if peers:
                peers.discard(conn.node_id)
                if not peers:
                    self._by_model.pop(m, None)

    async def remove(self, node_id: str) -> Optional[NodeConnection]:
        """Drop a node and return it so the caller can fail over its jobs."""
        async with self._lock:
            conn = self._nodes.pop(node_id, None)
            if conn is not None:
                self._unindex(conn)
                log.info("node offline: %s (%d jobs in flight)", conn.did, conn.load)
            return conn

    async def pick(self, model: str) -> Optional[NodeConnection]:
        """Least-loaded node that serves `model` and has spare capacity."""
        async with self._lock:
            candidates = [
                self._nodes[nid]
                for nid in self._by_model.get(model, ())
                if nid in self._nodes and self._nodes[nid].has_capacity
            ]
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.load)

    def get(self, node_id: str) -> Optional[NodeConnection]:
        return self._nodes.get(node_id)

    def snapshot(self) -> list[dict]:
        return [
            {
                "did": c.did,
                "models": sorted(c.models),
                "engine": c.engine,
                "load": c.load,
                "max_concurrency": c.max_concurrency,
            }
            for c in self._nodes.values()
        ]

    def models_available(self) -> dict[str, int]:
        return {m: len(nids) for m, nids in self._by_model.items()}


registry = Registry()
