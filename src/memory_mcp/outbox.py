"""Transactional outbox: Neo4j -> Qdrant.

Two stores cannot be updated atomically. Rather than hope both writes succeed,
every index mutation is committed to Neo4j *in the same transaction as the fact
itself*, then applied to Qdrant by this drainer and deleted on success.

Failures retry with the attempt count on the row; rows that exhaust their
attempts are parked as `dead` for inspection rather than silently dropped.
The invariant: Neo4j is correct, Qdrant is eventually correct.
"""

from __future__ import annotations

import asyncio
import logging

from .config import Settings, settings
from .graph_store import Neo4jGraphStore
from .models import OutboxOp
from .vector_store import QdrantVectorStore

log = logging.getLogger(__name__)


class OutboxDrainer:
    def __init__(
        self,
        graph: Neo4jGraphStore,
        vectors: QdrantVectorStore,
        cfg: Settings | None = None,
    ) -> None:
        self._graph = graph
        self._vectors = vectors
        self._cfg = cfg or settings()

    async def drain_once(self, limit: int = 100) -> int:
        """Apply up to `limit` pending operations. Returns how many succeeded."""
        operations = await self._graph.claim_outbox(limit)
        applied = 0
        for op in operations:
            try:
                await self._apply(op)
                await self._graph.complete_outbox(op["id"])
                applied += 1
            except Exception as exc:
                log.warning("outbox op %s (%s) failed: %s", op["id"], op.get("op"), exc)
                await self._graph.fail_outbox(
                    op["id"], str(exc), self._cfg.outbox_max_attempts
                )
        return applied

    async def _apply(self, op: dict) -> None:
        kind = OutboxOp(op["op"])
        if kind is OutboxOp.INDEX:
            await self._vectors.upsert(
                memory_id=op["memory_id"],
                user_id=op["user_id"],
                app_id=op["app_id"],
                agent_id=op.get("agent_id", "default"),
                vector=list(op["vector"]),
                kind=op.get("kind", "semantic"),
                created_ts=int(op.get("created_ts", 0)),
                predicate=op.get("predicate"),
                subject_key=op.get("subject_key"),
                valid=bool(op.get("valid", True)),
            )
        elif kind is OutboxOp.SET_VALIDITY:
            await self._vectors.set_validity(op["memory_id"], bool(op.get("valid", False)))
        elif kind is OutboxOp.DELETE:
            await self._vectors.delete([op["memory_id"]])
        elif kind is OutboxOp.DELETE_SCOPE:
            await self._vectors.delete_scope(op["user_id"], op["app_id"])
        else:  # pragma: no cover - OutboxOp is exhaustive
            raise ValueError(f"unknown outbox op: {op['op']}")

    async def run_forever(self, stop: asyncio.Event | None = None) -> None:
        """Background loop. Sleeps only when there was nothing to do."""
        stop = stop or asyncio.Event()
        while not stop.is_set():
            try:
                applied = await self.drain_once()
            except Exception:
                log.exception("outbox drain cycle failed")
                applied = 0
            if applied == 0:
                try:
                    await asyncio.wait_for(
                        stop.wait(), timeout=self._cfg.worker_poll_seconds
                    )
                except TimeoutError:
                    pass
