"""Qdrant ANN index.

Holds vectors plus the small payload that filtering needs *during* HNSW
traversal - scope, validity, kind, predicate. Text and graph structure live in
Neo4j; this store is rebuildable from it at any time.
"""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from .config import Settings, settings
from .models import Scope


@dataclass(slots=True)
class VectorHit:
    memory_id: str
    score: float


class QdrantVectorStore:
    def __init__(self, cfg: Settings | None = None, client: AsyncQdrantClient | None = None):
        self._cfg = cfg or settings()
        self._client = client or AsyncQdrantClient(
            url=self._cfg.qdrant_url,
            api_key=self._cfg.qdrant_api_key,
            prefer_grpc=False,
        )
        self._collection = self._cfg.qdrant_collection

    async def ensure_schema(self) -> None:
        """Create the collection and the payload indexes filtering depends on.

        Without keyword indexes, Qdrant cannot use its filterable HNSW path and
        falls back to scanning - the whole reason we chose it over brute force.
        """
        if await self._client.collection_exists(self._collection):
            info = await self._client.get_collection(self._collection)
            existing = info.config.params.vectors.size
            if existing != self._cfg.embedding_dim:
                raise ValueError(
                    f"collection {self._collection!r} stores {existing}-dim vectors but "
                    f"EMBEDDING_DIM is {self._cfg.embedding_dim}. Vectors from different "
                    "models are not comparable: re-embed into a new collection, or set "
                    "EMBEDDING_DIM back."
                )
        else:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(
                    size=self._cfg.embedding_dim,
                    distance=qm.Distance.COSINE,
                ),
                hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=200),
                optimizers_config=qm.OptimizersConfigDiff(default_segment_number=2),
            )

        for field, schema in (
            ("user_id", qm.PayloadSchemaType.KEYWORD),
            ("app_id", qm.PayloadSchemaType.KEYWORD),
            ("agent_id", qm.PayloadSchemaType.KEYWORD),
            ("valid", qm.PayloadSchemaType.BOOL),
            ("kind", qm.PayloadSchemaType.KEYWORD),
            ("predicate", qm.PayloadSchemaType.KEYWORD),
            ("subject_key", qm.PayloadSchemaType.KEYWORD),
            ("created_ts", qm.PayloadSchemaType.INTEGER),
        ):
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
            except Exception:  # already indexed
                pass

    async def upsert(
        self,
        *,
        memory_id: str,
        user_id: str,
        app_id: str,
        agent_id: str,
        vector: list[float],
        kind: str,
        created_ts: int,
        predicate: str | None = None,
        subject_key: str | None = None,
        valid: bool = True,
    ) -> None:
        await self._client.upsert(
            collection_name=self._collection,
            wait=True,
            points=[
                qm.PointStruct(
                    id=_point_id(memory_id),
                    vector=vector,
                    payload={
                        "memory_id": memory_id,
                        "user_id": user_id,
                        "app_id": app_id,
                        "agent_id": agent_id,
                        "kind": kind,
                        "predicate": predicate,
                        "subject_key": subject_key,
                        "valid": valid,
                        "created_ts": created_ts,
                    },
                )
            ],
        )

    async def search(
        self,
        *,
        scope: Scope,
        vector: list[float],
        limit: int,
        include_history: bool = False,
        kinds: list[str] | None = None,
        agent_scoped: bool = False,
    ) -> list[VectorHit]:
        must: list[qm.Condition] = [
            qm.FieldCondition(key="user_id", match=qm.MatchValue(value=scope.user_id)),
            qm.FieldCondition(key="app_id", match=qm.MatchValue(value=scope.app_id)),
        ]
        if agent_scoped:
            must.append(
                qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=scope.agent_id))
            )
        if not include_history:
            must.append(qm.FieldCondition(key="valid", match=qm.MatchValue(value=True)))
        if kinds:
            must.append(qm.FieldCondition(key="kind", match=qm.MatchAny(any=kinds)))

        result = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=qm.Filter(must=must),
            limit=limit,
            search_params=qm.SearchParams(hnsw_ef=self._cfg.qdrant_hnsw_ef),
            with_payload=["memory_id"],
        )
        return [VectorHit(memory_id=p.payload["memory_id"], score=p.score) for p in result.points]

    async def neighbours_of(
        self, *, scope: Scope, vector: list[float], limit: int = 20
    ) -> list[VectorHit]:
        """Nearest stored memories to a candidate fact.

        Used by reconciliation: the similarity comes back from the index, so we
        never re-embed memories we already stored.
        """
        return await self.search(scope=scope, vector=vector, limit=limit)

    async def set_validity(self, memory_id: str, valid: bool) -> None:
        await self._client.set_payload(
            collection_name=self._collection,
            payload={"valid": valid},
            points=[_point_id(memory_id)],
            wait=True,
        )

    async def delete(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.PointIdsList(points=[_point_id(m) for m in memory_ids]),
            wait=True,
        )

    async def delete_scope(self, user_id: str, app_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                        qm.FieldCondition(key="app_id", match=qm.MatchValue(value=app_id)),
                    ]
                )
            ),
            wait=True,
        )

    async def close(self) -> None:
        await self._client.close()


def _point_id(memory_id: str) -> str:
    """Qdrant point ids must be uuid or int; our memory ids are uuid4 hex."""
    m = memory_id.replace("-", "")
    return f"{m[0:8]}-{m[8:12]}-{m[12:16]}-{m[16:20]}-{m[20:32]}"
