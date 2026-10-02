"""Neo4j graph store - the source of truth.

Holds raw episodes, the facts derived from them, canonical entities, the
relations between those entities, bi-temporal validity, and the outbox that
drives Qdrant. Qdrant can be dropped and rebuilt from this; the reverse is not
true.

Temporal filtering vocabulary used throughout:
  moment          - the world-time instant to evaluate facts at (default now)
  include_history - ignore both time axes and return everything ever recorded
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from neo4j import AsyncGraphDatabase
from neo4j.time import DateTime as Neo4jDateTime

from .config import Settings, settings
from .models import (
    Confidence,
    ContentCategory,
    Entity,
    Episode,
    EpisodeStatus,
    Memory,
    MemoryKind,
    OutboxOp,
    Scope,
    as_utc,
    utcnow,
)
from .ontology import SourceType

FULLTEXT_INDEX = "memory_text_idx"

#: Applied to every memory read. `moment` is always bound.
_TIME_FILTER = """
  AND ($include_history OR m.invalid_at IS NULL)
  AND ($include_history OR m.valid_from IS NULL OR m.valid_from <= datetime($moment))
  AND ($include_history OR m.valid_to IS NULL OR m.valid_to > datetime($moment))
"""

_SCOPE_FILTER = "m.user_id = $user_id AND m.app_id = $app_id"


def _py(value: Any) -> Any:
    return value.to_native() if isinstance(value, Neo4jDateTime) else value


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _to_memory(node: dict, entities: list[dict] | None = None) -> Memory:
    return Memory(
        id=node["id"],
        user_id=node["user_id"],
        app_id=node.get("app_id", "default"),
        agent_id=node.get("agent_id", "default"),
        text=node["text"],
        kind=MemoryKind(node.get("kind", "semantic")),
        predicate=node.get("predicate"),
        subject_key=node.get("subject_key"),
        object_key=node.get("object_key"),
        object_value=node.get("object_value"),
        importance=node.get("importance", 0.5),
        confidence=Confidence(
            extraction=node.get("confidence_extraction", 1.0),
            source=node.get("confidence_source", 1.0),
            temporal=node.get("confidence_temporal", 1.0),
            entity=node.get("confidence_entity", 1.0),
        ),
        needs_verification=node.get("needs_verification", False),
        episode_id=node.get("episode_id"),
        source=node.get("source"),
        source_type=SourceType(node.get("source_type", "user_statement")),
        source_id=node.get("source_id"),
        embedding_model=node.get("embedding_model"),
        extraction_model=node.get("extraction_model"),
        valid_from=_py(node["valid_from"]),
        valid_to=_py(node.get("valid_to")),
        recorded_at=_py(node["recorded_at"]),
        invalid_at=_py(node.get("invalid_at")),
        superseded_by=node.get("superseded_by"),
        reinforced=node.get("reinforced", 0),
        # OPTIONAL MATCH yields a null-filled map when a memory has no entities.
        entities=[
            Entity(name=e["name"], type=e.get("type") or "thing")
            for e in (entities or [])
            if e and e.get("name")
        ],
    )


def _to_episode(node: dict) -> Episode:
    return Episode(
        id=node["id"],
        user_id=node["user_id"],
        app_id=node.get("app_id", "default"),
        agent_id=node.get("agent_id", "default"),
        raw_text=node["raw_text"],
        source=node.get("source"),
        source_type=SourceType(node.get("source_type", "user_statement")),
        source_id=node.get("source_id"),
        content_hash=node["content_hash"],
        ingested_at=_py(node["ingested_at"]),
        status=EpisodeStatus(node.get("status", "pending")),
        category=ContentCategory(node["category"]) if node.get("category") else None,
        error=node.get("error"),
        attempts=node.get("attempts", 0),
        metadata=json.loads(node["metadata"]) if node.get("metadata") else {},
    )


class Neo4jGraphStore:
    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or settings()
        self._driver = AsyncGraphDatabase.driver(
            self._cfg.neo4j_uri,
            auth=(self._cfg.neo4j_user, self._cfg.neo4j_password),
        )
        self._db = self._cfg.neo4j_database

    async def close(self) -> None:
        await self._driver.close()

    async def _run(self, query: str, **params) -> list[dict]:
        async with self._driver.session(database=self._db) as session:
            result = await session.run(query, **params)
            return [record.data() async for record in result]

    def _time_params(self, moment: datetime | None, include_history: bool) -> dict:
        # Naive datetimes reach here from callers parsing "2024-03-01"; Neo4j
        # would read those as server-local and shift the window by the UTC
        # offset, so every boundary is anchored explicitly.
        return {
            "moment": (as_utc(moment) or utcnow()).isoformat(),
            "include_history": include_history,
        }

    # ---------------------------------------------------------------- schema

    async def ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT episode_id IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
            # Idempotency: one episode per (scope, content hash).
            "CREATE CONSTRAINT episode_dedupe IF NOT EXISTS FOR (e:Episode) "
            "REQUIRE (e.user_id, e.app_id, e.content_hash) IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) "
            "REQUIRE (e.user_id, e.app_id, e.key) IS UNIQUE",
            "CREATE CONSTRAINT outbox_id IF NOT EXISTS FOR (o:Outbox) REQUIRE o.id IS UNIQUE",
            "CREATE INDEX memory_scope IF NOT EXISTS FOR (m:Memory) ON (m.user_id, m.app_id)",
            "CREATE INDEX memory_subject IF NOT EXISTS FOR (m:Memory) "
            "ON (m.subject_key, m.predicate)",
            "CREATE INDEX memory_recorded IF NOT EXISTS FOR (m:Memory) ON (m.recorded_at)",
            "CREATE INDEX episode_status IF NOT EXISTS FOR (e:Episode) ON (e.status)",
            "CREATE INDEX outbox_status IF NOT EXISTS FOR (o:Outbox) ON (o.status)",
            f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS "
            "FOR (m:Memory) ON EACH [m.text]",
        ]
        for stmt in statements:
            await self._run(stmt)

    # -------------------------------------------------------------- episodes

    async def create_episode(self, episode: Episode) -> tuple[Episode, bool]:
        """Idempotent insert. Returns (episode, created).

        Re-sending the same text in the same scope returns the original episode
        instead of reprocessing it - protects against client retries and
        duplicated conversation replays.
        """
        rows = await self._run(
            """
            MERGE (e:Episode {user_id: $user_id, app_id: $app_id, content_hash: $content_hash})
            ON CREATE SET
                e.id = $id, e.agent_id = $agent_id, e.raw_text = $raw_text,
                e.source = $source, e.source_type = $source_type, e.source_id = $source_id,
                e.ingested_at = datetime($ingested_at), e.status = $status,
                e.category = null, e.error = null, e.attempts = 0,
                e.metadata = $metadata
            WITH e
            MERGE (u:User {id: $user_id})
              ON CREATE SET u.created_at = datetime()
            MERGE (u)-[:OBSERVED]->(e)
            RETURN e
            """,
            id=episode.id,
            user_id=episode.user_id,
            app_id=episode.app_id,
            agent_id=episode.agent_id,
            raw_text=episode.raw_text,
            source=episode.source,
            source_type=episode.source_type.value,
            source_id=episode.source_id,
            content_hash=episode.content_hash,
            ingested_at=episode.ingested_at.isoformat(),
            status=episode.status.value,
            metadata=json.dumps(episode.metadata) if episode.metadata else None,
        )
        stored = _to_episode(rows[0]["e"])
        return stored, stored.id == episode.id

    async def set_episode_status(
        self,
        episode_id: str,
        status: EpisodeStatus,
        category: ContentCategory | None = None,
        error: str | None = None,
    ) -> None:
        await self._run(
            """
            MATCH (e:Episode {id: $id})
            SET e.status = $status,
                e.category = coalesce($category, e.category),
                e.error = $error,
                e.updated_at = datetime()
            """,
            id=episode_id,
            status=status.value,
            category=category.value if category else None,
            error=error,
        )

    async def get_episode(self, episode_id: str) -> Episode | None:
        rows = await self._run("MATCH (e:Episode {id: $id}) RETURN e", id=episode_id)
        return _to_episode(rows[0]["e"]) if rows else None

    async def claim_episodes(self, limit: int = 10) -> list[Episode]:
        """Take pending episodes off the queue and mark them in flight."""
        rows = await self._run(
            """
            MATCH (e:Episode) WHERE e.status = $pending
            WITH e ORDER BY e.ingested_at ASC LIMIT $limit
            SET e.status = $processing, e.claimed_at = datetime()
            RETURN e
            """,
            pending=EpisodeStatus.PENDING.value,
            processing=EpisodeStatus.PROCESSING.value,
            limit=limit,
        )
        return [_to_episode(r["e"]) for r in rows]

    async def release_episode(
        self, episode_id: str, error: str, max_attempts: int
    ) -> EpisodeStatus:
        """Hand a failed episode back to the queue, or park it.

        A transient failure (the model timed out) should be retried; a
        permanent one should stop burning attempts and stay visible.
        """
        rows = await self._run(
            """
            MATCH (e:Episode {id: $id})
            WITH e, coalesce(e.attempts, 0) + 1 AS attempts
            SET e.attempts = attempts,
                e.error = $error,
                e.status = CASE WHEN attempts >= $max THEN $failed ELSE $pending END
            RETURN e.status AS status
            """,
            id=episode_id,
            error=error[:500],
            max=max_attempts,
            failed=EpisodeStatus.FAILED.value,
            pending=EpisodeStatus.PENDING.value,
        )
        return EpisodeStatus(rows[0]["status"]) if rows else EpisodeStatus.FAILED

    async def reclaim_stale(self, timeout_seconds: float) -> dict:
        """Release claims held by workers that died.

        Without this, one crashed worker silently stalls those episodes and
        outbox rows forever - the queue looks healthy and nothing moves.
        """
        episodes = await self._run(
            """
            MATCH (e:Episode) WHERE e.status = $processing
              AND e.claimed_at < datetime() - duration({seconds: $timeout})
            SET e.status = $pending
            RETURN count(*) AS n
            """,
            processing=EpisodeStatus.PROCESSING.value,
            pending=EpisodeStatus.PENDING.value,
            timeout=timeout_seconds,
        )
        outbox = await self._run(
            """
            MATCH (o:Outbox) WHERE o.status = 'inflight'
              AND o.claimed_at < datetime() - duration({seconds: $timeout})
            SET o.status = 'pending'
            RETURN count(*) AS n
            """,
            timeout=timeout_seconds,
        )
        return {
            "episodes": episodes[0]["n"] if episodes else 0,
            "outbox": outbox[0]["n"] if outbox else 0,
        }

    async def episode_memories(self, episode_id: str) -> list[Memory]:
        rows = await self._run(
            """
            MATCH (:Episode {id: $id})-[:DERIVED]->(m:Memory)
            OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
            RETURN m, collect(e {.name, type: e.type}) AS entities
            """,
            id=episode_id,
        )
        return [_to_memory(r["m"], r["entities"]) for r in rows]

    # -------------------------------------------------------------- entities

    async def list_entities(self, scope: Scope, limit: int = 500) -> list[dict]:
        rows = await self._run(
            """
            MATCH (e:Entity {user_id: $user_id, app_id: $app_id})
            RETURN e.id AS id, e.key AS key, e.canonical_name AS canonical_name,
                   e.type AS type, e.aliases AS aliases, e.embedding AS embedding
            ORDER BY e.created_at DESC
            LIMIT $limit
            """,
            user_id=scope.user_id,
            app_id=scope.app_id,
            limit=limit,
        )
        return rows

    async def create_entity(self, scope: Scope, entity: dict) -> None:
        await self._run(
            """
            MERGE (e:Entity {user_id: $user_id, app_id: $app_id, key: $key})
            ON CREATE SET e.id = $id, e.canonical_name = $canonical_name, e.type = $type,
                          e.aliases = $aliases, e.embedding = $embedding,
                          e.created_at = datetime()
            """,
            user_id=scope.user_id,
            app_id=scope.app_id,
            key=entity["key"],
            id=entity["id"],
            canonical_name=entity["canonical_name"],
            type=entity["type"],
            aliases=entity.get("aliases", []),
            embedding=entity.get("embedding"),
        )

    async def add_alias(self, entity_id: str, alias: str) -> None:
        await self._run(
            """
            MATCH (e:Entity {id: $id})
            SET e.aliases = CASE WHEN $alias IN coalesce(e.aliases, [])
                                 THEN e.aliases
                                 ELSE coalesce(e.aliases, []) + $alias END
            """,
            id=entity_id,
            alias=alias,
        )

    # ----------------------------------------------------------------- write

    async def write_memory(
        self,
        memory: Memory,
        entity_ids: list[str],
        relations: list[dict],
        vector: list[float],
    ) -> None:
        """Write a memory, its edges, and its outbox entry in one transaction.

        The outbox row is what guarantees the vector index eventually matches;
        it is committed atomically with the fact, so a crash cannot lose it.
        """
        await self._run(
            """
            MERGE (u:User {id: $user_id})
              ON CREATE SET u.created_at = datetime()
            CREATE (m:Memory {
                id: $id, user_id: $user_id, app_id: $app_id, agent_id: $agent_id,
                text: $text, kind: $kind,
                predicate: $predicate, subject_key: $subject_key,
                object_key: $object_key, object_value: $object_value,
                importance: $importance,
                confidence_extraction: $c_extraction, confidence_source: $c_source,
                confidence_temporal: $c_temporal, confidence_entity: $c_entity,
                confidence: $confidence, needs_verification: $needs_verification,
                episode_id: $episode_id, source: $source, source_type: $source_type,
                source_id: $source_id, embedding_model: $embedding_model,
                extraction_model: $extraction_model,
                valid_from: datetime($valid_from),
                valid_to: CASE WHEN $valid_to IS NULL THEN null ELSE datetime($valid_to) END,
                recorded_at: datetime($recorded_at), invalid_at: null,
                superseded_by: null, reinforced: 0
            })
            MERGE (u)-[:OWNS]->(m)
            WITH m
            CALL (m) {
                MATCH (ep:Episode {id: $episode_id})
                MERGE (ep)-[:DERIVED]->(m)
            }
            CALL (m) {
                UNWIND $entity_ids AS eid
                    MATCH (e:Entity {id: eid})
                    MERGE (m)-[:MENTIONS]->(e)
            }
            CALL (m) {
                UNWIND $relations AS rel
                    MATCH (s:Entity {id: rel.subject_id})
                    MATCH (o:Entity {id: rel.object_id})
                    MERGE (s)-[r:RELATED {predicate: rel.predicate}]->(o)
                      ON CREATE SET r.created_at = datetime(), r.memory_id = m.id
                      ON MATCH  SET r.memory_id = m.id
            }
            CREATE (:Outbox {
                id: $outbox_id, op: $op, memory_id: m.id,
                user_id: $user_id, app_id: $app_id, agent_id: $agent_id,
                kind: $kind, predicate: $predicate, subject_key: $subject_key,
                vector: $vector, valid: true,
                created_ts: $created_ts, status: 'pending', attempts: 0,
                created_at: datetime()
            })
            RETURN m.id AS id
            """,
            id=memory.id,
            user_id=memory.user_id,
            app_id=memory.app_id,
            agent_id=memory.agent_id,
            text=memory.text,
            kind=memory.kind.value,
            predicate=memory.predicate,
            subject_key=memory.subject_key,
            object_key=memory.object_key,
            object_value=memory.object_value,
            importance=memory.importance,
            c_extraction=memory.confidence.extraction,
            c_source=memory.confidence.source,
            c_temporal=memory.confidence.temporal,
            c_entity=memory.confidence.entity,
            confidence=memory.confidence.final,
            needs_verification=memory.needs_verification,
            episode_id=memory.episode_id,
            source=memory.source,
            source_type=memory.source_type.value,
            source_id=memory.source_id,
            embedding_model=memory.embedding_model,
            extraction_model=memory.extraction_model,
            valid_from=memory.valid_from.isoformat(),
            valid_to=_iso(memory.valid_to),
            recorded_at=memory.recorded_at.isoformat(),
            entity_ids=entity_ids,
            relations=relations,
            outbox_id=f"ob-{memory.id}",
            op=OutboxOp.INDEX.value,
            vector=vector,
            created_ts=int(memory.recorded_at.timestamp()),
        )

    async def reinforce(self, memory_id: str) -> None:
        """A repeat sighting of a known fact: raises confidence, not row count."""
        await self._run(
            """
            MATCH (m:Memory {id: $id})
            SET m.reinforced = coalesce(m.reinforced, 0) + 1,
                m.importance = CASE WHEN m.importance < 0.95
                                    THEN m.importance + 0.05 ELSE m.importance END,
                m.last_seen = datetime()
            """,
            id=memory_id,
        )

    async def supersede(
        self,
        old_id: str,
        new_id: str,
        valid_to: datetime,
        at: datetime | None = None,
    ) -> None:
        """Close out a fact that a newer one replaced.

        Only world time moves. The old fact stopped being *true* at `valid_to`;
        it was never *wrong*, so `invalid_at` stays null and an as-of query can
        still answer with it. `invalid_at` is reserved for records we retract -
        corrections and forget() - which is a different question entirely.
        """
        await self._run(
            """
            MATCH (old:Memory {id: $old_id})
            SET old.valid_to = datetime($valid_to),
                old.superseded_by = $new_id
            WITH old
            MATCH (new:Memory {id: $new_id})
            MERGE (new)-[:SUPERSEDES]->(old)
            WITH old
            CALL (old) {
                // The edge this memory asserted stops holding too, or the graph
                // would still say the user lives in both cities.
                MATCH ()-[r:RELATED {memory_id: old.id}]->()
                SET r.valid_to = datetime($valid_to)
            }
            WITH old
            CREATE (:Outbox {
                id: randomUUID(), op: $op, memory_id: old.id,
                user_id: old.user_id, app_id: old.app_id,
                valid: false, status: 'pending', attempts: 0, created_at: datetime()
            })
            """,
            old_id=old_id,
            new_id=new_id,
            valid_to=(valid_to or at or utcnow()).isoformat(),
            op=OutboxOp.SET_VALIDITY.value,
        )

    async def invalidate(self, memory_id: str, at: datetime | None = None) -> None:
        """Stop believing a fact without claiming it stopped being true."""
        await self._run(
            """
            MATCH (m:Memory {id: $id})
            SET m.invalid_at = datetime($at)
            CREATE (:Outbox {
                id: randomUUID(), op: $op, memory_id: m.id,
                user_id: m.user_id, app_id: m.app_id,
                valid: false, status: 'pending', attempts: 0, created_at: datetime()
            })
            """,
            id=memory_id,
            at=(at or utcnow()).isoformat(),
            op=OutboxOp.SET_VALIDITY.value,
        )

    async def hard_delete(self, memory_ids: list[str]) -> int:
        rows = await self._run(
            """
            MATCH (m:Memory) WHERE m.id IN $ids
            WITH collect(m) AS ms
            CALL (ms) {
                UNWIND ms AS m
                CREATE (:Outbox {
                    id: randomUUID(), op: $op, memory_id: m.id,
                    user_id: m.user_id, app_id: m.app_id,
                    status: 'pending', attempts: 0, created_at: datetime()
                })
            }
            UNWIND ms AS m
            DETACH DELETE m
            RETURN count(*) AS n
            """,
            ids=memory_ids,
            op=OutboxOp.DELETE.value,
        )
        return rows[0]["n"] if rows else 0

    async def delete_scope(self, scope: Scope) -> int:
        """Right-to-erasure for one (user, app). Episodes go too."""
        rows = await self._run(
            """
            MATCH (m:Memory {user_id: $user_id, app_id: $app_id})
            WITH collect(m) AS ms
            OPTIONAL MATCH (e:Episode {user_id: $user_id, app_id: $app_id})
            WITH ms, collect(e) AS eps
            OPTIONAL MATCH (en:Entity {user_id: $user_id, app_id: $app_id})
            WITH ms, eps, collect(en) AS ens
            CREATE (:Outbox {
                id: randomUUID(), op: $op, user_id: $user_id, app_id: $app_id,
                status: 'pending', attempts: 0, created_at: datetime()
            })
            FOREACH (x IN ms | DETACH DELETE x)
            FOREACH (x IN eps | DETACH DELETE x)
            FOREACH (x IN ens | DETACH DELETE x)
            RETURN size(ms) AS n
            """,
            user_id=scope.user_id,
            app_id=scope.app_id,
            op=OutboxOp.DELETE_SCOPE.value,
        )
        return rows[0]["n"] if rows else 0

    # ------------------------------------------------------------------ read

    async def get_memories(self, memory_ids: list[str]) -> dict[str, Memory]:
        if not memory_ids:
            return {}
        rows = await self._run(
            """
            MATCH (m:Memory) WHERE m.id IN $ids
            OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
            RETURN m, collect(e {.name, type: e.type}) AS entities
            """,
            ids=memory_ids,
        )
        return {r["m"]["id"]: _to_memory(r["m"], r["entities"]) for r in rows}

    async def fulltext_search(
        self,
        scope: Scope,
        query: str,
        limit: int,
        moment: datetime | None = None,
        include_history: bool = False,
    ) -> list[tuple[str, float]]:
        """BM25 over memory text - catches exact tokens embeddings blur away."""
        escaped = _escape_lucene(query)
        if not escaped:
            return []
        rows = await self._run(
            f"""
            CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX}', $q, {{limit: $limit}})
            YIELD node AS m, score
            WHERE {_SCOPE_FILTER} {_TIME_FILTER}
            RETURN m.id AS id, score
            """,
            q=escaped,
            limit=limit * 3,
            user_id=scope.user_id,
            app_id=scope.app_id,
            **self._time_params(moment, include_history),
        )
        return [(r["id"], r["score"]) for r in rows[:limit]]

    async def expand_by_entities(
        self,
        scope: Scope,
        memory_ids: list[str],
        hops: int,
        limit: int,
        moment: datetime | None = None,
    ) -> list[str]:
        """Pull memories that share entities with the seed set."""
        if not memory_ids or hops < 1:
            return []
        rows = await self._run(
            f"""
            MATCH (seed:Memory)-[:MENTIONS]->(e:Entity)
            WHERE seed.id IN $ids AND e.user_id = $user_id AND e.app_id = $app_id
            MATCH (e)-[:RELATED*0..{max(0, hops - 1)}]-(near:Entity)
            MATCH (near)<-[:MENTIONS]-(m:Memory)
            WHERE {_SCOPE_FILTER} AND NOT m.id IN $ids {_TIME_FILTER}
            RETURN m.id AS id, count(DISTINCT near) AS shared,
                   max(m.recorded_at) AS newest
            ORDER BY shared DESC, newest DESC
            LIMIT $limit
            """,
            ids=memory_ids,
            user_id=scope.user_id,
            app_id=scope.app_id,
            limit=limit,
            **self._time_params(moment, False),
        )
        return [r["id"] for r in rows]

    async def facts_about(
        self, scope: Scope, subject_key: str, predicate: str, moment: datetime | None = None
    ) -> list[Memory]:
        """Currently-believed facts with the same subject and predicate.

        This is the structured reconciliation shortlist: exact, index-backed,
        and free of embedding calls.
        """
        rows = await self._run(
            f"""
            MATCH (m:Memory {{subject_key: $subject_key, predicate: $predicate}})
            WHERE {_SCOPE_FILTER} {_TIME_FILTER}
            RETURN m
            ORDER BY m.recorded_at DESC
            """,
            subject_key=subject_key,
            predicate=predicate,
            user_id=scope.user_id,
            app_id=scope.app_id,
            **self._time_params(moment, False),
        )
        return [_to_memory(r["m"]) for r in rows]

    async def history(
        self, scope: Scope, subject_key: str, predicate: str | None = None
    ) -> list[Memory]:
        """Every value this attribute has ever held, oldest first."""
        rows = await self._run(
            """
            MATCH (m:Memory {subject_key: $subject_key, user_id: $user_id, app_id: $app_id})
            WHERE $predicate IS NULL OR m.predicate = $predicate
            RETURN m
            ORDER BY coalesce(m.valid_from, m.recorded_at) ASC
            """,
            subject_key=subject_key,
            predicate=predicate,
            user_id=scope.user_id,
            app_id=scope.app_id,
        )
        return [_to_memory(r["m"]) for r in rows]

    async def list_memories(
        self,
        scope: Scope,
        kind: str | None = None,
        limit: int = 50,
        moment: datetime | None = None,
        include_history: bool = False,
    ) -> list[Memory]:
        rows = await self._run(
            f"""
            MATCH (m:Memory)
            WHERE {_SCOPE_FILTER} AND ($kind IS NULL OR m.kind = $kind) {_TIME_FILTER}
            OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
            RETURN m, collect(e {{.name, type: e.type}}) AS entities
            ORDER BY m.recorded_at DESC
            LIMIT $limit
            """,
            user_id=scope.user_id,
            app_id=scope.app_id,
            kind=kind,
            limit=limit,
            **self._time_params(moment, include_history),
        )
        return [_to_memory(r["m"], r["entities"]) for r in rows]

    async def neighbors(self, scope: Scope, entity_name: str, hops: int = 2) -> list[dict]:
        depth = max(1, min(hops, 3))
        rows = await self._run(
            f"""
            MATCH (e:Entity {{user_id: $user_id, app_id: $app_id}})
            WHERE toLower(e.canonical_name) = toLower($name)
               OR $normalized IN coalesce(e.aliases, [])
            MATCH path = (e)-[:RELATED*1..{depth}]-(other:Entity)
            WHERE ALL(rel IN relationships(path) WHERE rel.valid_to IS NULL)
            RETURN DISTINCT other.canonical_name AS name, other.type AS type,
                   [rel IN relationships(path) | rel.predicate] AS via,
                   length(path) AS hops
            ORDER BY hops ASC, name ASC
            LIMIT 50
            """,
            user_id=scope.user_id,
            app_id=scope.app_id,
            name=entity_name,
            normalized=entity_name.strip().lower(),
        )
        return rows

    async def stats(self, scope: Scope) -> dict:
        rows = await self._run(
            """
            OPTIONAL MATCH (m:Memory {user_id: $user_id, app_id: $app_id})
            WITH collect(m) AS ms
            OPTIONAL MATCH (e:Entity {user_id: $user_id, app_id: $app_id})
            WITH ms, count(DISTINCT e) AS entities
            OPTIONAL MATCH (ep:Episode {user_id: $user_id, app_id: $app_id})
            RETURN size(ms) AS total,
                   size([x IN ms WHERE x.invalid_at IS NULL]) AS believed,
                   size([x IN ms WHERE x.needs_verification]) AS needs_verification,
                   entities,
                   count(DISTINCT ep) AS episodes
            """,
            user_id=scope.user_id,
            app_id=scope.app_id,
        )
        return rows[0] if rows else {}

    # ---------------------------------------------------------------- outbox

    async def claim_outbox(self, limit: int = 100) -> list[dict]:
        rows = await self._run(
            """
            MATCH (o:Outbox) WHERE o.status = 'pending'
            WITH o ORDER BY o.created_at ASC LIMIT $limit
            SET o.status = 'inflight', o.attempts = coalesce(o.attempts, 0) + 1,
                o.claimed_at = datetime()
            RETURN o {.*} AS op
            """,
            limit=limit,
        )
        return [r["op"] for r in rows]

    async def complete_outbox(self, outbox_id: str) -> None:
        await self._run("MATCH (o:Outbox {id: $id}) DETACH DELETE o", id=outbox_id)

    async def fail_outbox(self, outbox_id: str, error: str, max_attempts: int) -> None:
        """Back to pending for another go, or parked as dead for inspection."""
        await self._run(
            """
            MATCH (o:Outbox {id: $id})
            SET o.last_error = $error,
                o.status = CASE WHEN o.attempts >= $max THEN 'dead' ELSE 'pending' END
            """,
            id=outbox_id,
            error=error[:500],
            max=max_attempts,
        )

    async def outbox_depth(self) -> dict:
        rows = await self._run(
            """
            MATCH (o:Outbox)
            RETURN o.status AS status, count(*) AS n
            """
        )
        return {r["status"]: r["n"] for r in rows}


_LUCENE_SPECIAL = set('+-&|!(){}[]^"~*?:\\/')


def _escape_lucene(query: str) -> str:
    cleaned = "".join("\\" + c if c in _LUCENE_SPECIAL else c for c in query)
    terms = [t for t in cleaned.split() if t]
    return " OR ".join(terms)
