"""The memory layer - orchestrates the pipeline and both stores.

    ingest -> episode -> gate -> extract -> resolve -> reconcile -> stores

Every stage is skippable or degradable: no API key falls back to verbatim
storage, a failed gate fails open, a failed Qdrant write is retried by the
outbox. What is never skipped is the episode: raw input is recorded before
anything else can go wrong, so memory can always be rebuilt.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from datetime import datetime

from .config import Settings, settings
from .embeddings import Embedder, build_embedder
from .entities import EntityResolver
from .extraction import Extractor
from .graph_store import Neo4jGraphStore
from .models import (
    Confidence,
    Entity,
    Episode,
    EpisodeStatus,
    Fact,
    IngestResult,
    Memory,
    Relation,
    ResolvedEntity,
    Scope,
    ScoredMemory,
    as_utc,
    utcnow,
)
from .ontology import (
    SourceType,
    is_single_valued,
    normalize_predicate,
    source_trust,
    spec_for,
)
from .outbox import OutboxDrainer
from .vector_store import QdrantVectorStore

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9']+")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _jaccard(a: str, b: str) -> float:
    """Lexical overlap - a second opinion on 'is this the same fact'."""
    ta, tb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class _Outcome:
    """What reconciliation decided about one incoming fact."""

    __slots__ = ("churn", "duplicate_of", "superseded")

    def __init__(self) -> None:
        self.duplicate_of: str | None = None
        self.superseded: list[str] = []
        self.churn: bool = False  # it replaced something: slightly less certain


class MemoryLayer:
    def __init__(
        self,
        cfg: Settings | None = None,
        graph: Neo4jGraphStore | None = None,
        vectors: QdrantVectorStore | None = None,
        embedder: Embedder | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        self._cfg = cfg or settings()
        self._graph = graph or Neo4jGraphStore(self._cfg)
        self._vectors = vectors or QdrantVectorStore(self._cfg)
        self._embedder = embedder or build_embedder(self._cfg)
        self._extractor = extractor or Extractor(self._cfg)
        self._entities = EntityResolver(self._graph, self._embedder, self._extractor, self._cfg)
        self._outbox = OutboxDrainer(self._graph, self._vectors, self._cfg)
        self._stop = asyncio.Event()
        self._workers: list[asyncio.Task] = []

    async def setup(self) -> None:
        await asyncio.gather(self._graph.ensure_schema(), self._vectors.ensure_schema())

    async def close(self) -> None:
        await self.stop_workers()
        await asyncio.gather(self._graph.close(), self._vectors.close())

    # --------------------------------------------------------------- workers

    def start_workers(self) -> None:
        """Background loops: drain the outbox, process queued episodes."""
        if self._workers:
            return
        self._stop.clear()
        self._workers = [
            asyncio.create_task(self._outbox.run_forever(self._stop)),
            asyncio.create_task(self._episode_loop()),
        ]

    async def stop_workers(self) -> None:
        self._stop.set()
        for task in self._workers:
            task.cancel()
        for task in self._workers:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("worker raised during shutdown")
        self._workers = []

    async def _episode_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.reclaim_stale()
                processed = await self.process_pending(limit=5)
            except Exception:
                log.exception("episode worker cycle failed")
                processed = 0
            if processed == 0:
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self._cfg.worker_poll_seconds
                    )
                except TimeoutError:
                    pass

    async def process_pending(self, limit: int = 5) -> int:
        episodes = await self._graph.claim_episodes(limit)
        for episode in episodes:
            try:
                await self.process_episode(episode)
            except Exception as exc:
                log.exception("episode %s failed", episode.id)
                await self._graph.release_episode(
                    episode.id, str(exc), self._cfg.episode_max_attempts
                )
        return len(episodes)

    async def reclaim_stale(self) -> dict:
        """Release work held by workers that died mid-flight."""
        return await self._graph.reclaim_stale(self._cfg.claim_timeout_seconds)

    async def drain_outbox(self, limit: int = 200) -> int:
        return await self._outbox.drain_once(limit)

    # ---------------------------------------------------------------- ingest

    async def remember(
        self,
        text: str,
        scope: Scope,
        source: str | None = None,
        source_type: SourceType = SourceType.USER_STATEMENT,
        source_id: str | None = None,
        wait: bool | None = None,
    ) -> IngestResult:
        """Record raw input, then derive facts from it.

        The episode is written synchronously and is what makes the call
        idempotent; extraction happens inline or on a worker depending on
        INGEST_ASYNC.
        """
        episode = Episode(
            user_id=scope.user_id,
            app_id=scope.app_id,
            agent_id=scope.agent_id,
            raw_text=text,
            source=source,
            source_type=source_type,
            source_id=source_id,
        ).with_hash()

        stored, created = await self._graph.create_episode(episode)
        if not created:
            # Same text, same scope, already seen: return what it produced.
            return IngestResult(
                episode_id=stored.id,
                status=stored.status,
                duplicate_episode=True,
                category=stored.category,
                created=await self._graph.episode_memories(stored.id),
            )

        synchronous = (not self._cfg.ingest_async) if wait is None else wait
        if synchronous:
            # Mark it in flight so a background worker cannot claim the same
            # episode and extract it a second time.
            await self._graph.set_episode_status(stored.id, EpisodeStatus.PROCESSING)
            try:
                return await self.process_episode(stored)
            except Exception as exc:
                # Don't strand the episode or hand the caller a traceback: it
                # goes back on the queue and a worker retries it.
                log.exception("inline processing of episode %s failed", stored.id)
                status = await self._graph.release_episode(
                    stored.id, str(exc), self._cfg.episode_max_attempts
                )
                return IngestResult(
                    episode_id=stored.id,
                    status=status,
                    reason=str(exc),
                )

        return IngestResult(episode_id=stored.id, status=EpisodeStatus.PENDING)

    async def process_episode(self, episode: Episode) -> IngestResult:
        """Gate, extract, reconcile and store one episode."""
        scope = Scope(
            user_id=episode.user_id, app_id=episode.app_id, agent_id=episode.agent_id
        )
        result = IngestResult(episode_id=episode.id, status=EpisodeStatus.DONE)

        verdict = await self._extractor.gate(episode.raw_text)
        result.category = verdict.category
        if verdict.injection_risk:
            # Never persist text that reads as an instruction to a model.
            result.status = EpisodeStatus.BLOCKED
            result.reason = verdict.reason
            await self._graph.set_episode_status(
                episode.id, EpisodeStatus.BLOCKED, verdict.category, verdict.reason
            )
            return result
        if not verdict.should_remember:
            result.status = EpisodeStatus.SKIPPED
            result.reason = verdict.reason or verdict.category.value
            await self._graph.set_episode_status(
                episode.id, EpisodeStatus.SKIPPED, verdict.category, verdict.reason
            )
            return result

        # Reconciliation compares against the index, so anything still queued
        # must land first - otherwise a fact written seconds ago is invisible
        # and we store a duplicate or miss a contradiction.
        await self.drain_outbox()

        facts = await self._extractor.extract(episode.raw_text, user_id=scope.user_id)
        if not facts:
            result.status = EpisodeStatus.SKIPPED
            result.reason = "extraction produced no facts"
            await self._graph.set_episode_status(
                episode.id, EpisodeStatus.SKIPPED, verdict.category
            )
            return result

        vectors = await self._embedder.embed([f.text for f in facts])
        for fact, vector in zip(facts, vectors):
            await self._store_fact(fact, vector, scope, episode, result)

        await self._graph.set_episode_status(episode.id, EpisodeStatus.DONE, verdict.category)
        return result

    async def _store_fact(
        self,
        fact: Fact,
        vector: list[float],
        scope: Scope,
        episode: Episode,
        result: IngestResult,
    ) -> None:
        resolved = await self._entities.resolve(scope, fact.entities)
        by_mention = {r.mention.strip().lower(): r for r in resolved}
        # Normalize here too: the layer must not trust its caller to have done it.
        fact.predicate = normalize_predicate(fact.predicate)

        subject = by_mention.get((fact.subject or "").strip().lower())
        obj = by_mention.get((fact.object or "").strip().lower())
        subject_key = subject.key if subject else None
        object_key = obj.key if obj else None

        outcome = await self._reconcile(
            fact, vector, scope, subject_key, object_key
        )
        if outcome.duplicate_of:
            await self._graph.reinforce(outcome.duplicate_of)
            result.reinforced.append(outcome.duplicate_of)
            return

        confidence = self._confidence(fact, resolved, episode.source_type, outcome.churn)
        memory = Memory(
            user_id=scope.user_id,
            app_id=scope.app_id,
            agent_id=scope.agent_id,
            text=fact.text,
            kind=fact.kind,
            predicate=fact.predicate,
            subject_key=subject_key,
            object_key=object_key,
            object_value=(fact.object or None) if not object_key else obj.canonical_name,
            importance=fact.importance,
            confidence=confidence,
            needs_verification=confidence.final < self._cfg.verify_threshold,
            episode_id=episode.id,
            source=episode.source,
            source_type=episode.source_type,
            source_id=episode.source_id,
            embedding_model=self._cfg.embedding_model,
            extraction_model=self._cfg.extraction_model,
            valid_from=fact.valid_from or episode.ingested_at,
            valid_to=fact.valid_to,
            entities=fact.entities,
        )

        await self._graph.write_memory(
            memory,
            entity_ids=[r.id for r in resolved],
            relations=_relation_params(by_mention, fact),
            vector=vector,
        )

        # Close out what this fact replaced: the old value stopped being true
        # when the new one started, not when we happened to hear about it.
        for stale_id in outcome.superseded:
            await self._graph.supersede(stale_id, memory.id, valid_to=memory.valid_from)
            result.superseded.append(stale_id)

        result.created.append(memory)

    # -------------------------------------------------------- reconciliation

    async def _reconcile(
        self,
        fact: Fact,
        vector: list[float],
        scope: Scope,
        subject_key: str | None,
        object_key: str | None,
    ) -> _Outcome:
        """Decide whether an incoming fact is new, a repeat, or an update.

        Structured first: if the fact has a subject and an ontology predicate,
        an exact index lookup answers it with no embeddings and no model call.
        Only unstructured facts fall through to similarity and a judge.
        """
        outcome = _Outcome()

        if fact.predicate and subject_key:
            existing = await self._graph.facts_about(scope, subject_key, fact.predicate)
            for candidate in existing:
                same_value = (
                    object_key is not None and candidate.object_key == object_key
                ) or (
                    object_key is None
                    and candidate.object_value
                    and fact.object
                    and candidate.object_value.strip().lower() == fact.object.strip().lower()
                )
                if same_value:
                    outcome.duplicate_of = candidate.id
                    return outcome
                if is_single_valued(fact.predicate) and _temporally_conflicting(fact, candidate):
                    outcome.superseded.append(candidate.id)
                    outcome.churn = True
            return outcome

        # Unstructured: ask the index for near neighbours. The scores come back
        # with the hits, so stored memories are never re-embedded.
        hits = await self._vectors.neighbours_of(scope=scope, vector=vector, limit=20)
        if not hits:
            return outcome
        memories = await self._graph.get_memories([h.memory_id for h in hits])

        for hit in hits:
            candidate = memories.get(hit.memory_id)
            if candidate is None or not candidate.is_current:
                continue
            lexical = _jaccard(fact.text, candidate.text)
            # Two independent signals agreeing beats one strong signal alone.
            if hit.score >= self._cfg.dedupe_threshold or (
                hit.score >= self._cfg.dedupe_threshold - 0.05 and lexical >= 0.6
            ):
                outcome.duplicate_of = candidate.id
                return outcome
            if hit.score >= self._cfg.contradiction_threshold:
                if await self._extractor.contradicts(candidate.text, fact.text):
                    outcome.superseded.append(candidate.id)
                    outcome.churn = True
        return outcome

    def _confidence(
        self,
        fact: Fact,
        resolved: list[ResolvedEntity],
        source_type: SourceType,
        churn: bool,
    ) -> Confidence:
        """Confidence as a product of independent signals.

        A model's own certainty is only one of them, and not the one that
        matters most - an inferred claim should never outrank a stated one.
        """
        temporal = 1.0
        spec = spec_for(fact.predicate)
        if spec and spec.temporal and fact.valid_from is None:
            temporal *= 0.95  # a dated kind of fact arriving undated
        if churn:
            temporal *= 0.9   # it displaced something; one of them is wrong

        return Confidence(
            extraction=fact.extraction_confidence,
            source=source_trust(source_type),
            temporal=round(temporal, 4),
            entity=min((r.confidence for r in resolved), default=1.0),
        )

    # ------------------------------------------------------------- read path

    async def recall(
        self,
        query: str,
        scope: Scope,
        limit: int | None = None,
        moment: datetime | None = None,
        include_history: bool = False,
        kinds: list[str] | None = None,
    ) -> list[ScoredMemory]:
        """Hybrid retrieval. `moment` asks what was true then, not what we knew."""
        limit = limit or self._cfg.recall_limit
        fetch = limit * 4

        query_vector = (await self._embedder.embed([query], query=True))[0]
        vector_hits, text_hits = await asyncio.gather(
            self._vectors.search(
                scope=scope,
                vector=query_vector,
                limit=fetch,
                include_history=include_history or moment is not None,
                kinds=kinds,
            ),
            self._graph.fulltext_search(
                scope, query, fetch, moment=moment, include_history=include_history
            ),
        )

        # Reciprocal rank fusion: combines rankings without needing the two
        # score scales (cosine, BM25) to be comparable.
        k = self._cfg.rrf_k
        fused: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        for rank, hit in enumerate(vector_hits):
            fused[hit.memory_id] = fused.get(hit.memory_id, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(hit.memory_id, []).append("vector")
        for rank, (memory_id, _score) in enumerate(text_hits):
            fused[memory_id] = fused.get(memory_id, 0.0) + 1.0 / (k + rank + 1)
            sources.setdefault(memory_id, []).append("fulltext")

        seeds = sorted(fused, key=fused.get, reverse=True)[:5]
        if seeds and not include_history:
            expanded = await self._graph.expand_by_entities(
                scope, seeds, self._cfg.graph_expansion_hops, limit, moment=moment
            )
            floor = min(fused.values()) if fused else 1.0 / (k + 1)
            for rank, memory_id in enumerate(expanded):
                if memory_id not in fused:
                    fused[memory_id] = floor * 0.5 / (rank + 1)
                    sources.setdefault(memory_id, []).append("graph")

        if not fused:
            return []

        memories = await self._graph.get_memories(list(fused))
        now = utcnow()
        at = moment or now
        scored = []
        for memory_id, base in fused.items():
            memory = memories.get(memory_id)
            if memory is None:
                continue
            if not include_history:
                if not memory.is_believed or not memory.is_true_at(at):
                    continue
            scored.append(
                ScoredMemory(
                    memory=memory,
                    score=base * self._weight(memory, now),
                    retrieved_by=sources.get(memory_id, []),
                )
            )

        if kinds:
            # The filter reaches Qdrant natively, but fulltext and graph
            # expansion feed the same result set - apply it to all three.
            wanted = set(kinds)
            scored = [s for s in scored if s.memory.kind.value in wanted]

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]

    def _weight(self, memory: Memory, now: datetime) -> float:
        """Importance and trust lift; age gently sinks, never to zero.

        Old facts are often the most stable ones ("born in Pune"), so recency
        is a nudge, not a cliff: the decay term only spans 0.7-1.0.
        """
        recorded = as_utc(memory.recorded_at) or now
        age_days = max((now - recorded).total_seconds() / 86400.0, 0.0)
        decay = 0.5 ** (age_days / self._cfg.recency_half_life_days)
        return (
            (1.0 + 0.2 * memory.importance)
            * (0.7 + 0.3 * decay)
            * (0.6 + 0.4 * memory.confidence.final)
        )

    # --------------------------------------------------------------- utility

    async def forget(
        self,
        scope: Scope,
        memory_ids: list[str] | None = None,
        query: str | None = None,
        hard: bool = False,
    ) -> list[str]:
        """Invalidate by id, or by what the memories say."""
        ids = list(memory_ids or [])
        if query:
            hits = await self.recall(query, scope, limit=5)
            ids.extend(h.memory.id for h in hits)
        ids = list(dict.fromkeys(ids))
        if not ids:
            return []

        # Only touch memories inside this scope, whatever ids were passed in.
        owned = await self._graph.get_memories(ids)
        ids = [
            mid
            for mid, memory in owned.items()
            if memory.user_id == scope.user_id and memory.app_id == scope.app_id
        ]
        if not ids:
            return []

        if hard:
            await self._graph.hard_delete(ids)
        else:
            for memory_id in ids:
                await self._graph.invalidate(memory_id)
        await self.drain_outbox()
        return ids

    async def forget_scope(self, scope: Scope) -> int:
        """Right-to-erasure: drop everything for one (user, app), both stores."""
        count = await self._graph.delete_scope(scope)
        await self.drain_outbox()
        return count

    async def timeline(self, scope: Scope, limit: int = 20) -> list[Memory]:
        return await self._graph.list_memories(scope, kind="episodic", limit=limit)

    async def history(self, scope: Scope, subject: str, predicate: str | None = None):
        """Every value an attribute has held, with the dates it held them."""
        subject_key = Entity(name=subject, type="person").key
        rows = await self._graph.history(scope, subject_key, predicate)
        if not rows:
            for entity_type in ("org", "project", "place", "thing"):
                subject_key = Entity(name=subject, type=entity_type).key
                rows = await self._graph.history(scope, subject_key, predicate)
                if rows:
                    break
        return rows

    async def relate(self, scope: Scope, entity: str, hops: int = 2) -> list[dict]:
        return await self._graph.neighbors(scope, entity, hops)

    async def profile(self, scope: Scope, limit: int = 12) -> list[Memory]:
        """A compact digest of currently-true facts, for prompt injection."""
        memories = await self._graph.list_memories(scope, limit=100)
        ranked = sorted(
            memories,
            key=lambda m: (
                m.importance * m.confidence.final * m.trust + 0.05 * m.reinforced
            ),
            reverse=True,
        )
        return ranked[:limit]

    async def stats(self, scope: Scope) -> dict:
        stats = await self._graph.stats(scope)
        stats["outbox"] = await self._graph.outbox_depth()
        return stats

    async def job_status(self, episode_id: str) -> dict:
        episode = await self._graph.get_episode(episode_id)
        if episode is None:
            return {"episode_id": episode_id, "status": "unknown"}
        memories = await self._graph.episode_memories(episode_id)
        return {
            "episode_id": episode.id,
            "status": episode.status.value,
            "category": episode.category.value if episode.category else None,
            "error": episode.error,
            "memories": [m.text for m in memories],
        }


def _temporally_conflicting(fact: Fact, candidate: Memory) -> bool:
    """Do two single-valued facts actually overlap in world time?

    "Worked at Acme until 2023" and "works at Beta since 2023" are sequential,
    not contradictory - only overlapping claims supersede.
    """
    new_from = as_utc(fact.valid_from) or utcnow()
    new_to = as_utc(fact.valid_to)
    old_from = as_utc(candidate.valid_from)
    old_to = as_utc(candidate.valid_to)

    if old_to is not None and old_to <= new_from:
        return False  # the old fact had already ended
    if new_to is not None and old_from is not None and new_to <= old_from:
        return False  # the new fact ends before the old one starts
    return True


def _relation_params(by_mention: dict[str, ResolvedEntity], fact: Fact) -> list[dict]:
    """Map extracted relations onto resolved entity ids, dropping unresolvable ones."""
    relations: list[Relation] = list(fact.relations)
    if fact.predicate and fact.subject and fact.object:
        relations.append(
            Relation(subject=fact.subject, predicate=fact.predicate, object=fact.object)
        )

    out = []
    seen = set()
    for rel in relations:
        subject = by_mention.get(rel.subject.strip().lower())
        obj = by_mention.get(rel.object.strip().lower())
        predicate = normalize_predicate(rel.predicate)
        if not (subject and obj and predicate) or subject.id == obj.id:
            continue
        key = (subject.id, predicate, obj.id)
        if key in seen:
            continue
        seen.add(key)
        out.append({"subject_id": subject.id, "predicate": predicate, "object_id": obj.id})
    return out
