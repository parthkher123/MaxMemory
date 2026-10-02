"""Shared fixtures.

Integration tests run against real Neo4j and Qdrant with the hash embedder and
no extraction model, so they exercise storage, retrieval, scoping, temporal
logic and reconciliation without spending API tokens. Anything that genuinely
needs a model is tested through a stub extractor instead.
"""

from __future__ import annotations

import os
import uuid

import pytest

from memory_mcp.config import Settings
from memory_mcp.embeddings import HashEmbedder
from memory_mcp.extraction import Extractor, GateResult
from memory_mcp.graph_store import Neo4jGraphStore
from memory_mcp.memory import MemoryLayer
from memory_mcp.models import ContentCategory, Fact, Scope
from memory_mcp.vector_store import QdrantVectorStore


def make_settings(**overrides) -> Settings:
    base = dict(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "memorypassword"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=None,
        qdrant_collection="memories_test",
        qdrant_hnsw_ef=128,
        embedding_provider="hash",
        embedding_model="hash-256",
        embedding_dim=256,
        embedding_base_url="https://api.openai.com/v1",
        voyage_api_key=None,
        openai_api_key=None,
        llm_provider="none",
        llm_api_key=None,
        llm_base_url="https://api.openai.com/v1",
        extraction_model="claude-opus-5",
        gate_model="claude-haiku-4-5",
        ingest_async=False,
        worker_poll_seconds=0.05,
        outbox_max_attempts=3,
        episode_max_attempts=2,
        claim_timeout_seconds=0.0,
        recall_limit=8,
        rrf_k=60,
        graph_expansion_hops=1,
        recency_half_life_days=180.0,
        dedupe_threshold=0.95,
        contradiction_threshold=0.84,
        entity_match_threshold=0.93,
        entity_ambiguous_threshold=0.85,
        verify_threshold=0.5,
        default_user_id="test",
        default_app_id="test-app",
        default_agent_id="test-agent",
    )
    base.update(overrides)
    return Settings(**base)


class StubExtractor(Extractor):
    """Deterministic stand-in for the three model calls.

    Facts are queued by the test; the gate passes unless the safety prefilter
    trips; contradiction is answered from a set of known-conflicting pairs.
    """

    def __init__(self, cfg: Settings) -> None:
        super().__init__(cfg)
        self.queued: list[list[Fact]] = []
        self.contradicting: set[tuple[str, str]] = set()
        self.same_entities: set[tuple[str, str]] = set()
        self.gate_calls = 0
        self.extract_calls = 0

    def queue(self, facts: list[Fact]) -> None:
        self.queued.append(facts)

    async def gate(self, text: str) -> GateResult:
        self.gate_calls += 1
        from memory_mcp.safety import injection_risk

        risk = injection_risk(text)
        if risk:
            return GateResult(
                should_remember=False,
                category=ContentCategory.NO_MEMORY,
                injection_risk=True,
                reason=risk,
            )
        return GateResult(should_remember=True, category=ContentCategory.PERSONAL_FACT)

    async def extract(self, text: str, *, user_id: str, now=None) -> list[Fact]:
        self.extract_calls += 1
        if self.queued:
            return self.queued.pop(0)
        return await super().extract(text, user_id=user_id, now=now)

    async def contradicts(self, existing: str, incoming: str) -> bool:
        return (existing, incoming) in self.contradicting

    async def same_entity(self, a: str, b: str, entity_type: str) -> bool:
        return (a, b) in self.same_entities or (b, a) in self.same_entities


@pytest.fixture
def cfg() -> Settings:
    return make_settings()


@pytest.fixture
async def layer(cfg: Settings):
    extractor = StubExtractor(cfg)
    candidate = MemoryLayer(
        cfg=cfg,
        graph=Neo4jGraphStore(cfg),
        vectors=QdrantVectorStore(cfg),
        embedder=HashEmbedder(cfg.embedding_dim),
        extractor=extractor,
    )
    candidate.stub = extractor  # type: ignore[attr-defined]
    try:
        await candidate.setup()
    except Exception as exc:  # stores not running
        await candidate.close()
        pytest.skip(f"stores unavailable: {exc}")
    yield candidate
    await candidate.close()


@pytest.fixture
def scope() -> Scope:
    return Scope(user_id=f"test-{uuid.uuid4().hex[:8]}", app_id="app-a", agent_id="agent-1")
