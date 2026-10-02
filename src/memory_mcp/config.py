"""Configuration, loaded once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# Dimensions we know; used when EMBEDDING_DIM is not set explicitly.
_KNOWN_DIMS = {
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3-large": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def _str(key: str, default: str = "") -> str:
    return os.getenv(key) or default


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key) or default)
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key) or default)
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Neo4j - source of truth
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str

    # Qdrant - ANN index
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    qdrant_hnsw_ef: int

    # Embeddings
    embedding_provider: str
    embedding_model: str
    embedding_dim: int
    embedding_base_url: str      # openai provider only: any compatible server
    voyage_api_key: str | None
    openai_api_key: str | None

    # LLM work. The gate and safety classifier run on every ingest, so they
    # get a cheaper model than extraction and contradiction judgement.
    llm_provider: str            # anthropic | openai | none
    llm_api_key: str | None
    llm_base_url: str            # openai provider only: any compatible server
    extraction_model: str
    gate_model: str

    # Ingestion
    ingest_async: bool
    worker_poll_seconds: float
    outbox_max_attempts: int
    episode_max_attempts: int
    claim_timeout_seconds: float

    # Retrieval tuning
    recall_limit: int
    rrf_k: int
    graph_expansion_hops: int
    recency_half_life_days: float

    # Reconciliation
    dedupe_threshold: float
    contradiction_threshold: float
    entity_match_threshold: float      # >= this similarity: same entity
    entity_ambiguous_threshold: float  # between the two: ask the model
    verify_threshold: float            # below this confidence: flag for review

    # Scope defaults
    default_user_id: str
    default_app_id: str
    default_agent_id: str


# Per LLM provider: (key variable it falls back to, extraction model, gate model).
_LLM_DEFAULTS = {
    "anthropic": ("ANTHROPIC_API_KEY", "claude-opus-5", "claude-haiku-4-5"),
    "openai": ("OPENAI_API_KEY", "gpt-4.1", "gpt-4.1-mini"),
    "none": ("", "", ""),
}


@lru_cache(maxsize=1)
def settings() -> Settings:
    llm_provider = _str("LLM_PROVIDER", "anthropic").lower()
    key_var, extraction_default, gate_default = _LLM_DEFAULTS.get(
        llm_provider, _LLM_DEFAULTS["anthropic"]
    )

    provider = _str("EMBEDDING_PROVIDER", "voyage").lower()
    default_model = {
        "voyage": "voyage-3.5",
        "openai": "text-embedding-3-small",
        "hash": "hash-256",
    }.get(provider, "voyage-3.5")
    model = _str("EMBEDDING_MODEL", default_model)

    return Settings(
        neo4j_uri=_str("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=_str("NEO4J_USER", "neo4j"),
        neo4j_password=_str("NEO4J_PASSWORD", "memorypassword"),
        neo4j_database=_str("NEO4J_DATABASE", "neo4j"),
        qdrant_url=_str("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_collection=_str("QDRANT_COLLECTION", "memories"),
        qdrant_hnsw_ef=_int("QDRANT_HNSW_EF", 128),
        embedding_provider=provider,
        embedding_model=model,
        embedding_dim=_int("EMBEDDING_DIM", _KNOWN_DIMS.get(model, 256)),
        embedding_base_url=_str("EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
        voyage_api_key=os.getenv("VOYAGE_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        llm_provider=llm_provider,
        llm_api_key=os.getenv("LLM_API_KEY") or (os.getenv(key_var) if key_var else None) or None,
        llm_base_url=_str("LLM_BASE_URL", "https://api.openai.com/v1"),
        extraction_model=_str("EXTRACTION_MODEL", extraction_default),
        gate_model=_str("GATE_MODEL", gate_default),
        ingest_async=_bool("INGEST_ASYNC", True),
        worker_poll_seconds=_float("WORKER_POLL_SECONDS", 2.0),
        outbox_max_attempts=_int("OUTBOX_MAX_ATTEMPTS", 8),
        episode_max_attempts=_int("EPISODE_MAX_ATTEMPTS", 3),
        claim_timeout_seconds=_float("CLAIM_TIMEOUT_SECONDS", 300.0),
        recall_limit=_int("RECALL_LIMIT", 8),
        rrf_k=_int("RRF_K", 60),
        graph_expansion_hops=_int("GRAPH_EXPANSION_HOPS", 1),
        recency_half_life_days=_float("RECENCY_HALF_LIFE_DAYS", 180.0),
        dedupe_threshold=_float("DEDUPE_THRESHOLD", 0.95),
        contradiction_threshold=_float("CONTRADICTION_THRESHOLD", 0.84),
        entity_match_threshold=_float("ENTITY_MATCH_THRESHOLD", 0.93),
        entity_ambiguous_threshold=_float("ENTITY_AMBIGUOUS_THRESHOLD", 0.85),
        verify_threshold=_float("VERIFY_THRESHOLD", 0.5),
        default_user_id=_str("DEFAULT_USER_ID", "default"),
        default_app_id=_str("DEFAULT_APP_ID", "default"),
        default_agent_id=_str("DEFAULT_AGENT_ID", "default"),
    )
