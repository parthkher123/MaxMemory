"""Domain types shared across the layer."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from .ontology import SourceType, source_trust


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid4().hex


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


class Scope(BaseModel):
    """Who this memory belongs to, and which app and agent may see it.

    user_id alone leaks: the same person using two of your products would share
    one memory pool. Every read and write filters on (user_id, app_id); agent_id
    is recorded for audit and available as an optional filter.
    """

    user_id: str
    app_id: str = "default"
    agent_id: str = "default"

    model_config = {"frozen": True}

    @property
    def key(self) -> str:
        return f"{self.user_id}|{self.app_id}"

    def as_params(self) -> dict:
        return {"user_id": self.user_id, "app_id": self.app_id, "agent_id": self.agent_id}


class MemoryKind(str, Enum):
    SEMANTIC = "semantic"      # durable facts: "works at Anthropic"
    EPISODIC = "episodic"      # things that happened: "shipped v2 on Tuesday"
    PREFERENCE = "preference"  # tastes and rules: "prefers dark mode"
    PROCEDURAL = "procedural"  # how-to: "deploys with make ship"


class ContentCategory(str, Enum):
    """What the extraction gate decided an episode contains."""

    NO_MEMORY = "no_memory"
    PERSONAL_FACT = "personal_fact"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    EVENT = "event"
    GOAL = "goal"
    TASK = "task"
    LOCATION = "location"
    WORK = "work"
    TEMPORAL_FACT = "temporal_fact"


class EpisodeStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    SKIPPED = "skipped"   # gate said there was nothing worth remembering
    BLOCKED = "blocked"   # safety classifier rejected it
    FAILED = "failed"   # gave up after EPISODE_MAX_ATTEMPTS


class OutboxOp(str, Enum):
    INDEX = "index"
    SET_VALIDITY = "set_validity"
    DELETE = "delete"
    DELETE_SCOPE = "delete_scope"


class Confidence(BaseModel):
    """Confidence as a product of independent signals, not one model's guess."""

    extraction: float = 1.0
    source: float = 1.0
    temporal: float = 1.0
    entity: float = 1.0

    @property
    def final(self) -> float:
        return round(self.extraction * self.source * self.temporal * self.entity, 4)


class Entity(BaseModel):
    name: str
    type: str = "thing"

    @property
    def key(self) -> str:
        """Normalized identity within one scope."""
        return f"{self.type.lower().strip()}:{normalize_name(self.name)}"


def normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name.strip().lower())
    return re.sub(r"\s+", " ", cleaned)


class ResolvedEntity(BaseModel):
    """An entity after resolution against the user's canonical entity set."""

    id: str
    key: str
    canonical_name: str
    type: str
    mention: str                  # what the text actually said
    confidence: float = 1.0
    created: bool = False


class Relation(BaseModel):
    subject: str
    predicate: str
    object: str


class Fact(BaseModel):
    """One atomic, self-contained statement - the unit we store."""

    text: str
    kind: MemoryKind = MemoryKind.SEMANTIC
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    # Structured triple, when the fact maps onto the ontology.
    predicate: str | None = None
    subject: str | None = None
    object: str | None = None
    # World time: when the fact became / stopped being true.
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class Episode(BaseModel):
    """Immutable raw input. Facts are derived from these and can be rebuilt."""

    id: str = Field(default_factory=new_id)
    user_id: str
    app_id: str = "default"
    agent_id: str = "default"
    raw_text: str
    source: str | None = None
    source_type: SourceType = SourceType.USER_STATEMENT
    source_id: str | None = None
    content_hash: str = ""
    ingested_at: datetime = Field(default_factory=utcnow)
    status: EpisodeStatus = EpisodeStatus.PENDING
    category: ContentCategory | None = None
    error: str | None = None
    attempts: int = 0
    metadata: dict = Field(default_factory=dict)

    def with_hash(self) -> Episode:
        self.content_hash = content_hash(self.raw_text, self.user_id, self.app_id)
        return self


def content_hash(text: str, user_id: str, app_id: str) -> str:
    """Idempotency key: the same text in the same scope is the same episode."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(f"{normalized}|{user_id}|{app_id}".encode()).hexdigest()


class Memory(BaseModel):
    id: str = Field(default_factory=new_id)
    user_id: str
    app_id: str = "default"
    agent_id: str = "default"
    text: str
    kind: MemoryKind = MemoryKind.SEMANTIC

    # Structured form, when the fact fits the ontology.
    predicate: str | None = None
    subject_key: str | None = None
    object_key: str | None = None
    object_value: str | None = None

    importance: float = 0.5
    confidence: Confidence = Field(default_factory=Confidence)
    needs_verification: bool = False

    # Provenance
    episode_id: str | None = None
    source: str | None = None
    source_type: SourceType = SourceType.USER_STATEMENT
    source_id: str | None = None
    embedding_model: str | None = None
    extraction_model: str | None = None

    # World time: when the fact was true.
    valid_from: datetime = Field(default_factory=utcnow)
    valid_to: datetime | None = None
    # System time: when we learned it, and when we stopped believing it.
    recorded_at: datetime = Field(default_factory=utcnow)
    invalid_at: datetime | None = None
    superseded_by: str | None = None

    reinforced: int = 0
    entities: list[Entity] = Field(default_factory=list)

    @property
    def is_believed(self) -> bool:
        """Do we still hold this record to be a correct statement?"""
        return self.invalid_at is None

    def is_true_at(self, moment: datetime | None = None) -> bool:
        """Was the fact true in the world at `moment`?"""
        moment = as_utc(moment) or utcnow()
        start = as_utc(self.valid_from)
        end = as_utc(self.valid_to)
        if start and moment < start:
            return False
        if end and moment >= end:
            return False
        return True

    @property
    def is_current(self) -> bool:
        return self.is_believed and self.is_true_at()

    @property
    def trust(self) -> float:
        return source_trust(self.source_type)


class ScoredMemory(BaseModel):
    memory: Memory
    score: float
    retrieved_by: list[str] = Field(default_factory=list)


class IngestResult(BaseModel):
    """What happened to one ingest call."""

    episode_id: str
    status: EpisodeStatus
    duplicate_episode: bool = False
    category: ContentCategory | None = None
    created: list[Memory] = Field(default_factory=list)
    reinforced: list[str] = Field(default_factory=list)
    superseded: list[str] = Field(default_factory=list)
    reason: str | None = None

    def summary(self) -> str:
        if self.duplicate_episode:
            return f"already ingested (episode {self.episode_id})"
        if self.status is EpisodeStatus.BLOCKED:
            return f"blocked: {self.reason or 'failed safety check'}"
        if self.status is EpisodeStatus.SKIPPED:
            return f"nothing worth remembering ({self.reason or 'no_memory'})"
        if self.status is EpisodeStatus.PENDING:
            return f"queued (episode {self.episode_id})"
        return (
            f"{len(self.created)} new, {len(self.reinforced)} reinforced, "
            f"{len(self.superseded)} superseded"
        )
