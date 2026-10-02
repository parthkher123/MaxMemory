"""Resolve entity mentions onto canonical entities.

"Ahmedabad", "Ahmedabad city" and "Ahmedabad, Gujarat" must become one node or
the graph fragments and multi-hop retrieval stops working. The ladder runs
cheapest-first and only reaches the model for genuinely ambiguous cases:

    normalize -> exact key -> alias -> embedding -> LLM -> create new
"""

from __future__ import annotations

import logging
import math
from difflib import SequenceMatcher

from .config import Settings, settings
from .embeddings import Embedder
from .extraction import Extractor
from .models import Entity, ResolvedEntity, Scope, new_id, normalize_name

log = logging.getLogger(__name__)

#: Don't brute-force similarity over an unbounded set.
MAX_CANDIDATES = 500

#: Embedding similarity alone is not enough to merge two entities. A model
#: that thinks "Python" and "Rust" are close is talking about topic, not
#: identity, and silently merging them corrupts the graph irreversibly. Names
#: that agree on nothing must be confirmed before they are merged.
LEXICAL_FLOOR = 0.45


def lexical_agreement(a: str, b: str) -> float:
    """How much two names look alike, with containment counting as full."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class EntityResolver:
    def __init__(self, graph, embedder: Embedder, extractor: Extractor, cfg: Settings | None = None):
        self._graph = graph
        self._embedder = embedder
        self._extractor = extractor
        self._cfg = cfg or settings()

    async def resolve(self, scope: Scope, mentions: list[Entity]) -> list[ResolvedEntity]:
        """Resolve a batch of mentions, creating canonical entities as needed."""
        if not mentions:
            return []

        # One fetch per batch, not per mention.
        known = await self._graph.list_entities(scope, limit=MAX_CANDIDATES)
        by_key = {e["key"]: e for e in known}
        by_alias: dict[str, dict] = {}
        for entity in known:
            for alias in entity.get("aliases") or []:
                by_alias.setdefault(alias, entity)

        vectors = await self._embedder.embed([m.name for m in mentions])
        resolved: list[ResolvedEntity] = []

        for mention, vector in zip(mentions, vectors):
            normalized = normalize_name(mention.name)
            if not normalized:
                continue

            match = by_key.get(mention.key) or by_alias.get(normalized)
            if match:
                resolved.append(
                    ResolvedEntity(
                        id=match["id"],
                        key=match["key"],
                        canonical_name=match["canonical_name"],
                        type=match["type"],
                        mention=mention.name,
                        confidence=1.0 if match["key"] == mention.key else 0.98,
                    )
                )
                continue

            fuzzy, similarity = self._nearest(mention, vector, known)
            agreement = (
                lexical_agreement(fuzzy["canonical_name"], mention.name) if fuzzy else 0.0
            )
            if (
                fuzzy is not None
                and similarity >= self._cfg.entity_match_threshold
                and agreement >= LEXICAL_FLOOR
            ):
                await self._graph.add_alias(fuzzy["id"], normalized)
                fuzzy.setdefault("aliases", []).append(normalized)
                by_alias[normalized] = fuzzy
                resolved.append(
                    ResolvedEntity(
                        id=fuzzy["id"],
                        key=fuzzy["key"],
                        canonical_name=fuzzy["canonical_name"],
                        type=fuzzy["type"],
                        mention=mention.name,
                        confidence=similarity,
                    )
                )
                continue

            if fuzzy is not None and similarity >= self._cfg.entity_ambiguous_threshold:
                # Close by embedding but not by name, or close by both but below
                # the auto-merge bar: ask before collapsing two identities.
                same = await self._extractor.same_entity(
                    fuzzy["canonical_name"], mention.name, mention.type
                )
                if same:
                    await self._graph.add_alias(fuzzy["id"], normalized)
                    fuzzy.setdefault("aliases", []).append(normalized)
                    by_alias[normalized] = fuzzy
                    resolved.append(
                        ResolvedEntity(
                            id=fuzzy["id"],
                            key=fuzzy["key"],
                            canonical_name=fuzzy["canonical_name"],
                            type=fuzzy["type"],
                            mention=mention.name,
                            confidence=0.9,  # model-confirmed, not certain
                        )
                    )
                    continue

            created = {
                "id": new_id(),
                "key": mention.key,
                "canonical_name": mention.name.strip(),
                "type": mention.type,
                "aliases": [normalized],
                "embedding": vector,
            }
            await self._graph.create_entity(scope, created)
            known.append(created)
            by_key[created["key"]] = created
            by_alias[normalized] = created
            resolved.append(
                ResolvedEntity(
                    id=created["id"],
                    key=created["key"],
                    canonical_name=created["canonical_name"],
                    type=created["type"],
                    mention=mention.name,
                    confidence=0.9,  # unverified: it may be a new name for a known thing
                    created=True,
                )
            )

        return resolved

    def _nearest(
        self, mention: Entity, vector: list[float], known: list[dict]
    ) -> tuple[dict | None, float]:
        """Closest same-type entity by embedding. Type guards against
        'Apple the company' collapsing into 'apple the fruit'."""
        best: dict | None = None
        best_score = 0.0
        for entity in known:
            if entity["type"] != mention.type:
                continue
            embedding = entity.get("embedding")
            if not embedding:
                continue
            score = _cosine(vector, embedding)
            if score > best_score:
                best, best_score = entity, score
        return best, best_score
