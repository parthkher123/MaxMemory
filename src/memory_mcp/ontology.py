"""Controlled vocabulary for facts: predicates, cardinality, source trust.

Free-form predicates sprawl (`works_at`, `employed_by`, `job_at`) until the
graph stops being traversable and contradiction detection stops working. Every
extracted relation is mapped onto this table or dropped to an unstructured fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Cardinality(str, Enum):
    ONE = "one"    # one current value: a new value supersedes the old
    MANY = "many"  # many can hold at once: new values coexist


class SourceType(str, Enum):
    USER_STATEMENT = "user_statement"
    VERIFIED_DOCUMENT = "verified_document"
    UPLOADED_DOCUMENT = "uploaded_document"
    API = "api"
    TOOL = "tool"
    IMPORTED_PROFILE = "imported_profile"
    ASSISTANT_STATEMENT = "assistant_statement"
    INFERRED = "inferred"


#: How much we believe a claim purely because of where it came from.
#: What the user said about themselves is ground truth; what a model guessed
#: about them is not, and the two must never rank equally.
SOURCE_TRUST: dict[SourceType, float] = {
    SourceType.USER_STATEMENT: 1.00,
    SourceType.VERIFIED_DOCUMENT: 1.00,
    SourceType.UPLOADED_DOCUMENT: 0.95,
    SourceType.API: 0.90,
    SourceType.TOOL: 0.90,
    SourceType.IMPORTED_PROFILE: 0.80,
    SourceType.ASSISTANT_STATEMENT: 0.50,
    SourceType.INFERRED: 0.30,
}


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    cardinality: Cardinality
    subject_type: str
    object_type: str
    temporal: bool = False  # does this fact usually have a start/end date?
    aliases: tuple[str, ...] = field(default_factory=tuple)


_SPECS: tuple[PredicateSpec, ...] = (
    PredicateSpec("LIVES_IN", Cardinality.ONE, "person", "place", True,
                  ("lives_in", "resides_in", "located_in_city", "based_in", "moved_to")),
    PredicateSpec("BORN_IN", Cardinality.ONE, "person", "place", False,
                  ("born_in", "birthplace")),
    PredicateSpec("WORKS_AT", Cardinality.ONE, "person", "org", True,
                  ("works_at", "employed_by", "job_at", "works_for", "joined")),
    PredicateSpec("HAS_ROLE", Cardinality.ONE, "person", "thing", True,
                  ("has_role", "works_as", "title_is", "is_a")),
    PredicateSpec("STUDIED_AT", Cardinality.MANY, "person", "org", True,
                  ("studied_at", "studies_at", "attended", "graduated_from")),
    PredicateSpec("HAS_SKILL", Cardinality.MANY, "person", "thing", False,
                  ("has_skill", "skilled_in", "knows_how_to", "experienced_in")),
    PredicateSpec("PREFERS", Cardinality.MANY, "person", "thing", False,
                  ("prefers", "likes", "favours", "favorite_is", "enjoys")),
    PredicateSpec("DISLIKES", Cardinality.MANY, "person", "thing", False,
                  ("dislikes", "hates", "avoids")),
    PredicateSpec("USES", Cardinality.MANY, "person", "tool", False,
                  ("uses", "works_with_tool", "runs")),
    PredicateSpec("OWNS", Cardinality.MANY, "person", "thing", False,
                  ("owns", "has", "possesses")),
    PredicateSpec("KNOWS", Cardinality.MANY, "person", "person", False,
                  ("knows", "friends_with", "acquainted_with")),
    PredicateSpec("MANAGES", Cardinality.MANY, "person", "person", True,
                  ("manages", "leads", "supervises", "manager_of")),
    PredicateSpec("REPORTS_TO", Cardinality.ONE, "person", "person", True,
                  ("reports_to", "managed_by", "works_under")),
    PredicateSpec("WORKS_ON", Cardinality.MANY, "person", "project", True,
                  ("works_on", "contributes_to", "builds", "built", "maintains")),
    PredicateSpec("INTERESTED_IN", Cardinality.MANY, "person", "thing", False,
                  ("interested_in", "curious_about", "learning")),
    PredicateSpec("HAS_GOAL", Cardinality.MANY, "person", "thing", True,
                  ("has_goal", "wants_to", "plans_to", "aims_to")),
    PredicateSpec("LOCATED_IN", Cardinality.ONE, "thing", "place", False,
                  ("located_in", "situated_in", "part_of_place")),
    PredicateSpec("SPEAKS", Cardinality.MANY, "person", "thing", False,
                  ("speaks", "fluent_in")),
)

PREDICATES: dict[str, PredicateSpec] = {spec.name: spec for spec in _SPECS}

_ALIAS_INDEX: dict[str, str] = {}
for _spec in _SPECS:
    _ALIAS_INDEX[_spec.name.lower()] = _spec.name
    for _alias in _spec.aliases:
        _ALIAS_INDEX[_alias.lower()] = _spec.name

#: The set a model is allowed to choose from, for prompts.
PREDICATE_NAMES: tuple[str, ...] = tuple(PREDICATES)

#: Tense and hedge wrappers a model puts around a predicate. World time
#: carries the tense, so these are stripped rather than modelled.
_MODIFIER_PREFIXES = (
    "currently_", "current_", "previously_", "formerly_", "used_to_",
    "still_", "now_", "also_", "has_been_", "is_", "was_",
)
_MODIFIER_SUFFIXES = ("_now", "_currently", "_previously", "_before", "_at_present")


def normalize_predicate(raw: str | None) -> str | None:
    """Map an extracted predicate onto the ontology, or None if it doesn't fit.

    None is not a failure: the fact is still stored as unstructured text and
    retrieved semantically. It just doesn't take part in structured
    contradiction logic.
    """
    if not raw:
        return None
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if key in _ALIAS_INDEX:
        return _ALIAS_INDEX[key]
    # Strip tense and hedge modifiers, then require an exact match. Substring
    # matching is deliberately not used: "enjoys_the_company_of" is not PREFERS.
    for prefix in _MODIFIER_PREFIXES:
        if key.startswith(prefix) and key[len(prefix):] in _ALIAS_INDEX:
            return _ALIAS_INDEX[key[len(prefix):]]
    for suffix in _MODIFIER_SUFFIXES:
        if key.endswith(suffix) and key[: -len(suffix)] in _ALIAS_INDEX:
            return _ALIAS_INDEX[key[: -len(suffix)]]
    return None


def spec_for(predicate: str | None) -> PredicateSpec | None:
    return PREDICATES.get(predicate) if predicate else None


def is_single_valued(predicate: str | None) -> bool:
    spec = spec_for(predicate)
    return spec is not None and spec.cardinality is Cardinality.ONE


def source_trust(source_type: SourceType | str) -> float:
    try:
        return SOURCE_TRUST[SourceType(source_type)]
    except (ValueError, KeyError):
        return 0.5
