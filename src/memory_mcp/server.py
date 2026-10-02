"""MCP server exposing the memory layer.

Tool descriptions are prompts: they are the only thing telling the calling
model when to store and when to look things up. Keep them concrete.

Everything retrieved leaves through `safety.fence()`. Memories are text other
people wrote; they are handed to the model as data, never as instructions.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from mcp.server.mcpserver import MCPServer

from .config import settings
from .memory import MemoryLayer
from .models import Scope
from .ontology import SourceType
from .safety import fence

log = logging.getLogger(__name__)
_layer: MemoryLayer | None = None


@asynccontextmanager
async def lifespan(_server: MCPServer):
    global _layer
    _layer = MemoryLayer()
    await _layer.setup()
    _layer.start_workers()
    log.info("memory layer ready")
    try:
        yield {}
    finally:
        await _layer.close()
        _layer = None


mcp = MCPServer("memory", lifespan=lifespan)


def _layer_or_raise() -> MemoryLayer:
    if _layer is None:
        raise RuntimeError("memory layer is not initialized")
    return _layer


def _scope(user_id: str | None, app_id: str | None = None, agent_id: str | None = None) -> Scope:
    cfg = settings()
    return Scope(
        user_id=user_id or cfg.default_user_id,
        app_id=app_id or cfg.default_app_id,
        agent_id=agent_id or cfg.default_agent_id,
    )


@mcp.tool()
async def remember(
    text: str,
    user_id: str | None = None,
    app_id: str | None = None,
    agent_id: str | None = None,
    source_type: str = "user_statement",
    source: str | None = None,
    wait: bool = False,
) -> str:
    """Store what the user just told you about themselves.

    Call this when the user states something durable: who they are, what they
    work on, who they work with, what they prefer, how they like things done.
    Pass the statement in full sentences - it is split into atomic facts,
    deduplicated, dated, and reconciled against contradicting memories.

    Do not call this for questions, small talk, or facts about the world; the
    intake filter will discard those anyway.

    source_type matters for trust: use `user_statement` for what the user said
    themselves, and `inferred` for your own conclusions about them - inferred
    claims are ranked well below stated ones. Set wait=true if you need the
    facts back in this turn rather than letting a worker process them.
    """
    try:
        kind = SourceType(source_type)
    except ValueError:
        kind = SourceType.USER_STATEMENT
    result = await _layer_or_raise().remember(
        text,
        _scope(user_id, app_id, agent_id),
        source=source,
        source_type=kind,
        wait=wait,
    )
    lines = [result.summary()]
    for memory in result.created:
        detail = f"  + [{memory.kind.value}] {memory.text}"
        if memory.predicate:
            detail += f"  ({memory.predicate}, confidence {memory.confidence.final:.2f})"
        lines.append(detail)
    if result.superseded:
        lines.append(f"  superseded: {', '.join(result.superseded)}")
    if result.status.value == "pending":
        lines.append(f"  check with memory_job('{result.episode_id}')")
    return "\n".join(lines)


@mcp.tool()
async def recall(
    query: str,
    user_id: str | None = None,
    app_id: str | None = None,
    limit: int = 8,
    kinds: list[str] | None = None,
    as_of: str | None = None,
) -> str:
    """Look up what you know about this user, by meaning.

    Use before answering anything that depends on the user's situation,
    history, or preferences. The query is natural language - "what does she
    think about deadlines" works better than keywords.

    By default this returns only what is true *now*. Pass `as_of` as an ISO
    date to ask what was true then instead ("where did they live in 2023").
    Optionally filter `kinds`: semantic | episodic | preference | procedural.
    """
    moment = None
    if as_of:
        try:
            moment = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        except ValueError:
            return f"Could not parse as_of={as_of!r}; use an ISO date like 2024-03-01."

    hits = await _layer_or_raise().recall(
        query, _scope(user_id, app_id), limit=limit, moment=moment, kinds=kinds
    )
    return fence([_format_hit(h) for h in hits])


def _format_hit(hit) -> str:
    memory = hit.memory
    meta = (
        f"{memory.kind.value}; confidence {memory.confidence.final:.2f}; "
        f"via {'+'.join(hit.retrieved_by)}; id={memory.id}"
    )
    return f"[{hit.score:.3f}] {memory.text}  ({meta})"


@mcp.tool()
async def forget(
    user_id: str | None = None,
    app_id: str | None = None,
    memory_ids: list[str] | None = None,
    query: str | None = None,
    hard: bool = False,
) -> str:
    """Remove memories, by id or by description.

    Call when the user asks you to forget something. By default the memory is
    invalidated (retained but never retrieved); pass hard=true only when the
    user wants it genuinely erased. Prefer `memory_ids` from a recall() result
    over `query`, which removes whatever the description matches.
    """
    if not memory_ids and not query:
        return "Nothing to forget: pass memory_ids or query."
    removed = await _layer_or_raise().forget(
        _scope(user_id, app_id), memory_ids, query, hard
    )
    verb = "deleted" if hard else "invalidated"
    return f"{verb} {len(removed)} memories: {', '.join(removed) or 'none'}"


@mcp.tool()
async def history(
    subject: str,
    user_id: str | None = None,
    app_id: str | None = None,
    predicate: str | None = None,
) -> str:
    """Show how a fact about someone changed over time.

    Answers "where have they lived", "where have they worked" - every value the
    attribute has held, with the dates it held them and what replaced it.
    Optional predicate, one of: LIVES_IN, BORN_IN, WORKS_AT, HAS_ROLE, STUDIED_AT, HAS_SKILL, PREFERS, DISLIKES, USES, OWNS, KNOWS, MANAGES, REPORTS_TO, WORKS_ON, INTERESTED_IN, HAS_GOAL, LOCATED_IN, SPEAKS.
    """
    rows = await _layer_or_raise().history(_scope(user_id, app_id), subject, predicate)
    if not rows:
        return f"No history for {subject!r}."
    lines = []
    for memory in rows:
        window = f"{memory.valid_from:%Y-%m-%d} -> " + (
            f"{memory.valid_to:%Y-%m-%d}" if memory.valid_to else "now"
        )
        state = "current" if memory.is_current else "superseded"
        lines.append(f"{window}  [{state}]  {memory.text}")
    return fence(lines)


@mcp.tool()
async def timeline(
    user_id: str | None = None, app_id: str | None = None, limit: int = 20
) -> str:
    """List what has happened to this user, newest first.

    Episodic memories only - events with a time, not standing facts.
    """
    memories = await _layer_or_raise().timeline(_scope(user_id, app_id), limit)
    if not memories:
        return "No episodic memories."
    return fence([f"{m.valid_from:%Y-%m-%d}  {m.text}" for m in memories])


@mcp.tool()
async def relate(
    entity: str, user_id: str | None = None, app_id: str | None = None, hops: int = 2
) -> str:
    """Traverse the user's knowledge graph outward from a person or thing.

    Answers questions similarity search cannot reach on its own, like "who does
    Dana work with" or "what is connected to the Atlas project".
    """
    rows = await _layer_or_raise().relate(_scope(user_id, app_id), entity, hops)
    if not rows:
        return f"Nothing connected to {entity!r}."
    return fence(
        [
            f"{entity} --{' -> '.join(r['via'])}--> {r['name']} "
            f"({r['type']}, {r['hops']} hop(s))"
            for r in rows
        ]
    )


@mcp.tool()
async def profile(
    user_id: str | None = None, app_id: str | None = None, limit: int = 12
) -> str:
    """The user's most important currently-true facts, as a short digest.

    Read this once at the start of a conversation to orient yourself, then use
    recall() for anything specific.
    """
    memories = await _layer_or_raise().profile(_scope(user_id, app_id), limit)
    if not memories:
        return "No memories stored for this user yet."
    return fence([f"- {m.text}" for m in memories])


@mcp.tool()
async def memory_job(episode_id: str) -> str:
    """Check what happened to a queued remember() call."""
    status = await _layer_or_raise().job_status(episode_id)
    lines = [f"episode {status['episode_id']}: {status['status']}"]
    if status.get("category"):
        lines.append(f"  category: {status['category']}")
    if status.get("error"):
        lines.append(f"  error: {status['error']}")
    for text in status.get("memories", []):
        lines.append(f"  + {text}")
    return "\n".join(lines)


@mcp.tool()
async def memory_stats(user_id: str | None = None, app_id: str | None = None) -> str:
    """Counts of episodes, facts, entities, and pending index work."""
    layer = _layer_or_raise()
    scope = _scope(user_id, app_id)
    stats = await layer.stats(scope)
    outbox = stats.get("outbox", {})
    return (
        f"scope={scope.key}  episodes={stats.get('episodes', 0)}  "
        f"facts={stats.get('total', 0)}  believed={stats.get('believed', 0)}  "
        f"needs_verification={stats.get('needs_verification', 0)}  "
        f"entities={stats.get('entities', 0)}  "
        f"outbox={outbox or 'empty'}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mcp.run()


if __name__ == "__main__":
    main()
