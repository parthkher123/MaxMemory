# Memory MCP

A persistent, long-term memory layer for LLM agents, exposed as an
[MCP](https://modelcontextprotocol.io) server.

Plug it into Claude Code (or any MCP client) and the assistant can **remember**
what a user tells it, **recall** it in later conversations, track how facts
**change over time**, and **forget** on request.

- **Neo4j** is the source of truth: raw episodes, extracted facts, entities,
  relationships, bi-temporal validity, and provenance.
- **Qdrant** is the vector index for semantic search, rebuildable from Neo4j.
- **An LLM** extracts atomic facts and judges contradictions, and **an
  embedding model** powers semantic search. Both are pluggable; see
  [Providers](#providers).

> **Episode → Fact → Entity → Relationship → Temporal State → Retrieval**
> rather than Text → Embedding → Vector Search.

A vector store returns *similar text*. A memory layer has to answer *what is
true about this user now*, *what was true in 2023*, *how sure are we*, and
*where did that come from*. That needs a graph, two clocks, and provenance.

---

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Providers](#providers)
- [Connecting to Claude Code](#connecting-to-claude-code)
- [Usage](#usage)
- [Tools reference](#tools-reference)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Limitations and roadmap](#limitations-and-roadmap)

---

## Features

| | |
|---|---|
| **Fact extraction** | Free text is split into atomic facts mapped onto a fixed predicate ontology (`LIVES_IN`, `WORKS_AT`, `HAS_SKILL`, …). |
| **Contradiction handling** | "I moved to Mumbai" supersedes "I live in Pune"; "I know Rust" coexists with "I know Python". |
| **Time travel** | `recall(..., as_of="2023-06-01")` answers what was true at that moment. `history()` shows every value an attribute has held. |
| **Hybrid retrieval** | Vector search + full-text search, fused with RRF, expanded over the entity graph, ranked by importance, confidence and recency. |
| **Confidence & provenance** | Every fact knows where it came from and how much to trust it. Inferred claims never outrank what the user said. |
| **Multi-tenant** | Every read and write is scoped to `(user_id, app_id)`. |
| **Injection defence** | Text that reads like instructions is refused at intake, and recalled memories are fenced as untrusted data. |
| **Reliable writes** | A transactional outbox keeps Qdrant consistent with Neo4j even across crashes and outages. |
| **Idempotent** | Replaying the same conversation never duplicates memories. |
| **Provider-agnostic** | Claude, OpenAI, or any OpenAI-compatible server (Ollama, Groq, OpenRouter, …) for the LLM; Voyage, OpenAI or a local server for embeddings. Fully local setups work too. |

---

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| Docker Desktop | any recent | Runs Neo4j and Qdrant. Must be **running** whenever the server is used. |
| An LLM | — | Anthropic or OpenAI API key, or a local server such as Ollama. See [Providers](#providers). |
| An embedding model | — | Voyage or OpenAI API key, or a local server. |

Both models are optional for trying things out, but see
[Running without API keys](#running-without-api-keys).

---

## Installation

The commands below are for **Windows PowerShell**. On macOS/Linux, replace
`.venv\Scripts\` with `.venv/bin/`.

### 1. Start the databases

Make sure Docker Desktop is open and shows **Engine running**, then:

```powershell
docker compose up -d
```

This starts two containers:

| Container | Ports | Purpose |
|---|---|---|
| `memory-neo4j` | `7687` (Bolt), `7474` (browser UI) | Graph store, source of truth |
| `memory-qdrant` | `6333` (HTTP), `6334` (gRPC) | Vector index |

Check that both report `healthy`:

```powershell
docker ps --format "{{.Names}}: {{.Status}}"
```

### 2. Install the package

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

### 3. Configure

```powershell
Copy-Item .env.example .env
notepad .env
```

Pick an LLM and an embedding provider. The default is Claude + Voyage:

```ini
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...

EMBEDDING_PROVIDER=voyage
VOYAGE_API_KEY=pa-...
```

For OpenAI, Ollama and others, see [Providers](#providers). The database
defaults already match `docker-compose.yml`. `.env` is git-ignored, so never
commit your keys.

### 4. Verify

```powershell
.venv\Scripts\python -m pytest -q
```

All tests should pass. If the integration tests are **skipped**, the
containers aren't reachable. See [Troubleshooting](#troubleshooting).

---

## Providers

The layer needs two kinds of model, and each is configured on its own:

| Role | Used for | Setting | Options |
|---|---|---|---|
| **LLM** | Intake gate, fact extraction, contradiction judge, entity tiebreaks | `LLM_PROVIDER` | `anthropic`, `openai`, `none` |
| **Embeddings** | Semantic search, deduplication, entity matching | `EMBEDDING_PROVIDER` | `voyage`, `openai`, `hash` |

`openai` means **any OpenAI-compatible API**, not just OpenAI itself. Point
`LLM_BASE_URL` / `EMBEDDING_BASE_URL` at another server and it works the same
way. You can mix and match, for example Claude for extraction and OpenAI for
embeddings.

### Claude + Voyage (default)

```ini
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_PROVIDER=voyage
VOYAGE_API_KEY=pa-...
```

Default models: `claude-opus-5` (extraction), `claude-haiku-4-5` (gate),
`voyage-3.5` (1024-dim embeddings).

### OpenAI

```ini
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

Default models: `gpt-4.1` (extraction), `gpt-4.1-mini` (gate). Override them
with `EXTRACTION_MODEL` / `GATE_MODEL`.

### Fully local with Ollama (no API keys)

```powershell
ollama pull llama3.1
ollama pull nomic-embed-text
```

```ini
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:11434/v1
EXTRACTION_MODEL=llama3.1
GATE_MODEL=llama3.1

EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
```

No key is needed when the base URL points at a local server. Small local models
follow the extraction schema less reliably than hosted ones. When a reply
doesn't parse, that episode is stored verbatim instead of being lost.

### Other OpenAI-compatible services

Set `LLM_PROVIDER=openai`, the service's base URL, its key, and its model
names:

| Service | `LLM_BASE_URL` |
|---|---|
| Groq | `https://api.groq.com/openai/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Together | `https://api.together.xyz/v1` |
| LM Studio | `http://localhost:1234/v1` |
| vLLM | `http://localhost:8000/v1` |

```ini
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=gsk_...
EXTRACTION_MODEL=<model name on that service>
GATE_MODEL=<model name on that service>
```

`LLM_API_KEY` overrides `OPENAI_API_KEY` for the LLM only, so the LLM and
embeddings can use different accounts or servers.

The model must support JSON output. The server requests a JSON schema
response format and also describes the schema in the prompt, so servers that
ignore `response_format` still work with capable models.

### Switching embedding models

Vectors from different models aren't comparable. If you change
`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` or `EMBEDDING_DIM` after storing
memories, use a new `QDRANT_COLLECTION` (or delete the old one) and re-embed.
Switching the **LLM** needs nothing special: it only affects new memories.

### Adding a new provider

Providers live behind small interfaces, so adding one is a single class:

- **LLM:** implement `parse(model, system, user, schema, max_tokens)`, which
  returns a validated pydantic object, in
  [`llm.py`](src/memory_mcp/llm.py), and register it in `build_llm()`.
- **Embeddings:** implement `embed(texts, query=False)` in
  [`embeddings.py`](src/memory_mcp/embeddings.py), and register it in
  `build_embedder()`.

---

## Connecting to Claude Code

The server speaks MCP over **stdio**: the client launches it as a child
process. You don't keep it running in a terminal yourself.

```powershell
claude mcp add memory -- "C:\full\path\to\Memory_MCP\.venv\Scripts\memory-mcp.exe"
```

Then **restart Claude Code** (MCP servers load at session start) and run
`/mcp`. `memory` should show as **connected**.

For other MCP clients (Claude Desktop, Cursor, …), add an entry like this to
the client's MCP config:

```json
{
  "mcpServers": {
    "memory": {
      "command": "C:\\full\\path\\to\\Memory_MCP\\.venv\\Scripts\\memory-mcp.exe"
    }
  }
}
```

The server finds `.env` in the project root regardless of the working directory
it is launched from.

---

## Usage

Once connected, just talk to the assistant. The tool descriptions tell it when
to store and when to look things up.

```text
You:    Remember that I live in Pune and work on the Memory MCP project.
Claude: → remember(text="I live in Pune and work on the Memory MCP project.")

You:    Where do I live?
Claude: → recall(query="Where does the user live?")
        You live in Pune.

You:    I moved to Mumbai last month.
Claude: → remember(...)            # Pune is superseded, not deleted

You:    Where have I lived?
Claude: → history(subject="user", predicate="LIVES_IN")
        Pune (until 2026-09), then Mumbai (now).

You:    Forget where I work.
Claude: → forget(query="where the user works")
```

By default `remember()` is **asynchronous**: it returns an episode id at once
and a background worker extracts the facts a few seconds later. If `recall()`
comes back empty straight after a `remember()`, wait a moment, or call
`memory_job(episode_id)` to check progress.

---

## Tools reference

All tools accept optional `user_id` and `app_id`. When omitted they fall back
to `DEFAULT_USER_ID` / `DEFAULT_APP_ID`.

| Tool | Key parameters | What it does |
|---|---|---|
| `remember` | `text`, `source_type` (`user_statement` \| `inferred`), `source`, `wait` | Episode → gate → extract → resolve → reconcile → store. `wait=true` blocks and returns the facts. |
| `recall` | `query`, `limit=8`, `kinds`, `as_of` | Hybrid retrieval of what is true now, or at the ISO date `as_of`. `kinds`: `semantic`, `episodic`, `preference`, `procedural`. |
| `forget` | `memory_ids` or `query`, `hard=false` | Retracts memories (kept for audit, never retrieved). `hard=true` erases them. |
| `history` | `subject`, `predicate` | Every value an attribute has held, with date ranges. |
| `timeline` | `limit=20` | Episodic memories (events), newest first. |
| `relate` | `entity`, `hops=2` | Walks the knowledge graph outward, following only edges that still hold. |
| `profile` | `limit=12` | Short digest of the most important currently-true facts, for orienting at the start of a conversation. |
| `memory_job` | `episode_id` | Status of a queued `remember()` call, with any error and the facts produced. |
| `memory_stats` | — | Counts of episodes, facts, entities, and pending index work. |

**Supported predicates:** `LIVES_IN`, `BORN_IN`, `WORKS_AT`, `HAS_ROLE`,
`STUDIED_AT`, `HAS_SKILL`, `PREFERS`, `DISLIKES`, `USES`, `OWNS`, `KNOWS`,
`MANAGES`, `REPORTS_TO`, `WORKS_ON`, `INTERESTED_IN`, `HAS_GOAL`, `LOCATED_IN`,
`SPEAKS`.

---

## How it works

### Write path

```
remember(text)
     │
     ▼
┌────────────┐   raw text, hashed for idempotency, never discarded
│  EPISODE   │   (:Episode)-[:DERIVED]->(:Memory)
└─────┬──────┘
      ▼
┌────────────┐   cheap classifier: anything worth remembering here?
│    GATE    │   also rejects text that reads as instructions to a model
└─────┬──────┘
      ▼
┌────────────┐   atomic facts + ontology predicate + world-time dates
│  EXTRACT   │   + per-fact extraction confidence
└─────┬──────┘
      ▼
┌────────────┐   mention -> canonical entity
│  RESOLVE   │   exact key → alias → embedding → LLM → create
└─────┬──────┘
      ▼
┌────────────┐   duplicate? update? coexist?
│ RECONCILE  │   structured triples settle it with an index lookup;
└─────┬──────┘   only unstructured facts reach the judge model
      ▼
┌────────────┐   Neo4j commit includes the outbox row,
│   STORES   │   so the Qdrant write cannot be lost
└────────────┘
```

### Read path

```
query → embed → Qdrant ANN (scope-filtered inside HNSW)
              ‖ Neo4j full-text (BM25)
      → reciprocal rank fusion
      → graph expansion over shared entities
      → rank by rrf × importance × confidence × recency
      → filter to what is true at the requested moment
      → fence as untrusted data
```

### Data model

```
(:User)-[:OBSERVED]->(:Episode)-[:DERIVED]->(:Memory)-[:MENTIONS]->(:Entity)
                                 (:Memory)-[:SUPERSEDES]->(:Memory)
                                 (:Entity)-[:RELATED {predicate, valid_to}]->(:Entity)
(:Outbox)  pending index operations, drained into Qdrant
```

You can browse the graph at <http://localhost:7474>
(user `neo4j`, password `memorypassword`).

### Two clocks

| Field | Means | Answers |
|---|---|---|
| `valid_from` / `valid_to` | when the fact was true **in the world** | "Where did they live in 2023?" |
| `recorded_at` / `invalid_at` | when **we** learned it / retracted it | "What did we believe in March?" |

These are deliberately separate. Superseding a fact because the world changed
sets `valid_to` only. The old record was never *wrong*, so an as-of query can
still answer with it. `invalid_at` is reserved for records we retract:
corrections and `forget()`.

### Confidence

```
final = extraction × source_trust × temporal_consistency × entity_resolution
```

A `user_statement` has source trust 1.0, an `inferred` claim 0.3, so a guess
can never outrank something the user said. Facts below `VERIFY_THRESHOLD` are
flagged `needs_verification` for the app to confirm.

### Predicate ontology

Free-form predicates sprawl (`works_at`, `employed_by`, `job_at`) until the
graph stops being traversable. Extracted relations are mapped onto the closed
list above, or kept as unstructured text. Each predicate has a cardinality:
`LIVES_IN` is single-valued, so a new city supersedes the old; `HAS_SKILL` is
multi-valued, so Python and Rust coexist.

This is also what makes reconciliation cheap. A structured triple is settled
by an indexed lookup on `(subject_key, predicate)`, with no embeddings and no
model call. Only unstructured facts fall through to similarity plus a judge.

### Guarantees

- **Scope.** Every read and write filters on `(user_id, app_id)`; `agent_id`
  is recorded and available as a filter. One person using two of your products
  does not get one memory pool. `forget()` ignores ids from another scope.
- **Idempotency.** An episode's key is `sha256(normalized_text | user | app)`.
  Replaying a conversation or retrying a failed request returns the original
  episode instead of duplicating work.
- **Cross-store consistency.** Two stores cannot be written atomically, so the
  Qdrant operation is committed to Neo4j *in the same transaction as the fact*
  and applied by a drainer. Failures retry with an attempt count; rows that
  exhaust it are parked as `dead` rather than dropped. Neo4j is always correct;
  Qdrant is eventually correct and can be rebuilt from Neo4j.
- **Injection defence.** Stored memories end up in future prompts, so they are
  treated as hostile on both sides. A regex prefilter plus the gate classifier
  refuse to persist text that reads as an instruction, and everything retrieved
  leaves through a fence that tells the model it is data. Blocked episodes are
  kept, marked `blocked`, for audit.

---

## Configuration

All settings come from environment variables, loaded from `.env`. See
[`.env.example`](.env.example) for the full annotated list.

### Connections

| Variable | Default | Notes |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `memorypassword` | Must match `docker-compose.yml`. |
| `NEO4J_DATABASE` | `neo4j` | |
| `QDRANT_URL` | `http://localhost:6333` | |
| `QDRANT_API_KEY` | *(empty)* | Only for Qdrant Cloud. |
| `QDRANT_COLLECTION` | `memories` | |

### LLM

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai` (any compatible server), or `none`. |
| `ANTHROPIC_API_KEY` | *(empty)* | Used when `LLM_PROVIDER=anthropic`. |
| `OPENAI_API_KEY` | *(empty)* | Used when `LLM_PROVIDER=openai`, and by `openai` embeddings. |
| `LLM_API_KEY` | *(empty)* | Optional override for the LLM's key only. |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | `openai` provider only. Local servers need no key. |
| `EXTRACTION_MODEL` | `claude-opus-5` / `gpt-4.1` | Fact extraction and contradiction judgement. Default depends on the provider. |
| `GATE_MODEL` | `claude-haiku-4-5` / `gpt-4.1-mini` | Runs on every episode, so it's the cheap one. |

With no usable key, extraction is off and text is stored verbatim.

### Embeddings

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `voyage` | `voyage`, `openai` (any compatible server), or `hash` (offline). |
| `EMBEDDING_MODEL` | `voyage-3.5` / `text-embedding-3-small` / `hash-256` | Default depends on the provider. |
| `EMBEDDING_DIM` | known models: their size; otherwise `256` | **Set this explicitly** for models the server doesn't know. |
| `EMBEDDING_BASE_URL` | `https://api.openai.com/v1` | `openai` provider only. |
| `VOYAGE_API_KEY` | *(empty)* | Used when `EMBEDDING_PROVIDER=voyage`. |

> Changing the embedding model requires a new Qdrant collection and
> re-embedding (see [Switching embedding models](#switching-embedding-models)).
> Every memory records the `embedding_model` and `extraction_model` that
> produced it, so a migration can find what needs rebuilding.

### Ingestion

| Variable | Default | Effect |
|---|---|---|
| `INGEST_ASYNC` | `true` | `remember()` returns an episode id and a worker extracts. `false` blocks until done. |
| `WORKER_POLL_SECONDS` | `2.0` | How often workers check for new work. |
| `OUTBOX_MAX_ATTEMPTS` | `8` | Retries before an index operation is parked as `dead`. |
| `EPISODE_MAX_ATTEMPTS` | `3` | Retries before an episode is parked as `failed`. |
| `CLAIM_TIMEOUT_SECONDS` | `300` | A crashed worker's claim is released after this long. |

### Retrieval and reconciliation

| Variable | Default | Effect |
|---|---|---|
| `RECALL_LIMIT` | `8` | Default number of results. |
| `RRF_K` | `60` | Reciprocal rank fusion constant. |
| `GRAPH_EXPANSION_HOPS` | `1` | How far recall expands over shared entities. |
| `RECENCY_HALF_LIFE_DAYS` | `180` | How quickly old memories lose rank. |
| `QDRANT_HNSW_EF` | `128` | Raise (16–512) for better recall on large tenants. |
| `DEDUPE_THRESHOLD` | `0.95` | At or above: same fact, reinforce instead of insert. |
| `CONTRADICTION_THRESHOLD` | `0.84` | At or above: possibly conflicting, ask the judge. |
| `ENTITY_MATCH_THRESHOLD` | `0.93` | Auto-merge bar for entity resolution. |
| `ENTITY_AMBIGUOUS_THRESHOLD` | `0.85` | Between this and the match bar, ask the model. |
| `VERIFY_THRESHOLD` | `0.5` | Below this, facts are flagged `needs_verification`. |

### Scope defaults

| Variable | Default |
|---|---|
| `DEFAULT_USER_ID` | `default` |
| `DEFAULT_APP_ID` | `default` |
| `DEFAULT_AGENT_ID` | `default` |

### Running without API keys

The server runs with no models at all, which is useful for tests and offline
development (for real offline use, prefer [Ollama](#fully-local-with-ollama-no-api-keys)):

- with no LLM (`LLM_PROVIDER=none`, or no key), text is stored **verbatim** as a single memory,
  with no fact extraction, so `history`, `relate` and supersession don't
  really work;
- with no embedding key, a deterministic **hash embedder** is used, so recall
  relies mostly on keyword overlap.

The server logs a warning at startup when it falls back. Don't use this mode
for real recall.

---

## Project structure

```
Memory_MCP/
├── src/memory_mcp/
│   ├── server.py         # MCP server and tool definitions (entry point: memory-mcp)
│   ├── memory.py         # MemoryLayer: orchestrates the write and read pipelines
│   ├── extraction.py     # Gate, fact extraction, contradiction judge (prompts)
│   ├── llm.py            # LLM providers: Anthropic, OpenAI-compatible
│   ├── entities.py       # Entity resolution
│   ├── ontology.py       # Predicate list, cardinalities, source types
│   ├── graph_store.py    # Neo4j access
│   ├── vector_store.py   # Qdrant access
│   ├── outbox.py         # Transactional outbox drainer
│   ├── embeddings.py     # Embedding providers: Voyage, OpenAI-compatible, hash
│   ├── safety.py         # Injection prefilter and output fencing
│   ├── models.py         # Pydantic models (Memory, Scope, Episode, ...)
│   └── config.py         # Settings loaded from .env
├── scripts/
│   └── worker.py         # Standalone ingestion + outbox worker
├── tests/
│   ├── test_unit.py          # Runs anywhere
│   ├── test_llm.py           # Provider selection, OpenAI-compatible client (no network)
│   └── test_integration.py   # Needs the containers
├── docker-compose.yml    # Neo4j + Qdrant
├── pyproject.toml
└── .env.example
```

### Running the worker separately

The MCP server runs the episode and outbox loops in-process. For higher
volume, run them as a separate process:

```powershell
.venv\Scripts\python scripts\worker.py
```

---

## Testing

```powershell
.venv\Scripts\python -m pytest -q        # everything
.venv\Scripts\python -m pytest -q tests\test_unit.py tests\test_llm.py   # no Docker needed
.venv\Scripts\ruff check .               # lint
```

- **`test_unit.py`** covers ontology normalization, injection patterns,
  confidence arithmetic, temporal overlap, and ranking.
- **`test_llm.py`** covers provider selection and the OpenAI-compatible
  request/response handling, against a mocked HTTP transport.
- **`test_integration.py`** covers every guarantee above end to end: episode
  idempotency, as-of queries, sequential vs. overlapping facts, cardinality,
  scope isolation, outbox retry after a simulated Qdrant outage, and erasure.
  If Neo4j or Qdrant is unreachable, these tests are **skipped**, not failed.

---

## Troubleshooting

**`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`**
Docker Desktop isn't running. Start it, wait for **Engine running**, and check
with `docker info`. If it fails to start with a WSL error, run `wsl --update`
and restart Docker Desktop.

**Integration tests are all skipped**
The containers aren't up or aren't healthy. Run `docker compose up -d` and
check `docker ps`.

**`memory` doesn't appear in `/mcp`, or shows as failed**
- Restart Claude Code after `claude mcp add`.
- Check that the path to `memory-mcp.exe` is absolute and correct.
- Check that Docker is running. The server connects to both databases at
  startup and fails if it can't.

**`recall()` returns nothing right after `remember()`**
Ingestion is asynchronous. Wait a few seconds, or check
`memory_job(episode_id)`. Set `INGEST_ASYNC=false` to make `remember()` block.

**Warning: `falling back to the offline hash embedder`**
The embedding provider has no key (`VOYAGE_API_KEY` or `OPENAI_API_KEY`) and
no custom `EMBEDDING_BASE_URL`. Add one to `.env` and restart the client.

**Warning: `LLM provider ... selected but no API key set`, or memories stored word-for-word**
The LLM provider has no key. Set `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or
`LLM_API_KEY` to match `LLM_PROVIDER`, then restart the client.

**`extraction failed, storing text verbatim` in the logs**
The model's reply wasn't valid JSON for the schema, or the request failed.
This is common with small local models; try a larger one. Also check that
`EXTRACTION_MODEL` / `GATE_MODEL` are model names the server actually has.

**`collection 'memories' stores N-dim vectors but EMBEDDING_DIM is M` at startup**
You switched embedding models without a new collection. Set
`QDRANT_COLLECTION` to a new name, or set `EMBEDDING_DIM` back. See
[Switching embedding models](#switching-embedding-models).

**Starting from a clean slate**
`docker compose down -v` deletes both databases' volumes, so **all memories
are lost**.

---

## Limitations and roadmap

Not built yet:

- **No reranker.** RRF is the final ranker. A cross-encoder over the top 50
  is the largest remaining quality win.
- **No MMR**, so a result set can contain several phrasings of one fact.
- **No query decomposition** for multi-part questions.
- **No eval harness.** Every threshold above is a considered guess, not a
  measured optimum. This is the next thing to build; until it exists, tuning
  those numbers is guesswork.
- **No consolidation/reflection**, tiering, PII classification, retention
  policies, read audit logs, or recall-usage tracking.

Known edge cases:

- **Reconciliation only sees indexed memories.** The pipeline drains the
  outbox before reconciling to avoid the obvious race, but two contradictory
  facts ingested concurrently by separate workers can both land.
- **Claim recovery is time-based.** A worker that stalls for longer than
  `CLAIM_TIMEOUT_SECONDS` without dying can have its episode reclaimed and
  processed twice. Reconciliation dedupes the result, but the work is wasted.
