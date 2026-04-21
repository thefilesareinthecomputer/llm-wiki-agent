# LLM Wiki Agent

A working implementation of the [LLM Wiki Pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) proposed by Andrej Karpathy, extended with medallion-tiered storage and hybrid local-or-cloud inference. Runs in Docker, uses Obsidian-compatible markdown as the source of truth.

![LLM Wiki Agent UI](static-assets/images/demo.gif)

_Any Ollama model works — switch at runtime from the UI dropdown._

## Why this exists

Most personal AI agents pick a side: fully local (privacy) or fully cloud (speed). This one switches between them at runtime, same KB either way. Use Ollama locally for sensitive material. Switch to cloud models for heavier reasoning. Embeddings can be local (Ollama) or cloud (Gemini). You decide per session.

**Three ideas this repo explores:**

1. **The LLM Wiki Pattern as a working system.** Focused single-concept pages, explicit `[[wiki-links]]`, graph traversal as the primary navigation mode.
2. **Medallion tiers applied to personal knowledge.** `canon/` (read-only gold) > `knowledge/wiki/` (agent-writable silver) > `knowledge/raw/` (source bronze). Search ranks tiers accordingly.
3. **Honest tool execution.** The agent never reports work it didn't do. Tool results are labeled `COMPLETE`, `TRUNCATED`, or `NOT_EXECUTED`. The UI surfaces the distinction.

## Inference Modes

| Use Case | Chat Model | Embeddings | Network |
|----------|------------|------------|---------|
| Fully local / sensitive data | Ollama (local) | Ollama (local) | None required |
| Hybrid (recommended) | Ollama (local or cloud) | Gemini (cloud) | Embeddings only |
| Fully cloud / maximum speed | Ollama Cloud | Gemini (cloud) | Required |

Switch chat models at runtime from the UI. Embedding provider switches via `EMBEDDING_PROVIDER` in `.env` (requires a lance-data wipe — embedding spaces aren't compatible across providers).

## Quick Start

```bash
# Clone and build
git clone https://github.com/thefilesareinthecomputer/llm-wiki-agent.git && cd llm-wiki-agent

# Create .env with your Gemini API key
echo "GOOGLE_GEMINI_API_KEY=AIza..." > .env
echo "EMBEDDING_PROVIDER=gemini" >> .env

docker compose up --build -d

# Access at http://localhost:8080
```

**Prerequisites:** Docker Desktop, Ollama Mac app with at least one model pulled, Google Gemini API key.

**Changing code:** `src/` and `tests/` are copied into the image at build time (not bind-mounted). After edits, rebuild: `docker compose down && docker compose up --build -d`, then run tests in the container.

## Features

- **Local inference** via Ollama — no cloud API required for chat
- **Gemini embeddings** — fast cloud-based vector search (gemini-embedding-001, 768-dim)
- **Model switching** — change models at runtime from the UI dropdown
- **Conversation sessions** — create, switch, delete chats; history persists across reloads
- **KB tools** — 14 tools: list/read tree/section/search/save; graph_neighbors, graph_traverse, graph_search, graph_stats, describe_node; folder_tree; search_conversations, read_conversation; lint_knowledge, compile_knowledge (native Ollama JSON-schema tool calling)
- **Knowledge graph** — in-memory graph with SIMILAR, INTER_FILE, CROSS_DOMAIN, PARENT_CHILD, REFERENCES (wiki-link / markdown / prose), RELATES_TO edges. Edge provenance (`link_text`, `link_kind`, `evidence`) surfaced in graph tools.
- **Medallion architecture** — `canon/` (gold, read-only), `knowledge/wiki/` (silver, agent-writable), `knowledge/raw/` (bronze, source material). `save_knowledge` always lands under `wiki/`; search ranks canon > wiki > raw at equal similarity.
- **LLM Wiki Pattern alignment** — focused single-concept pages with explicit `[[wiki-link]]`s; `compile_knowledge(source)` plans new pages from raw/ inputs; `lint_knowledge()` flags omnibus-file warning signs (oversized chunks, heading collisions, orphans, broken links). See ARCHITECTURE.md for full pattern.
- **Heading trees with token costs** — agent sees structure and size before loading sections
- **Adaptive tool budget** — max 15 section loads, min 8000 tokens remaining; per-class per-turn caps (explore/write/maintenance); `compile_knowledge` and `save_knowledge` **each** consume one **write** slot; tool traffic capped at ~50% of context window; honest in-band refusals
- **Streaming tool loop** — Ollama native `tool_calls`; execute in `asyncio.to_thread`, feed `role="tool"` results back; per-class budgets + context cap + dedup of identical `(name, args)`; max 10 iterations
- **Honest tool framing** — `[TOOL_RESULT: name | COMPLETE ...]` or `TRUNCATED ...`; refusals / duplicate skips use **`NOT_EXECUTED`** (never labeled `COMPLETE`). `read_knowledge_section` self-reports via `[SECTION: ...]` with `offset`. SSE `tool_result` JSON includes `executed` for the UI
- **Section-based chunking** — H1-H5 hierarchy, recursive splitting, document-level LLM overviews
- **LLM summary preservation** — reindex/save/watcher never overwrite stored LLM summaries with mechanical fallbacks; mechanical only fills genuinely new chunks
- **Watcher path-suppression** — `save_knowledge` mutes its own write cascade so each save runs exactly one inline reindex, not four
- **Knowledge base** — semantic search over `knowledge/` and `canon/` markdown files
- **SSE streaming** — thinking tokens; per-iteration lifecycle (`iteration_start`, `tool_call`, `tool_executing`, `tool_done`, `tool_result`, `heartbeat`). Chat UI: one tool bubble per call (`tool_call` creates it; `tool_executing` only drives the timer), summarized args for large payloads (e.g. `save_knowledge.content` → length + preview)
- **KB file watcher** — automatic reindexing when markdown files change
- **Debug logging** — structured JSONL to `/app/logs/` (chat, index, tools)
- **Glassmorphism UI** — dark theme, conversations + KB browser, per-iteration containers, collapsible tool/thinking bubbles, live tool elapsed timers, pulsing thinking indicator

## Tech Stack

| Layer | Technology |
|-------|------------|
| Container | Docker (`python:3.12-slim`, non-root user) |
| Runtime | Python 3.12 |
| Web Server | FastAPI + Uvicorn |
| Model Inference | Ollama Mac app (HTTP API at `host.docker.internal:11434`) |
| Embeddings | Google Gemini `gemini-embedding-001` (768-dim, cloud) or Ollama `nomic-embed-text` (768-dim, local) |
| Conversation Store | JSON files (`/app/sessions/`) |
| Vector Search | LanceDB (Lance format, 768-dim embeddings) |
| Knowledge Graph | In-memory `KnowledgeGraph` class (JSON persistence) |
| Token Counting | tiktoken (cl100k_base) |
| Knowledge Base | Markdown files (Obsidian-compatible), section-based chunking |
| UI | Vanilla JS + CSS (dark glassmorphism theme) |

## Storage Architecture

| System | Location | Purpose | Status |
|--------|----------|---------|--------|
| Chat sessions | `/app/sessions/*.json` | CRUD, auto-title, history, tool-call metadata | Active |
| KB vectors | `/app/lancedb/` | Semantic search over markdown | Active |
| KB graph | `/app/lancedb/graph.json` | Graph edges and nodes | Active |
| Debug logs | `/app/logs/*.log` | Structured JSONL per module | Active |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | UI homepage |
| `/conversations` | GET | List conversations (most recent first) |
| `/conversations` | POST | Create new conversation |
| `/conversations/{id}` | GET | Get conversation turns |
| `/conversations/{id}` | DELETE | Delete conversation |
| `/chat` | POST | Send message, get SSE stream (requires `conversation_id`) |
| `/models` | GET | List available models + current |
| `/model` | POST | Switch model |
| `/kb/knowledge` | GET | List knowledge files |
| `/kb/canon` | GET | List canon files (with subfolders) |
| `/kb/file/{path}` | GET | Get file content |
| `/kb/search?q=` | GET | Semantic search |
| `/kb/stats` | GET | Index statistics (incl. graph stats + embedding model) |
| `/kb/folder-tree` | GET | Folder hierarchy tree for LLM consumption |
| `/kb/reindex` | POST | Rebuild index (body: `{entities: true, summaries: true}`) |
| `/sse?token=` | GET | SSE keepalive connection |

## Project Structure

```
llm-wiki-agent/
├── src/
│   ├── main.py              # Entry point — init, start, cleanup
│   ├── debug_log.py         # Structured JSONL logging to /app/logs/
│   ├── models/
│   │   └── gateway.py       # Ollama HTTP client, streaming, model switch
│   ├── web/
│   │   └── app.py           # FastAPI server, tool loop, SSE, conversations, KB
│   ├── memory/
│   │   └── store.py         # JSON session persistence (CRUD, auto-title)
│   ├── knowledge/
│   │   ├── index.py         # LanceDB index, Gemini/Ollama embeddings, doc summaries, graph
│   │   ├── chunker.py       # H1-H5 heading trees, token costs, recursive chunk splitting
│   │   ├── graph.py         # In-memory knowledge graph, folder tree, traversal
│   │   ├── wiki_links.py    # [[wiki-link]] parsing and resolution
│   │   └── prose_bridges.py # Prose reference extraction for graph edges
│   └── agent/
│       ├── runtime.py       # Agent loop (stub — tool loop in app.py)
│       ├── tools.py          # 14 KB tools + per-class budgets, Obsidian writes, lint/compile, conversation tools
│       ├── tokenizer.py      # Token counting (cl100k_base), slice_tokens, sentence-boundary truncate
│       ├── kb_paths.py       # Canonical filename helpers
│       └── watcher.py       # KB file watcher
├── ui/
│   ├── index.html           # Main UI — conversation list, KB browser
│   ├── app.js               # Frontend — SSE, markdown, sessions, tool events, highlight.js
│   ├── kb-graph-hud.js      # Self-contained graph HUD overlay
│   └── style.css            # Glassmorphism dark theme, tool call bubbles
├── tests/                    # ~723 unit+integration tests (e2e/evals separate); KB, graph, API, tools, watcher
├── knowledge/                # Writable KB (Obsidian-compatible)
├── canon/                    # Read-only KB (agent cannot modify)
├── Dockerfile                # python:3.12-slim, non-root user
├── docker-compose.yml        # Volume mounts, env_file, host.docker.internal for Ollama
├── requirements.txt          # Python deps
├── ARCHITECTURE.md           # Full technical documentation
└── LICENSE                   # MIT
```

## Testing

Rebuild the image after changing `src/` or `tests/`. Then:

```bash
docker exec llm-wiki-agent timeout 600 python -m pytest tests/ --ignore=tests/e2e --ignore=tests/evals -q
```

Expect **724 passed**, 0 skipped (main suite). E2E needs live Ollama: `docker exec llm-wiki-agent python -m pytest tests/e2e -v`.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `gemini` | `gemini` for cloud API, `ollama` for local nomic-embed-text |
| `GOOGLE_GEMINI_API_KEY` | (required) | Gemini API key for embeddings (starts with `AIza...`) |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Default chat model name |
| `SUMMARY_MODEL` | `llama3.1:8b` | Model for document-level LLM summaries |
| `LANCEDB_DIR` | `/app/lancedb` | LanceDB data directory |

### Volume Mounts (docker-compose.yml)

| Host | Container | Purpose |
|------|-----------|---------|
| `./knowledge` | `/app/knowledge` | Writable KB (shared with Obsidian) |
| `./canon` | `/app/canon` | Read-only reference material |
| `./lance-data` | `/app/lancedb` | LanceDB vector store |

### Switching Embedding Providers

To switch between Gemini and Ollama embeddings:

1. Edit `.env`: set `EMBEDDING_PROVIDER=ollama` (or `gemini`)
2. **Wipe lance data**: `rm -rf lance-data/*` (embedding spaces are incompatible)
3. Rebuild: `docker compose up --build -d`

## Development Principles

- Spec before code
- Tests are proof — "seems right" is not done
- Small tasks (~5 files max)
- Vertical slices
- Surface assumptions explicitly
- Track LOC/module metrics in ARCHITECTURE.md
- No sensitive data in git-tracked files

## Current State

**Working:**
- Chat with Ollama models (streaming SSE)
- 14 KB tools (incl. `describe_node`, `search_conversations`, `read_conversation`, graph drill-down params, lint/compile)
- Knowledge graph with SIMILAR/INTER_FILE/CROSS_DOMAIN/PARENT_CHILD/REFERENCES/RELATES_TO edges with provenance (`link_text`, `link_kind`, `evidence`)
- Medallion KB layout: `canon/` (gold) > `knowledge/wiki/` (silver) > `knowledge/raw/` (bronze); tier-weighted search ranking
- LLM Wiki Pattern: focused single-concept pages, `[[wiki-link]]` parsing → REFERENCES edges, `compile_knowledge(source)` planning, `lint_knowledge()` omnibus warnings
- Gemini embeddings (768-dim, cloud, fast indexing ~1 min)
- Document-level LLM summaries (opt-in via `/kb/reindex?summaries=true`); LLM summaries survive every reindex/save/watcher event
- Heading trees with token costs and section summaries
- Structured debug logging (JSONL to /app/logs/)
- Auto-nudge when model goes silent after tool results
- Honest tool-result framing (`COMPLETE` / `TRUNCATED` / `NOT_EXECUTED`; sentence-boundary truncation; `offset` on `read_knowledge_section`; SSE `tool_result.executed`)
- Per-iteration UI containers, SSE heartbeats, collapsible tool/thinking bubbles (amber styling when `executed` is false), live tool elapsed timers
- Token-budgeted history walk + intra-turn 50%-of-context cap (no Turn 2-3 deadlock; older tool turns rendered as compact stubs)
- **Tool-loop guard** — within- and cross-iteration dedup of identical `(name, args)`; per-class per-turn budgets; early break when tool traffic exceeds ~50% of context window
- **Graph addressing** — heading-only lookup, `"file > heading"` shortcut, `graph_stats` paths round-trip through `graph_neighbors`, `graph_neighbors` pagination via `offset`/`edge_type`/`limit`
- **`graph_search` rich output** — filename + heading + score + summary

**Known Issues:**
- Empty search results (`search_knowledge("")`, `graph_search("")`) return "No matches" instead of falling back to `list_knowledge()` / graph stats.
- KB watcher emits multiple `on_modified` events per single save; `suppress_paths` covers `save_knowledge`'s own self-cascade but not the duplicate-event-per-save case in general (no per-path debounce yet).
- `build_index(force=False)` skips unchanged files via mtime but never deletes chunks for files removed from the KB (orphaned-chunk sweep is open work).

## Next Steps

1. **UI: Expand/Collapse all toggle** — header button to expand/collapse every tool and thinking bubble at once. Collapsed bubbles aren't included in copy-paste; expanded ones are. Worth surfacing as a real control.
2. **Watcher debouncing** — 250ms per-path debounce inside `KBEventHandler` to collapse the duplicate-event-per-save problem.
3. **Empty-query fallbacks** — `graph_search("")` → graph stats + usage hint; `graph_neighbors()` with no results → suggest `list_knowledge()`.
4. **Orphaned chunk cleanup** — sweep stored filenames vs current files on startup, drop chunks for deleted files.
5. ~~**Ollama native tool calling**~~ — shipped: `tools=[...]` JSON Schema on `/api/chat`; legacy `[TOOL: ...]` parser removed.

## License

[MIT](LICENSE)