<div align="center">

# LLM Wiki Agent

**Your knowledge base, maintained by an AI that actually reads.**

A working implementation of the [LLM Wiki Pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) —
medallion-tiered storage, hybrid local-or-cloud inference, and a knowledge graph that grows as you use it.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED.svg)](https://www.docker.com/)
[![Ollama](https://img.shields.io/badge/Ollama-000000.svg)](https://ollama.ai/)

<br/>

<img src="static-assets/images/demo.gif" alt="LLM Wiki Agent — dark glassmorphism UI with streaming tool calls, knowledge graph HUD, and conversation management" width="780" />

*Any Ollama model works — switch at runtime from the UI dropdown.*

</div>

---

## Why this exists

Most AI agents force a choice: fully local for privacy, or fully cloud for speed. This one doesn't.

Use Ollama locally for sensitive material. Switch to cloud models for heavier reasoning. Embeddings local (Ollama) or cloud (Gemini). Same knowledge base either way. You decide per session.

**Three ideas this repo explores:**

1. **The LLM Wiki Pattern as a working system.** Focused single-concept pages, explicit `[[wiki-links]]`, graph traversal as the primary navigation mode — the pattern [Karpathy described](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), built and extended.
2. **Medallion tiers for personal knowledge.** `canon/` (read-only gold) > `knowledge/wiki/` (agent-writable silver) > `knowledge/raw/` (source bronze). Search ranks tiers accordingly — the agent can read your sources but never overwrite them.
3. **Honest tool execution.** The agent never reports work it didn't do. Tool results are labeled `COMPLETE`, `TRUNCATED`, or `NOT_EXECUTED`. The UI surfaces the distinction in real time.

## Inference Modes

| Use Case | Chat Model | Embeddings | Network |
|----------|------------|------------|---------|
| Fully local / sensitive data | Ollama (local) | Ollama (local) | None required |
| **Hybrid (recommended)** | **Ollama (local or cloud)** | **Gemini (cloud)** | **Embeddings only** |
| Fully cloud / maximum speed | Ollama Cloud | Gemini (cloud) | Required |

Switch chat models at runtime from the UI. Embedding provider switches via `EMBEDDING_PROVIDER` in `.env` (requires a lance-data wipe — embedding spaces aren't compatible across providers).

---

## Quick Start

```bash
git clone https://github.com/thefilesareinthecomputer/llm-wiki-agent.git
cd llm-wiki-agent

# Create .env with your Gemini API key
echo "GOOGLE_GEMINI_API_KEY=AIza..." > .env
echo "EMBEDDING_PROVIDER=gemini" >> .env

docker compose up --build -d

# Open http://localhost:8080
```

**Prerequisites:** Docker Desktop, [Ollama](https://ollama.ai) with at least one model pulled, Google Gemini API key.

**Changing code:** `src/` and `tests/` are copied into the image at build time. After edits, rebuild: `docker compose down && docker compose up --build -d`, then run tests in the container.

---

## What it does

| | |
|---|---|
| **15 KB tools** | list, read tree/section, search, save; graph neighbors/traverse/search/stats, describe node, folder tree; search/read conversations; lint, compile — all via native Ollama JSON-schema tool calling |
| **Knowledge graph** | In-memory graph with 6 edge types (SIMILAR, INTER_FILE, CROSS_DOMAIN, PARENT_CHILD, REFERENCES, RELATES_TO) and edge provenance — `link_text`, `link_kind`, `evidence` surfaced in every graph tool response |
| **Medallion architecture** | `canon/` (gold, read-only) > `knowledge/wiki/` (silver, agent-writable) > `knowledge/raw/` (bronze, source). Search ranks canon above wiki above raw at equal similarity |
| **LLM Wiki Pattern** | Single-concept pages with `[[wiki-links]]`; `compile_knowledge(source)` plans new pages from raw inputs; `lint_knowledge()` flags omnibus files, orphans, and broken links |
| **Honest tool framing** | `COMPLETE` / `TRUNCATED` / `NOT_EXECUTED` — the agent never labels a skipped or duplicate tool call as complete. UI shows the distinction in real time |
| **Adaptive budgets** | Max 15 section loads per turn, min 8K tokens remaining, per-class caps (explore/write/maintenance), ~50% context cap on tool traffic, honest in-band refusals |
| **Streaming tool loop** | Ollama native `tool_calls`; `asyncio.to_thread` execution; dedup of identical `(name, args)` pairs; SSE lifecycle events per iteration |
| **Obsidian-compatible** | Markdown files with frontmatter, `[[wiki-links]]`, heading trees with token costs — works as an Obsidian vault out of the box |

<details>
<summary><strong>Full feature list</strong></summary>

- Local inference via Ollama — no cloud API required for chat
- Gemini embeddings — fast cloud-based vector search (gemini-embedding-001, 768-dim)
- Model switching at runtime from the UI dropdown
- Conversation sessions — create, switch, delete; history persists across reloads
- Heading trees with token costs — agent sees structure and size before loading sections
- Section-based chunking — H1-H5 hierarchy, recursive splitting, document-level LLM overviews
- LLM summary preservation — reindex/save/watcher never overwrite stored LLM summaries with mechanical fallbacks
- Watcher path-suppression — `save_knowledge` mutes its own write cascade so each save triggers one inline reindex, not four
- SSE streaming — thinking tokens, per-iteration lifecycle (`iteration_start`, `tool_call`, `tool_executing`, `tool_done`, `tool_result`, `heartbeat`)
- KB file watcher — automatic reindexing when markdown files change
- Structured debug logging — JSONL to `/app/logs/`
- Glassmorphism UI — dark theme, conversations + KB browser, collapsible tool/thinking bubbles, live tool elapsed timers
</details>

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Container | Docker (`python:3.12-slim`, non-root user) |
| Runtime | Python 3.12 |
| Web Server | FastAPI + Uvicorn |
| Model Inference | Ollama (HTTP API at `host.docker.internal:11434`) |
| Embeddings | Google Gemini `gemini-embedding-001` (cloud) or Ollama `nomic-embed-text` (local) |
| Vector Search | LanceDB (Lance format, 768-dim) |
| Knowledge Graph | In-memory `KnowledgeGraph` class (JSON persistence) |
| Token Counting | tiktoken (cl100k_base) |
| Knowledge Base | Markdown (Obsidian-compatible), section-based chunking |
| UI | Vanilla JS + CSS (dark glassmorphism theme) |

---

## Project Structure

```
llm-wiki-agent/
├── src/
│   ├── main.py              # Entry point — init, start, cleanup
│   ├── debug_log.py         # Structured JSONL logging
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
│       ├── tools.py          # 15 KB tools, per-class budgets, Obsidian writes, lint/compile
│       ├── tokenizer.py      # Token counting, slice_tokens, sentence-boundary truncate
│       ├── kb_paths.py       # Canonical filename helpers
│       └── watcher.py       # KB file watcher
├── ui/
│   ├── index.html           # Main UI — conversation list, KB browser
│   ├── app.js               # Frontend — SSE, markdown, sessions, tool events
│   ├── kb-graph-hud.js      # Self-contained graph HUD overlay
│   └── style.css            # Glassmorphism dark theme
├── tests/                    # ~723 unit + integration tests; e2e/evals separate
├── knowledge/                # Writable KB (Obsidian-compatible)
├── canon/                    # Read-only KB (agent cannot modify)
├── Dockerfile                # python:3.12-slim, non-root user
├── docker-compose.yml        # Volume mounts, env_file, host.docker.internal for Ollama
├── requirements.txt          # Python deps
└── ARCHITECTURE.md           # Full technical documentation
```

---

## Testing

```bash
docker exec llm-wiki-agent timeout 600 python -m pytest tests/ --ignore=tests/e2e --ignore=tests/evals -q
```

**724 passed**, 0 skipped (main suite). E2E needs live Ollama: `docker exec llm-wiki-agent python -m pytest tests/e2e -v`.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `gemini` | `gemini` for cloud API, `ollama` for local nomic-embed-text |
| `GOOGLE_GEMINI_API_KEY` | (required) | Gemini API key for embeddings (starts with `AIza...`) |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Default chat model |
| `SUMMARY_MODEL` | `llama3.1:8b` | Model for document-level LLM summaries |
| `LANCEDB_DIR` | `/app/lancedb` | LanceDB data directory |

### Volume Mounts

| Host | Container | Purpose |
|------|-----------|---------|
| `./knowledge` | `/app/knowledge` | Writable KB (shared with Obsidian) |
| `./canon` | `/app/canon` | Read-only reference material |
| `./lance-data` | `/app/lancedb` | LanceDB vector store |

### Switching Embedding Providers

1. Edit `.env`: set `EMBEDDING_PROVIDER=ollama` (or `gemini`)
2. **Wipe lance data**: `rm -rf lance-data/*` (embedding spaces are incompatible)
3. Rebuild: `docker compose up --build -d`

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical walkthrough: module map, data flow, embedding pipeline, knowledge graph construction, tool loop mechanics, and resolved design decisions.

---

## License

[MIT](LICENSE)