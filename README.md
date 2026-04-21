# LLM Wiki Agent

Implementation of the [LLM Wiki Pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — medallion-tiered storage, hybrid inference, knowledge graph. Runs in Docker. Obsidian-compatible markdown as source of truth.

[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg)](https://fastapi.tiangolo.com/) [![Docker](https://img.shields.io/badge/Docker-2496ED.svg)](https://www.docker.com/) [![Ollama](https://img.shields.io/badge/Ollama-000000.svg)](https://ollama.ai/)

<img src="static-assets/images/demo.gif" alt="LLM Wiki Agent UI" width="780" />

Switch models at runtime from the UI dropdown. Any Ollama model works.

## Why

Most agents pick a side: local for privacy, cloud for speed. This doesn't. Ollama locally for sensitive work. Cloud models for heavier reasoning. Same KB either way. You decide per session.

1. **LLM Wiki Pattern as a working system.** Single-concept pages, `[[wiki-links]]`, graph traversal as navigation — [Karpathy's pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), built out.
2. **Medallion tiers.** `canon/` (gold, read-only) > `knowledge/wiki/` (silver, agent-writable) > `knowledge/raw/` (bronze, source). Search ranks accordingly. Agent reads sources, never overwrites them.
3. **Honest tool execution.** Results labeled `COMPLETE`, `TRUNCATED`, or `NOT_EXECUTED`. Skipped and duplicate calls never marked complete. UI shows the difference.

## Inference Modes

| Use Case | Chat | Embeddings | Network |
|----------|------|------------|---------|
| Fully local | Ollama (local) | Ollama (local) | None |
| Hybrid | Ollama (local or cloud) | Gemini (cloud) | Embeddings only |
| Fully cloud | Ollama Cloud | Gemini (cloud) | Required |

## Quick Start

```bash
git clone https://github.com/thefilesareinthecomputer/llm-wiki-agent.git
cd llm-wiki-agent

echo "GOOGLE_GEMINI_API_KEY=AIza..." > .env
echo "EMBEDDING_PROVIDER=gemini" >> .env

docker compose up --build -d

# http://localhost:8080
```

Docker Desktop + [Ollama](https://ollama.ai) with a model pulled + Gemini API key.

`src/` and `tests/` are copied at build time, not bind-mounted. After edits: `docker compose down && docker compose up --build -d`, then test in container.

## What it does

| | |
|---|---|
| **15 KB tools** | list, read tree/section, search, save; graph neighbors/traverse/search/stats, describe node, folder tree; search/read conversations; lint, compile — native Ollama JSON-schema tool calling |
| **Knowledge graph** | 6 edge types (SIMILAR, INTER_FILE, CROSS_DOMAIN, PARENT_CHILD, REFERENCES, RELATES_TO). Edge provenance (`link_text`, `link_kind`, `evidence`) in every graph tool response |
| **Medallion architecture** | `canon/` > `knowledge/wiki/` > `knowledge/raw/`. Agent writes only to wiki. Search ranks canon above wiki above raw |
| **LLM Wiki Pattern** | Single-concept pages, `[[wiki-links]]`, `compile_knowledge(source)` plans pages from raw, `lint_knowledge()` flags omnibus files/orphans/broken links |
| **Honest tool framing** | `COMPLETE` / `TRUNCATED` / `NOT_EXECUTED` — skipped and duplicate calls never labeled complete |
| **Adaptive budgets** | 15 section loads max per turn, 8K tokens min remaining, per-class caps, ~50% context cap on tool traffic |
| **Streaming tool loop** | Ollama native `tool_calls`, `asyncio.to_thread` execution, dedup of identical `(name, args)`, SSE lifecycle events |
| **Obsidian-compatible** | Frontmatter, `[[wiki-links]]`, heading trees with token costs — works as a vault |

<details>
<summary>Full feature list</summary>

- Local inference via Ollama — no cloud API required for chat
- Gemini embeddings (gemini-embedding-001, 768-dim, cloud) or Ollama (nomic-embed-text, local)
- Model switching at runtime
- Conversation sessions — create, switch, delete; persists across reloads
- Heading trees with token costs — agent sees structure before loading
- Section-based chunking — H1-H5 hierarchy, recursive splitting, doc-level LLM overviews
- LLM summary preservation — reindex/save/watcher never overwrite stored LLM summaries
- Watcher path-suppression — save triggers one inline reindex, not four
- SSE streaming — thinking tokens, per-iteration lifecycle events
- KB file watcher — auto reindex on markdown changes
- Structured JSONL debug logging
- Glassmorphism UI — dark theme, collapsible tool/thinking bubbles, live elapsed timers
</details>

## Tech Stack

| Layer | Technology |
|-------|------------|
| Container | Docker (`python:3.12-slim`, non-root) |
| Runtime | Python 3.12 |
| Web Server | FastAPI + Uvicorn |
| Inference | Ollama (`host.docker.internal:11434`) |
| Embeddings | Gemini `gemini-embedding-001` (cloud) or Ollama `nomic-embed-text` (local) |
| Vector Search | LanceDB (768-dim) |
| Knowledge Graph | In-memory, JSON persistence |
| Token Counting | tiktoken (cl100k_base) |
| Knowledge Base | Markdown, Obsidian-compatible, section-based chunking |
| UI | Vanilla JS + CSS |

## Project Structure

```
llm-wiki-agent/
├── src/
│   ├── main.py              # Entry point
│   ├── debug_log.py         # Structured JSONL logging
│   ├── models/
│   │   └── gateway.py       # Ollama client, streaming, model switch
│   ├── web/
│   │   └── app.py           # FastAPI, tool loop, SSE, conversations, KB
│   ├── memory/
│   │   └── store.py         # JSON session persistence
│   ├── knowledge/
│   │   ├── index.py         # LanceDB, embeddings, summaries, graph
│   │   ├── chunker.py       # H1-H5 heading trees, chunk splitting
│   │   ├── graph.py         # Knowledge graph, folder tree, traversal
│   │   ├── wiki_links.py    # [[wiki-link]] parsing
│   │   └── prose_bridges.py # Prose reference extraction
│   └── agent/
│       ├── tools.py          # 15 KB tools, budgets, Obsidian writes
│       ├── tokenizer.py      # Token counting, slice, truncate
│       ├── kb_paths.py       # Canonical filename helpers
│       └── watcher.py       # KB file watcher
├── ui/
│   ├── index.html
│   ├── app.js               # SSE, markdown, sessions, tool events
│   ├── kb-graph-hud.js      # Graph HUD overlay
│   └── style.css            # Dark glassmorphism theme
├── tests/                    # ~723 unit + integration; e2e/evals separate
├── knowledge/                # Writable KB
├── canon/                    # Read-only KB
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── ARCHITECTURE.md
```

## Testing

```bash
docker exec llm-wiki-agent timeout 600 python -m pytest tests/ --ignore=tests/e2e --ignore=tests/evals -q
```

724 passed, 0 skipped. E2E needs live Ollama: `docker exec llm-wiki-agent python -m pytest tests/e2e -v`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `gemini` | `gemini` (cloud) or `ollama` (local) |
| `GOOGLE_GEMINI_API_KEY` | (required) | Starts with `AIza...` |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Default chat model |
| `SUMMARY_MODEL` | `llama3.1:8b` | LLM summary model |
| `LANCEDB_DIR` | `/app/lancedb` | LanceDB data dir |

**Volume mounts:** `./knowledge` → `/app/knowledge`, `./canon` → `/app/canon`, `./lance-data` → `/app/lancedb`

**Switching embedding providers:** set `EMBEDDING_PROVIDER=ollama` in `.env`, wipe `lance-data/*` (spaces are incompatible), rebuild.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) — module map, data flow, embedding pipeline, graph construction, tool loop, design decisions.

## License

[MIT](LICENSE)