# LLM Wiki Agent Architecture

## Module Map

```
src/
├── main.py                  (103)  Entry point, async startup, server + watcher
├── web/
│   └── app.py              (1857)  FastAPI server, SSE streaming, tool loop, context compaction, system prompt (L0-L4), graph subgraph/overview API
├── models/
│   └── gateway.py           (244)  LLM gateway (Ollama SDK async), SUPPORTS_TOOLS_MODELS allowlist, TOOL_CAPABLE_FALLBACKS, CLOUD_MODELS
├── knowledge/
│   ├── index.py            (1986)  KB indexing, LanceDB, Gemini/Ollama embeddings, section summaries, mtime check, `_build_graph_edges` (SIMILAR cap + rank)
│   ├── chunker.py           (475)  Markdown heading chunker, heading trees, context headers
│   └── graph.py             (630)  In-memory knowledge graph, folder tree (`format_folder_tree` + optional `root_path` drill-down), traversal, entity extraction; `get_stats()` returns `edge_share` + per-hub `share_non_pc` (P0-2)
├── agent/
│   ├── tools.py            (4188)  KB tools (15: 13 base + 2 conversation), per-class budgets (orient/explore/write/maintenance), write formatting (Obsidian frontmatter: block-list tags/aliases, date-only dates, tag save validation; wiki-link conversion; ## Related), `describe_node`/`graph_traverse`/`folder_tree` Phase D params, caller-aware `_resolve_chunk_nodes` + `query=` reranker (P0-1), `exclude_edge_types='parent_child'` accept-and-note (P0-3), lint (flat-similarity clusters + wiki hygiene), compile (relative `../../canon/` cite examples), source-citation validation
│   ├── kb_paths.py          (201)  Canonical filename helpers: to_canonical / from_canonical (`<source>:<relpath>`)
│   ├── tokenizer.py         (129)  cl100k_base token counting + slice_tokens + sentence-boundary truncate
│   ├── runtime.py           (127)  Agent loop stub (not wired — tool loop in app.py)
│   └── watcher.py           (129)  KB file watcher + suppress_paths registry (avoids save_knowledge cascade)
├── memory/
│   └── store.py             (225)  JSON conversation store + token-budgeted history walk + tool-call metadata
└── debug_log.py              (62)  Structured JSONL logging to /app/logs/

ui/
├── index.html                      HTML shell — loads app.js, style.css, kb-graph-hud.js; multi-line textarea input
├── app.js                          Chat UI: SSE handling, tool bubbles (auto-collapse on turn end), context usage display, textarea auto-grow
├── style.css                       Gunmetal neural dark theme — consistent across all UI elements
└── kb-graph-hud.js          (919)  Self-contained graph HUD overlay — intercepts SSE, heat-driven radial layout, click-to-pin tooltips

                                                              ─────────
                                                       Total: ~10,400 lines (src)
```

## Data Flow

```
User message
    │
    ├─► L0-L5 system prompt assembly (app.py:_build_system_prompt)
    │   L0: Identity + voice (Q/Alfred/Jarvis tone, boundaries)
    │   L1: Behavioral framework (Mind-en-Place rules, domain competences)
    │   L2: RAG context (if KB hits)
    │   L3: Session history
    │   L4: Tool literacy (mental model, playbook, search discipline, edge types, budgets)
    │   L5: Mempalace (future)
    │
    ├─► Model Gateway → Ollama /api/chat (streaming SSE)
    │   │
    │   ├─► thinking tokens (URL-encoded) ──► event: thinking
    │   └─► content tokens (URL-encoded) ──► event: token
    │
    ├─► message.tool_calls present? (Ollama native tool calling)
    │   YES ──► Per-class budget check (explore/orient/write/maintenance) + per-turn dedup
    │           ──► tool_executing SSE ──► Execute by name in asyncio.to_thread
    │           ──► tool_done SSE ──► Frame result with [TOOL_RESULT: name | COMPLETE|TRUNCATED N of M chars]
    │                                  + remaining_budget footer
    │           ──► tool_result SSE ──► Feed back as role="tool" message ──► Stream again
    │           ──► Context compaction at 60% of window (stub old tool results, keep latest iteration)
    │           ──► Hard brake at 85% post-compaction; Ollama 500 recovery with graceful finalization
    │   NO  ──► Done
    │
    ├─► Lifecycle SSE: iteration_start (per pass), heartbeat (every 3s during silent gaps)
    │
    └─► Auto-nudge: if model produces no content after tool results, inject nudge message
```

## Embedding Pipeline

```
KB files (.md)
    │
    ├─► chunk_file() ──► Section chunks (H1-H5 headings)
    │
    ├─► GeminiEmbeddingFunction (cloud, 768-dim, ~200ms/batch)
    │   OR OllamaEmbeddingFunction (local nomic-embed-text, ~2min/439 chunks)
    │   Configured via EMBEDDING_PROVIDER env var
    │
    ├─► LanceDB table: knowledge_base
    │   Each chunk: {id, vector, document, filename, source, heading, ...}
    │   Columns: id, vector (768-dim), document, filename, source, heading,
    │             chunk_index, summary, token_count, mtime, file_tokens,
    │             section_count, file_outline, path, folder, type
    │             file_tokens, section_count, file_outline, path, folder, type
    │   type: "section" (regular) or "overview" (doc-level LLM summary)
    │
    ├─► _build_graph_edges() ──► SIMILAR (capped + ranked), INTER_FILE, CROSS_DOMAIN, PARENT_CHILD, …
    │
    └─► KnowledgeGraph ──► JSON persistence (`graph.json` under mounted `/app/lancedb` → host `./lance-data/`)
```

## Knowledge Graph

```
Node types: CHUNK, FOLDER, ENTITY, CONCEPT
Edge types: SIMILAR, INTER_FILE, CROSS_DOMAIN, PARENT_CHILD, REFERENCES, RELATES_TO

Graph construction:
1. _init_graph_nodes_only() ── FOLDER nodes from build_folder_tree(), CHUNK nodes from LanceDB table
2. _build_graph_edges() ── Three-branch similarity + heading cross-file matching:
   - SIMILAR: same file, cosine > `INTRA_FILE_SIMILAR_THRESHOLD` (0.78); at most `INTRA_FILE_SIMILAR_CAP` (5) edges per source node, ranked by weight; each edge stores `attributes.intra_rank` / `intra_total` for tool provenance (`rank N/M in file` via `_format_edge_provenance`).
   - INTER_FILE: same source, different file, cosine > 0.55
   - CROSS_DOMAIN: different source, cosine > 0.60
   - Heading-name matching: same leaf heading in 2+ different files → INTER_FILE (weight 0.65)
   - PARENT_CHILD from heading hierarchy + folder tree
3. _extract_entities() ── ENTITY/CONCEPT nodes + RELATES_TO edges via langextract (optional, slow)

Folder IDs match between build_folder_tree() and chunk metadata:
  build_folder_tree: folder node name = relative path (e.g., "ai")
  chunk metadata folder = relative path to base_dir (e.g., "ai")
  → folder_id = "folder_{source}_{folder.replace('/', '_')}" matches both sides
```

## LLM Wiki Pattern (Karpathy alignment)

The KB is structured as an **agent-maintained wiki**: many small markdown
pages, each one focused on a single concept and explicitly cross-linked.
This is the design pattern Andrej Karpathy described as the LLM-native
shape for long-form knowledge — and it's what every retrieval, write, and
graph mechanism in `src/knowledge/` and `src/agent/` is optimized for.

### Medallion Tiers (four-tier)

| Tier   | Path                | Writable | Role |
|--------|---------------------|----------|------|
| canon  | `canon/`            | no       | Gold — curated, authoritative reference. Cited as ground truth. |
| wiki   | `knowledge/wiki/`   | yes      | Silver — agent-compiled focused pages, one concept per file. |
| memory | `knowledge/memory/` | yes      | Memory — short-lived notes / journal pages. Same chunker + frontmatter as wiki, lower search weight. Conversations are also part of this tier via `search_conversations` / `read_conversation`. |
| raw    | `knowledge/raw/`    | no       | Bronze — source material (transcripts, dumps, notes). Compiled FROM, not edited. |

`save_knowledge()` accepts `wiki/` and `memory/` prefixes (or flat names → `wiki/`) and refuses `raw/` and `canon/`. Search ranking applies `TIER_SEARCH_WEIGHTS` (canon boosted, raw suppressed, memory between raw and wiki) so the most authoritative material wins at equal cosine similarity.

### Anti-pattern: Omnibus Files

A single 400-section "everything I know about X" markdown file is the
shape that breaks every part of this design:

- **Embedding noise** — the chunker still splits on H1-H5, but a giant
  file full of weakly-related sections produces chunks whose vector
  centroid drifts toward the file's mean topic, not the section's actual
  meaning. Search-by-meaning collapses into search-by-table-of-contents.
- **Heading-path collisions** — generic headings like "Overview" /
  "Notes" / "References" repeat across an omnibus and dominate
  `graph_neighbors` results, drowning out signal from focused pages.
- **Edit blast radius** — touching one section forces a re-embed of
  every chunk in the file (and a graph-edge rebuild for every
  neighbor). On a 400-section omnibus that's hundreds of unnecessary
  Gemini calls per save.
- **Wiki-link impossibility** — `[[topic]]` only works when there's a
  page named `topic.md`. Concepts buried inside an omnibus can't be
  linked to without a section anchor, and section anchors don't survive
  rewrites.
- **Compilation paralysis** — when every related concept lives in the
  same file, the agent can't tell what's prior knowledge vs new claim;
  the wiki-link graph degenerates into self-loops.

The cure is to compile *focused* pages — one concept per file, ~500-2000
tokens, named after the concept itself (`cortisol.md`, not
`endocrinology-notes.md`). The toolchain enforces this:

- `compile_knowledge(source)` reads a raw/ file (or a conversation ref)
  and returns a compilation prompt that suggests *one* wiki page
  filename, lists related existing pages to link, and surfaces canon
  anchors to cite using **filesystem-relative** markdown
  (`../../canon/...` from the suggested wiki path), not `canon:` /
  `knowledge:` strings inside `(...)` hrefs. Never recommends "dump
  the whole transcript into one file."
- `save_knowledge` defaults under `wiki/`, writes Obsidian-safe YAML
  (block-list `tags` / `aliases`, date-only `created` / `updated`,
  refuses bogus tag shapes), auto-injects a `## Related` block of
  similarity-suggested `[[wiki-link]]`s, and triggers a graph rebuild
  that emits `REFERENCES` edges from those links.
- `lint_knowledge` flags **oversized chunks** (`> 2000 tokens`),
  **heading collisions** (same leaf heading in 3+ files),
  **flat-similarity clusters** (many intra-file SIMILAR edges whose
  weights fall in a narrow band), and **wiki/memory hygiene** on disk
  (nested `tags` / `aliases` in frontmatter; markdown links pointing at
  `canon:...` or `knowledge:...` targets) — signals that layout or
  hand-edited frontmatter may break Obsidian or the vault linker.

### Edge Provenance (P2)

Every `REFERENCES` edge carries `attributes`:

| Attribute      | Source |
|----------------|--------|
| `link_text`    | The display text the agent wrote |
| `link_kind`    | `wiki` (`[[...]]`) / `markdown` (`[](...)`) / `prose` (heuristic) |
| `target_anchor`| Optional `#section` fragment, normalized |
| `evidence`     | Sentence excerpt for prose-bridge edges |

`graph_neighbors`, `graph_search`, and `graph_traverse` render this so
the agent can see *why* two chunks are linked, not just *that* they are.
Intra-file SIMILAR edges additionally surface ordinal **rank-in-file**
when `intra_rank` / `intra_total` are present (Phase D), so a tight cosine
band is readable as signal order, not noise.
`graph_stats` rolls these into a Provenance Classes breakdown:
explicit-reference (author intent) vs prose-bridge (model-inferred) vs
similarity (embedding cosine) vs hierarchy (folder/heading nesting).

**Edge-share canonical numbers (P0-2).** `KnowledgeGraph.get_stats()`
emits an `edge_share` dict (per-edge-type fraction of total edges, only
types with `count > 0`) and a `share_non_pc` field on every
`most_connected` entry. `graph_stats` renders a single
`edge_share: parent_child=23.0% similar=58.8% …` line right under the
nodes/edges header so the model never has to recompute fractions, and
`GET /kb/stats` returns the same numbers under `graph.edge_share` for
the UI to consume without a second round-trip. The "99.7% parent_child"
class of model-arithmetic errors stops at the formatter, not at runtime.

**Heading resolver caller-aware suggestions (P0-1).**
`_resolve_chunk_nodes(..., caller, query, kb_index)` is the single
shared resolver for `graph_neighbors` / `graph_traverse` /
`describe_node`. The `caller` arg controls the call form rendered in
ambiguous-match suggestion lines so `describe_node` emits
`describe_node(...)`, not always `graph_neighbors(...)`. Optional
`query` runs a single batched `kb_index._embedding_fn([query, *cands])`
call and reranks candidates by cosine; lexical priority (exact-leaf
match → source-locality → substring position) is the deterministic
fallback when no query is supplied or the embedder fails. All ranking
logic lives in `_rank_candidates`; embedding failures are silent
(never crash, never refuse).

**`exclude_edge_types='parent_child'` accept-and-note (P0-3).**
`graph_traverse` no longer hard-refuses when the caller passes
`parent_child` in the exclusion list. The token is silently dropped (it
is always excluded by construction) and a one-line note prepends the
result. Only genuinely-unknown edge-type tokens still trigger the
existing `Invalid exclude_edge_types` error.

### Maintenance Tools (P3)

- `lint_knowledge(scope="")` — orphans, broken links, heading
  collisions, oversized chunks, flat-similarity clusters, plus wiki/memory
  scans for nested list-shaped `tags`/`aliases` and markdown `(...)` hrefs
  that use `canon:` / `knowledge:` pseudo-targets. `scope=` limits to one
  file where applicable.
- `compile_knowledge(source, query="")` — read-only planning. Reads a
  raw file or conversation slice, finds related wiki + canon, returns a
  structured compilation plan with suggested wiki slug and **relative**
  canon citation examples (`../../canon/...`).

## Tool Loop (app.py — native tool calling)

```
1. Emit iteration_start SSE (per pass)
2. Stream model response with tools=build_tool_registry(kb_tools) on the request
   (heartbeat SSE every 3s during silent gaps)
3. Inspect message.tool_calls (Ollama native). Per-turn dedup of identical
   (name, args) suppresses re-runs.
4. For each surviving tool call:
   a. class_for_tool(name) -> {explore | orient | write | maintenance}; refuse
      if class budget for this turn is exhausted (in-band refusal, no nudge).
   b. Emit tool_call SSE (tool + args for UI), then tool_executing before
      dispatch via asyncio.to_thread.
   c. Emit tool_done SSE (elapsed_ms, executed boolean — false when refused
      or duplicate-skipped without running the handler).
   d. Wrap raw output: [TOOL_RESULT: name | COMPLETE|TRUNCATED N of M chars],
      or **[TOOL_RESULT: name | NOT_EXECUTED]** when executed is false, so
      refusals are never mis-labeled COMPLETE; append remaining_budget footer.
   e. Emit tool_result SSE: framed result, frame_info, and top-level executed
      flag (UI uses this + NOT_EXECUTED text for styling).
   f. Feed framed result back as a role="tool" message keyed by tool_call_id
   g. Stream again (loop continues until model stops emitting tool_calls)
5. Context compaction: before each iteration, if accumulated messages exceed
   COMPACT_TRIGGER (60%) of context window, _compact_messages() fires:
   - System prompt, user question, and most recent iteration kept verbatim
   - Older tool results replaced with 1-line stubs (header only)
   - Older assistant content trimmed to 300 chars
   - Compaction notice injected so model knows context was compressed
   Hard brake at 85% post-compaction; forced summary on any remaining overflow.
   Also fires before the forced-summary stream to prevent Ollama OOM.
6. Ollama 500 recovery: if Ollama returns a 500 mid-loop (model overload)
   and the agent has already produced content, response is finalized with a
   "context limit reached" note instead of surfacing a raw error.
7. If no content after tool results: inject auto-nudge
6. Persist assistant turn with structured tool_calls + tool_results metadata
7. Emit chat_done event

Removed in tool-calling refactor:
  - src/agent/tool_parser.py (regex parser for [TOOL: ...] format)
  - MAX_TOTAL_TOOL_EXECUTIONS hard ceiling (replaced by per-class budgets)
  - forced-summary nudge path (replaced by in-band budget refusals)

Debug logging: /app/logs/chat.log (JSONL)
  Events: chat_start, iteration_start, stream_start, stream_end,
          tool_call_dispatched, tool_execution_start, tool_result_framed,
          tool_results_fed_back, budget_refusal, intra_turn_cap_reached,
          auto_nudge, chat_end
```

## Embedding Providers

| Provider | Model | Dim | Speed | Config |
|----------|-------|-----|-------|--------|
| Gemini (default) | gemini-embedding-001 | 768 | ~200ms/batch of 100 | `EMBEDDING_PROVIDER=gemini` + `GOOGLE_GEMINI_API_KEY` |
| Ollama (fallback) | nomic-embed-text | 768 | ~2min/439 chunks | `EMBEDDING_PROVIDER=ollama` |

Both produce 768-dim vectors. Gemini is cloud-only (paid tier: no training on data, 55-day temp retention). Ollama is local-only.

**Switching providers requires wiping `lance-data/`** — embedding spaces are incompatible.

## Summarization

| Mode | LLM calls | Time | What agent sees |
|------|-----------|------|------------------|
| Mechanical (default) | 0 | Instant | First 250 chars per section |
| LLM section summaries | ~440 | ~13 min (3 workers, supergemma4-26b local) | Dense 2-3 sentence summaries per section + doc overview |

- Mechanical summaries: extracted from first meaningful line of each chunk
- LLM section summaries: parallel per-section summarization via `_generate_section_summaries()` (ThreadPoolExecutor, 3 workers, 60s timeout)
  - Each section gets a dense 2-3 sentence summary with proper nouns, mechanisms, cross-domain bridges
  - Falls back to mechanical on timeout or failure
  - 500 char truncation (vs 250 for mechanical)
- Document overview: one call per file via `_generate_doc_summary()`, creates a "Document Overview" chunk (type=overview)
- Configured via `/kb/reindex` with `{"summaries": true}`
- SUMMARY_MODEL env var controls which model (default: supergemma4-26b)
- Worker count: 3 (Ollama processes ~1-2 requests at a time; 12 workers caused queue backup and timeouts)

## Key Decisions (Resolved)

| Decision | Resolution |
|----------|-----------|
| Embedding model | ~~all-MiniLM-L6-v2 (384-dim)~~ → ~~nomic-embed-text (768-dim, local)~~ → **Gemini gemini-embedding-001 (768-dim, cloud)** |
| Graph DB implementation | Lightweight custom `KnowledgeGraph` (graph.py). JSON persistence, in-memory traversal. |
| Graph edge building | **Batch numpy cosine similarity** on stored embeddings. Zero extra API calls. Replaces per-chunk N+1 storm. |
| LLM summaries | ~~per-chunk batch (439 calls)~~ → **Document-level LLM overviews (1 call per file)**. Section summaries always mechanical. |
| Graph edges | SIMILAR (same **file**, score > 0.78, max 5 per node, ranked), INTER_FILE (same source, different file, > 0.55), CROSS_DOMAIN (different source, > 0.60), PARENT_CHILD (heading hierarchy + folder tree). |
| Tool result nudge | Auto-nudge injected when model produces no content after tool results (empty full_response). |
| Folder path IDs | Chunk metadata `folder` relative to base_dir (e.g., "ai"), matches folder node names from build_folder_tree(). |
| Container crash resilience | `build_index()` wrapped in try/except; server stays up. `main.py` awaits server task only. |
| ModelGateway client | Lazy `@property` on `self._client`. No eager `httpx.AsyncClient()` in `__init__`. |
| Test embeddings | `FakeEmbeddingFunction` in conftest.py — deterministic hash-based 768-dim vectors. Zero Gemini calls in tests. |
| Tool parser | **REMOVED** in tool-calling refactor — replaced by Ollama native tool calling (`tools=[...]` JSON-schema channel). Schemas are derived from each `KBTools` method signature + docstring; `build_tool_registry()` assembles the registry. |
| Tool budgets | Monolithic `MAX_TOTAL_TOOL_EXECUTIONS=30` replaced by per-class budgets: `explore=10` (R13, raised from 8), `orient=5`, `write=2`, `maintenance=3`. `orient` class covers `list_knowledge`, `folder_tree`, `graph_stats`. Every tool result includes a `remaining_budget` footer; forced-summary nudge path deleted. |
| Canonical filenames | All graph nodes, folder-tree entries, resolver, wiki-link builder, and the LanceDB `path` column use `<source>:<relpath>` (`agent/kb_paths.py`). Grep-enforced test forbids raw `.filename` comparisons elsewhere. One-shot startup migration rewrites legacy LanceDB rows. |
| Memory tier | Fourth medallion tier between raw and silver. `save_knowledge` accepts `memory/` prefix; `_compute_tier` recognizes the path; conversations participate via `search_conversations` / `read_conversation`. Search weight sits between raw and wiki. |
| Obsidian frontmatter | `_build_frontmatter` emits `aliases`, `tags` (YAML **block lists**), `created`, `updated` (**date-only** `YYYY-MM-DD`), `source`, `tier`. Reads legacy `date-created` / `last-modified` when loading old files but does not write them back. `created` preserved across edits; `updated` bumps each save. `save_knowledge` refuses tags that flatten empty or look like serialized Python/JSON lists. `_convert_markdown_links_to_wiki` rewrites in-vault `[Text](wiki/x.md)` to `[[x\|Text]]`, skipping canon, external, image, and code spans. Display-text alias preserved on any non-exact match (Obsidian wiki-links are case-sensitive in display). |
| Deterministic Related block | `_compute_related_block` sorts by descending weight then ascending filename. Re-saving a page is a no-op on disk. |
| Conversation-source validation | `save_knowledge` parses `[turn:N](#conversation-id)` citations in `## Sources` and refuses the write if any thread or turn doesn't exist in the live `ConversationStore`. |
| Chunk provenance | Each chunk content includes YAML context header before embedding: `file`, `position` (N/M), `ancestors` (heading path minus own heading), `doc_summary` (first 150 chars). Makes parent context searchable via vector search. Preamble chunks use parent heading name instead of `"(preamble)"`. |
| Vector store | **LanceDB** (embedded, crash-resistant Lance format). Replaces ChromaDB (corruption, stale client, orphaned folders). |
| Chat lock deadlock | `request.is_disconnected()` checks in token stream loop + tool loop start + after stream. Lock releases on `break`. |
| Idempotent startup | `build_index(force=False)` on startup: mtime check per file, skip unchanged. `build_index(force=True)` on `/kb/reindex` and watcher callback: full wipe + rebuild. |
| Honest tool truncation | `_frame_tool_result` adds `COMPLETE`/`TRUNCATED` markers; budget refusals and duplicate skips use **`NOT_EXECUTED`** (never mislabel a short refusal as `COMPLETE`). Sentence-boundary cut; per-tool char overrides (e.g. `graph_traverse=12000`); `read_knowledge_section` self-handles via `[SECTION: ... LOADED N of M tokens (COMPLETE\|TRUNCATED -- offset=N)]` header with `offset` resume. SSE `tool_result` includes `executed` for UI styling. Kills silent clipping that drove model fabrication. |
| Adaptive tool budget | Replaces theatrical `999999` budget. Tracks accumulated tool tokens vs context window (`_tool_token_cap`, `_can_afford_load`). Refusal explains the cap (loads, min remaining, 50% fraction). Per-model context windows in `MODEL_CONTEXT_WINDOWS`. |
| History under budget | `get_history_within_budget(max_tokens)` replaces fixed `n=10`. Walks newest→oldest; old turns with tool metadata rendered as compact `[earlier turn: called X, loaded sections Y from Z]` stubs via `_compact_turn`; last 2 turns kept full. Raw `[TOOL: ...]` brackets defensively stripped from assistant history fed to the model. |
| Per-iteration UI + lifecycle SSE | `.iteration` containers in `.message-content` fix bubble-wipe on markdown re-render (only `.iteration-text` re-renders). Server emits `iteration_start`, `tool_executing`, `tool_done`, `heartbeat` (every 3s); UI shows pulsing "Thinking…", live tool elapsed timers, collapses tool bubbles >2k chars. |
| Intra-turn context cap + compaction | When accumulated messages exceed 60% of context window (`COMPACT_TRIGGER`), `_compact_messages()` replaces older tool results with 1-line header stubs and trims older assistant content to 300 chars, preserving the most recent iteration in full. If still over 85% after compaction, the loop breaks with `forced_summary_reason`. Also fires before the forced-summary stream. If Ollama returns a 500 mid-loop and the agent has already produced content, the response is gracefully finalized instead of erroring. Replaces the old hard 50% break. |
| Tool-loop deadlock guard | Three-layer protection against the model spamming the same tool call. (1) Within-iteration: identical `(name, args)` collapse to one execution. (2) Cross-iteration: a `(name, args)` already executed this turn is skipped with a "you already ran this" system note instead of re-running. (3) Per-class per-turn budgets (explore=10, orient=5, write=2, maintenance=3) refuse in-band when exhausted. The `MAX_TOTAL_TOOL_EXECUTIONS=30` hard ceiling and forced-summary nudge path were removed in the tool-calling refactor (A3). |
| LLM summary preservation | `_index_file` snapshots existing per-heading summaries before delete; mechanical fallback NEVER overwrites a stored summary that differs from `_mechanical_summary(content)` for that chunk. LLM summaries and human edits survive every reindex (`reindex_file`, `save_knowledge`, watcher). Closes the regression where every file edit silently degraded semantic quality. |
| Watcher path-suppression | `agent.watcher.suppress_paths(paths, seconds=5.0)` registers paths to skip the next watcher event. `save_knowledge` registers `file.md`, `log.md`, `index.md` before its inline reindex so the watcher doesn't redundantly re-embed everything. Auto-expires so a stale entry can never permanently mute a real edit. |

## Implementation Notes

- **Embedding collection** — LanceDB table `knowledge_base_v2` (Gemini 768-dim era). Legacy v1 auto-deleted on rebuild. Input text truncated to 8000 chars before embed call. `EMBEDDING_PROVIDER=gemini` (default) or `ollama` (nomic-embed-text fallback). Switching providers requires wiping `lance-data/` — embedding spaces are incompatible.
- **Section chunking** — files split on H1-H5; each chunk has heading + content + token_count + summary. With `llm_summaries=True`, section summaries via local supergemma model (dense 2-3 sentences, 500-char limit) plus a "Document Overview" chunk per file. Parallel `ThreadPoolExecutor` (3 workers, 60s timeout). Mechanical fallback (first-line) on summary failure.
- **Folder Tree API** — `build_folder_tree()` creates folder nodes with README.md summaries; `format_folder_tree()` renders for LLM consumption. Surfaced via `/kb/folder-tree` endpoint and `get_folder_tree()` helper.
- **KB stats split** — `/kb/stats` returns file count (source files) and vector count (section chunks) separately; `table.count_rows()` returns chunk count, not file count, so `_file_count` is tracked separately.
- **Debug logging** — JSONL files in `/app/logs/`: `chat.log`, `index.log`, `tools.log`. Each entry has `ts`, `module`, `level`, `event`, `data` fields.
- **JSON session format** — `{"id", "title", "created_at", "updated_at", "turns": [{"role", "content", "timestamp"}]}`. Only `role`+`content` sent to model (timestamp stripped from LLM context).

## Session Learnings

1. **FastAPI lifespan trap** — `on_event` doesn't fire in TestClient. Use `@asynccontextmanager` + `lifespan=` param.
2. **Test infra hell escape** — Product first, tests second. Timebox test debugging to 10 min.
3. **No commits without user testing** — Automated tests pass ≠ product works. User must verify.
4. **No home dir installs** — Never install packages in ~/. Project deps stay in project.
5. **Static analysis is a hint** — call_graph.py can't do type inference. Manually validate before acting.
6. **Double message bug** — Strip current user message from history context to avoid duplication.
7. **SSE testing** — Infinite SSE streams can't be tested with TestClient. Call endpoint function directly.
8. **KB stats** — Section-based chunking means `table.count_rows()` returns chunk count, not file count. Track `_file_count` separately.
9. **_hard_split infinite loop** — When `overlap >= max_tokens`, halving causes infinite loop. Fix: `step = max(1, max_tokens - overlap)`.
10. **Graph clear on rebuild** — `_init_graph_nodes_only()` must `self.graph.clear()` before adding nodes, otherwise stale data accumulates.
11. **LanceDB query pre-embedding** — LanceDB doesn't auto-embed queries. Must call `self._embedding_fn([query])[0]` before `table.search(query_vec)`. ChromaDB's `query_texts` auto-embedded, LanceDB does not.
12. **asyncio.run in event loop** — `_generate_summary()` uses sync httpx.Client, not asyncio.run(). Works in both thread and async contexts.
13. **LLM summary model** — Per-chunk summaries are too slow (439 calls). Document-level overview is 1 call per file (~20 total). Section summaries always mechanical. SUMMARY_MODEL env var (default: supergemma4-26b).
14. **Document Overview chunks** — When `llm_summaries=True`, each file gets a "Document Overview" chunk (type=overview) with LLM summary. Embedded and searchable. Appears in heading tree and search results.
15. **Folder path mismatch** — Chunk metadata `folder` must be relative to base_dir (e.g., "ai"), NOT relative to /app (e.g., "knowledge/ai"). Folder nodes use relative-to-base-dir names. Mismatch breaks all PARENT_CHILD chunk→folder edges.
16. **Gemini API key format** — Real Gemini API keys start with `AIza...`. OAuth2 access tokens (`AQ.Ab8RN...`) are NOT API keys and will fail with 401/403. Zero-vector embeddings poison the entire table — must wipe lance-data/ after switching keys.
17. **Gemini batch limit** — `batchEmbedContents` accepts max 100 texts per call. Chunk into batches of 100, fall back to one-at-a-time on failure.
18. **Container crash on build_index** — `asyncio.wait(FIRST_COMPLETED)` kills server when runtime stub exits. Fix: await server task only. Wrap build_index in try/except so server stays up on indexing failure.
19. **google-genai vs google-generativeai** — Use `google-genai` (current SDK). `google-generativeai` is deprecated. The new SDK uses `genai.Client(api_key=...)` and `client.models.embed_content()` with `types.EmbedContentConfig`.
20. **Graph edge N+1 storm** — `_build_graph_edges()` was making ~440 API calls per reindex (one query per chunk). Fixed: fetch all embeddings in one call, compute pairwise cosine similarity with numpy. Zero extra API calls.
21. **~~ChromaDB state corruption~~** — Resolved by migrating to LanceDB. Lance format uses immutable segments with atomic writes — no corruption, no self-heal needed.
22. **Lenient tool parser** — `_KNOWN_TOOLS` whitelist + `_BARE_TOOL_PATTERN` regex catches bare `name(args)` calls. Merged with bracketed pattern, sorted by position. Defense-in-depth for models that skip `[TOOL: ...]` format.
23. **Chunk context headers** — YAML-like header prepended to chunk content BEFORE embedding: `file`, `position` (N/M), `ancestors` (heading path minus own heading), `doc_summary`. Makes parent context searchable via vector search. Preamble chunks use parent heading name. `chunk_file()` computes `filename`, `total_chunks`, `doc_summary` fields; `_build_context_header()` assembles the header; `_index_file()` prepends it.
24. **Idempotent startup** — `build_index(force=False)` on startup: checks stored `mtime` per file, skips unchanged files. `/kb/reindex` endpoint and watcher callback use `force=True` for full rebuild. `_file_mtime_unchanged()` queries LanceDB for existing chunks with matching mtime.
25. **Chat lock deadlock** — `_chat_lock` held indefinitely if client disconnects mid-stream. Fix: `await request.is_disconnected()` checks at tool loop start, inside token stream loop, and after stream ends. Lock releases via `async with` context manager on `break`.
26. **Three-branch similarity** — SIMILAR edges require same file (not just same source). Cross-file same-source pairs get INTER_FILE (threshold 0.55). Cross-source pairs get CROSS_DOMAIN (threshold 0.60). Original SIMILAR caught all same-source pairs, producing only 3 inter_file edges vs 219 after fix. **Phase D:** intra-file SIMILAR raised to 0.78, capped at 5 per node with ordinal rank metadata for provenance.
27. **Ollama concurrency** — Local LLM processes ~1-2 requests at a time. 12 ThreadPoolExecutor workers created queue backup and timeouts (20s). 3 workers + 60s timeout = zero timeouts across 440 sections. The model is the bottleneck, not the parallelism.
28. **LLM summary prompt discipline** — "Plan then summarize" produces planning trace in output. Must explicitly say "Output ONLY the summary — no planning steps." Add post-processing regex to strip any remaining "Plan:" headers and numbered planning lists.
29. **Heading slug normalization** — Graph tools need `_normalize_heading()` to match "Marcus Aurelius" → "marcus-aurelius". Simple `.lower()` comparison misses hyphenated slugs. Regex: `re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')`.
30. **Knowledge write formatting** — `save_knowledge()` auto-injects: YAML frontmatter (Obsidian block lists for `tags`/`aliases`, date-only `created`/`updated`, no legacy date keys on write), H1 heading, TOC, section dividers (`---`). Also calls `_append_log()` for mutation tracking and `_rebuild_index()` to keep graph current.
31. **Section-level search** — `search_knowledge()` uses `search()` (not `search_grouped()`) for section-level results. Agent gets `(filename, heading, summary, score)` — can use `read_knowledge_section` directly.
32. **Mechanical summaries are write-poison without a snapshot** — every `reindex_file` and `save_knowledge` call passed `llm_summaries=False`, and `_index_file` deleted prior chunks before re-creating them, so any LLM summary in the file silently flipped back to mechanical first-line extraction on the next save/edit/watcher event. Fix: snapshot `{heading: summary}` before delete; on write, mechanical only fills chunks where prior is missing OR equals what `_mechanical_summary(content)` would produce now. Structural detection — no metadata column needed.
33. **Tool-loop deadlock signature** — model emits `[TOOL: search_knowledge(query="X")]` 3+ times in one response, then again next iteration. `max_tool_iterations` and the 50% context cap don't catch it because (a) per-iteration limit only counts iterations not executions, and (b) `search_knowledge` results are small enough to never breach 50%. Fix: dedupe `(name, args)` per turn (within and across iterations) + hard `MAX_TOTAL_TOOL_EXECUTIONS=30` ceiling. Repeat-only iteration triggers `forced_summary_reason="duplicate_tool_loop"` and a nudge to use what's already in context.
34. **`save_knowledge` watcher cascade** — writing `file.md` + `log.md` + `index.md` then calling `_index_file` inline triggered THREE additional watcher-driven reindexes per save (each running embeddings, edge rebuilds). Fix: `agent.watcher.suppress_paths(paths, seconds=5.0)` registers paths to mute the next event; `save_knowledge` calls it before writing. Auto-expiry guarantees no permanent muting.
35. **Docker BuildKit export flakes** — rare `parent snapshot ... does not exist` during `docker compose up --build`. Recovery: `docker compose build --no-cache` then `docker compose up -d` (see **Ship workflow** in `CLAUDE.md`).
36. **log_event vs frame_info** — `frame_info` now includes `executed`; `log_event(..., **frame_info)` must not also pass `executed=…` or Python raises duplicate keyword errors.

## Next Steps

P0 — RESOLVED in this batch:

- ~~Heading-path resolution in `graph_neighbors`~~ — `_resolve_chunk_nodes()` handles leaf-only fallback + `file > heading` shortcut.
- ~~`graph_search` result format~~ — now emits `(filename, heading, score, summary)`.
- ~~Graph path-format reconciliation~~ — `graph_stats` returns dict-shaped most-connected entries.
- ~~`graph_neighbors` pagination~~ — `offset`, `edge_type`, `limit` params land paginated output with truncation footers.
- ~~Mechanical summaries overwriting LLM versions~~ — snapshot-and-preserve in `_index_file`; structural mechanical detection.
- ~~Tool-loop deadlock~~ — per-turn dedup (within + across iterations) + `MAX_TOTAL_TOOL_EXECUTIONS=30` hard ceiling.
- ~~`save_knowledge` watcher cascade~~ — `suppress_paths()` mutes the inline-write paths for 5s.

Tool-calling refactor + memory tier — RESOLVED:

- ~~Ollama native tool calling~~ — A1: `build_tool_registry` + per-method docstrings → Ollama `tools=` schemas; `tool_parser.py` deleted; `[TOOL: ...]` removed from L4.
- ~~Brittle save-knowledge regex~~ — A1: native tool calls handle escaped/embedded brackets/parens cleanly.
- ~~Filename / heading resolver ambiguity~~ — A2: canonical `<source>:<relpath>` everywhere via `agent/kb_paths.py`; grep-enforced.
- ~~Forced-summary path drops tool calls~~ — A3: forced-summary nudge path deleted; per-class budgets refuse in-band.
- ~~Conversations as knowledge source~~ — B2/B3: `search_conversations`, `read_conversation`; `compile_knowledge(source_type="conversation")`; `save_knowledge` validates `## Sources` turn citations.
- ~~Lint not callable as tool~~ — A1: `lint_knowledge` registered as a maintenance-class tool.
- ~~Graph provenance hidden from agent~~ — B4: `describe_node`; every graph tool renders `_format_edge_provenance`; L4 edge-type legend.
- ~~Obsidian-incompatible frontmatter / no wiki-links~~ — C1: Obsidian-valid frontmatter; markdown→`[[wiki-link]]` conversion; deterministic `## Related` block.

Phase D (post-audit tool feedback) — RESOLVED:

- ~~Graph tool ergonomics~~ — `describe_node(..., min_weight=)`; `graph_traverse(..., offset=, limit=, exclude_edge_types=)` with paginated footer; `folder_tree("canon/sub/path")` drill-down via `format_folder_tree(..., root_path=)`.
- ~~Intra-file SIMILAR noise~~ — `_build_graph_edges`: threshold 0.78, max 5 SIMILAR edges per node, `intra_rank` / `intra_total` on edges; `_format_edge_provenance` shows `rank N/M in file`.
- ~~L4 discoverability~~ — cheat sheet block in `_build_system_prompt` when tools enabled.
- ~~Lint flat clusters~~ — `lint_knowledge` reports files with many intra-file SIMILAR edges in a tight weight band.

Open backlog:

1. **UI: Expand/Collapse all toggle** — chat-header button to expand or collapse every tool and thinking bubble at once.
2. **Empty search result fallbacks** — `graph_search("")` should return graph stats + usage hint. `graph_neighbors()` with no results should suggest `list_knowledge()`.
4. **Code block CSS** — `pre code { white-space: pre-wrap; word-break: break-word; overflow-x: auto; max-width: 100%; }` to fix overflow.
5. **Orphaned chunk cleanup** — `build_index(force=False)` skips unchanged files via mtime check and replaces changed files, but never removes chunks for files deleted from the knowledge directory. Need a sweep that compares stored filenames against current files and deletes orphaned chunks.
6. **Watcher debouncing** — single user save still emits multiple `on_modified` events; `suppress_paths` covers `save_knowledge`'s self-cascade but not the duplicate-event-per-save problem in general. Add a 250ms debounce per path inside `KBEventHandler`.

Phase D follow-ups (deferred from `phase-d-tool-feedback` plan — surfaced by the agent's self-audit, intentionally not implemented in that pass):

7. **BM25 / keyword pre-filter** — proper-noun queries ("Marcus Aurelius", "Mewtwo") leak across cosine clusters because the embedder treats names as semantic tokens. Add a cheap BM25 pass over chunk text (or a `tantivy` index) and merge top-K with vector results before reranking. Karpathy's 2026 LLM-Wiki pattern explicitly pairs vector search with keyword retrieval for exactly this reason.
8. **Cross-encoder reranker** — once hybrid retrieval lands, slot a small reranker (bge-reranker-base or similar via Ollama) over the top 20 merged hits. Cuts the "all results look 0.82" problem at retrieval time instead of at provenance-display time.
9. **Per-file z-score normalisation of cosine scores** — D3 attacks the flat-similarity problem with rank+cap; the more principled fix is to z-score each file's intra-file similarity distribution so SIMILAR weights are comparable across heterogeneous and tightly-clustered files. Adds a stats pass during `_build_graph_edges` (one extra mean+std per file) — defer until D3's rank-display proves insufficient in real use.
10. **`.kbignore` glob mechanism** — insurance against future agent-development notes (or any other dev-content) polluting the index. Walk markdown files but skip anything matching a glob in `knowledge/.kbignore` / `canon/.kbignore`. Cheap, additive, and bounds the blast radius of accidental indexing.
11. **`CLASS_BUDGETS["explore"]` raise (8 → 10)** — D1-D4 are resolved, orient class offloaded 3 cheap tools, and the rewritten L4 system prompt teaches search discipline. User reports the 8-call budget is slightly too tight for meaningful multi-hop traversals. Raise by 1-2 and observe; the system prompt sufficiency rules ("after TWO explore calls, attempt an answer") should prevent waste. See roadmap for details.

## Roadmap (Critical-Path)

Items below are prioritized by user impact. Each includes current state and the delta to desired state.

### R1. KB write quality — sub-agent proofreader

**Current state:** `save_knowledge` validates frontmatter shape, conversation citations, tag formats, and (post-R12) placeholder-URL hosts. Content quality (completeness, structure, chat-context consistency) is otherwise up to the main LLM — no second check.

**Desired state (staged):**
1. **Phase 1 — pure-function content check**, no sub-agent. A `_validate_content_quality(content)` method in `tools.py` that runs synchronously before `file_path.write_text()` and refuses the write with specific feedback when the body is structurally malformed. This is the minimum viable guard and does not require the R7 sub-agent framework.
2. **Phase 2 — second-model proofreader** (depends on R7). A small/fast model (e.g. `qwen3:0.6b` local) reads the draft + the current conversation tail and returns a pass / fail / suggest-edits verdict. Runs only when Phase 1 passes.

**Phase 1 design (ready to build):**
- Body length: refuse pages with fewer than ~200 chars of actual prose after the frontmatter + H1 + TOC + dividers are stripped. Pages that consist entirely of scaffolding are a known failure mode.
- Heading hygiene: refuse pages with an H2 that has no body text before the next heading of the same or higher level. Orphan sections are a model failure signature.
- Sources discipline: if the page cites specific claims (regex: `"(study|paper|research|report|data)\s+(shows|suggests|proves|indicates)"`), require a `## Sources` section to be present. If the page is a pure synthesis that invents no facts, no requirement.
- Frontmatter sanity (belt-and-suspenders to the existing `_tags_save_validation_error`): `created` ≤ `updated`, no stray keys.
- Return a structured refusal identical in shape to `_validate_urls` so the model has one refusal protocol, not several.
- **NEW: use the R12 validator as the template.** Same error-message header (`save_knowledge: refusing to write — ...`), same bullet-per-problem body, same "fix and retry" tail.

**Phase 1 test plan:**
- Unit tests in `tests/test_content_validation.py`, same shape as `tests/test_url_validation.py`.
- Integration test: a scripted turn where the model writes a 100-char body — save must refuse, chat must continue, model must be able to retry in the same turn under the write budget.
- Negative: a legit real page (like `canon/mind/cortisol.md` format) must pass.

**Phase 2 shape (future):**
- Sub-agent contract: `{"decision": "ok" | "reject", "rationale": "...", "suggested_edits": "..."}`.
- Runs inside `compile_knowledge` before returning the plan, NOT between plan and save — by the time `save_knowledge` fires the model has committed, so proofreading there blocks user intent. Reviewing the plan is cheaper and teaches the main model via the returned critique.
- Caps: one proofread per turn (cost), rejection feedback goes back as a structured tool result (same envelope as a refusal).

**Delta from current:**
- Phase 1: add `_validate_content_quality`, add one call-site in `save_knowledge`, add test file, update CLAUDE.md's save_knowledge row.
- Phase 2: blocked on R7.

### R2. Reindex progress bar + UI discoverability

**Current state:** `POST /kb/reindex` runs synchronously and returns JSON after completion. `build_index()` emits no progress events; the UI has a single button that goes dark for ~1–3 minutes with no feedback. Other KB-panel buttons (canon list, knowledge list, search filter toggle) have no tooltips.

**Desired state:**
1. `build_index()` accepts an optional `progress_callback` invoked per file with `(phase, current, total, path)` where `phase ∈ {"chunking","embedding","graph_edges","llm_summaries","done"}`.
2. `POST /kb/reindex` becomes `SSE /kb/reindex` that streams events matching the callback payload. Legacy JSON-response variant kept as `POST /kb/reindex/sync` for scripts.
3. UI adds a progress bar bound to the SSE stream plus a label showing current file and phase.
4. Every button in the KB panel gets a `title=` attribute; icon-only buttons get an `aria-label`.

**SSE event design:**
- Event name: `reindex_progress`
- JSON payload: `{"phase": str, "current": int, "total": int, "path": str | null, "detail": str | null}`
- Terminal event: `reindex_progress` with `phase="done"`, plus `event: done` so the UI reuses the existing SSE close handling.
- Heartbeat event every 5s during long embedding phases so the UI doesn't time out.
- Cancellation: none in v1; a hung reindex requires a container restart. Adding cancel is a separate item.

**Button tooltip inventory** (current file: `ui/index.html`):
- "Reindex" — "Rebuild the knowledge base vector store and graph from current files. ~1–3 minutes."
- "Reindex + summaries" — "Full reindex plus LLM-generated per-section summaries (slower, ~5–10 minutes, requires SUMMARY_MODEL)."
- Canon/Knowledge toggle — "Switch between gold-tier (canon, read-only) and silver-tier (wiki, writable) file views."
- Search filter — "Restrict semantic search to this tier only."

**Delta from current:**
- `knowledge/index.py`: add `progress_callback` parameter, call it at the start of each file's chunking, embedding, and graph-edge phases, once per doc-level LLM summary.
- `web/app.py`: add `GET /kb/reindex/stream` (SSE), keep `POST /kb/reindex` but have it call into the same code path with a no-op callback.
- `ui/app.js`: wire SSE handler, progress bar DOM, disable the button while running.
- `ui/index.html`: `title=` attrs on all KB-panel buttons.

**Both R1 and R2 tests:** unit + integration following the same pattern as R9/R11/R12 in this increment. Minimum test_count baseline floor after either landing: current + new tests, no regression.

### R3. Autonomous deep-research tasks

**Current state:** The agent operates in a single synchronous tool loop per chat turn. It can traverse the graph and read sections, but the user must prompt each step. No background or long-running task capability.

**Desired state:** User can submit a "deep research" query (e.g., "Write me a comprehensive analysis of stoicism and neuroscience connections across the KB"). The system runs an autonomous multi-turn research loop in the background — searching, traversing, reading, compiling — and delivers a finished artifact (wiki page or structured report) when done. Progress visible in the UI.

**Delta:** Major architectural addition. Requires: (a) background task queue (asyncio Task or separate worker); (b) sub-agent with its own tool budget and context window; (c) progress streaming to UI; (d) result delivery mechanism (auto-save to wiki, or present for approval). Depends on R7 (sub-agent framework).

### R4. User task management

**Current state:** No task tracking. Conversations are the only persistent state. The agent has no concept of "open tasks" or "things we said we'd do."

**Desired state:** Users can create, list, update, and close tasks through conversation. Tasks persist across sessions. The agent can reference open tasks, mark them done, and proactively remind the user. Tasks are searchable alongside conversations and KB content.

**Delta:** New `tasks/` JSON store (similar to conversation store). New tools: `create_task`, `list_tasks`, `update_task`. Task search integrated into `search_conversations` or as a separate tool. System prompt awareness of open tasks.

### R5. Date-awareness

**Current state:** L0 system prompt includes `Now: {UTC timestamp}`. The agent has no other temporal awareness — no knowledge of file modification dates relative to "now," no concept of how old a conversation is, no ability to say "we discussed this 3 days ago."

**Desired state:** Agent can reference temporal context: file ages, conversation recency, task due dates. Search results can be filtered or boosted by recency. The agent can say "this wiki page hasn't been updated in 2 months" or "we covered this topic yesterday."

**Delta:** Surface `mtime` / `created` dates in tool outputs (already stored in LanceDB chunks). Add `days_ago` computation to `search_knowledge` and `search_conversations` results. Optional recency-boost parameter on search. System prompt guidance on temporal reasoning.

### R6. Graph node source-file querying

**Current state:** Graph tools (`describe_node`, `graph_neighbors`, `graph_traverse`) return node metadata and edge lists. The agent can see that edges exist and their types/weights, but understanding *why* a specific edge was created requires reading the underlying source section via `read_knowledge_section` — a separate tool call.

**Desired state:** Ability to inline-expand source content for any graph node directly from graph tools, or at minimum a one-call path from "I see this node" to "here's its content." Understanding of what each edge type means operationally (already documented in L4 prompt, but the agent sometimes ignores it).

**Delta:** Potentially add an `include_content=True` flag to `describe_node` that appends the section text (budget-checked). Alternatively, improve L4 prompt to make the `describe_node → read_knowledge_section` two-step more automatic. Evaluate whether the current edge-type documentation in the system prompt is sufficient or needs examples.

### R7. Sub-agent framework

**Current state:** Single-agent architecture. The main LLM handles all tool calls, all reasoning, all writing. No delegation capability. `agent/runtime.py` (127 lines) is an unused stub.

**Desired state:** Main agent can spawn sub-agents for specific tasks: (a) KB writing (with strict formatting/validation training); (b) deep research (autonomous multi-hop traversal); (c) task execution. Sub-agents share the tool registry but have their own context windows and budgets. Results return to the main chat gracefully.

**Delta:** Major architectural work. Requires: (a) sub-agent lifecycle management (spawn, monitor, collect results); (b) tool-sharing mechanism (sub-agents call existing KBTools methods); (c) context isolation (each sub-agent gets its own message history); (d) result integration (sub-agent output appears in the main chat as a structured response). `runtime.py` is the natural home. Consider: should sub-agents use the same model or a smaller/faster one?

### R8. Long-horizon stateful tool loops

**Current state:** Tool loop runs within a single `/chat` request. `max_tool_iterations=10` and per-class budgets cap the loop. Context compaction at 60% allows longer loops than before, but the fundamental constraint is one HTTP request = one tool loop.

**Desired state:** Multi-turn stateful tool loops that survive across requests. The agent can say "I'll continue researching this" and pick up where it left off in the next turn, retaining the traversal state (which nodes visited, which sections read, what's left to explore).

**Delta:** Requires persistent tool-loop state (visited nodes, partial results, remaining plan) serialized between turns. Could be stored in the conversation metadata or a separate state store. Depends on R7 (sub-agents) for the autonomous variant.

### R9. `save_knowledge` path bug — double-nesting

**Current state:** `_normalize_wiki_path` prepends `wiki/` when `parts[0]` is not in `("wiki", "memory")`. But if the model passes `knowledge/wiki/foo.md` (a canonical-ish relative path), `parts[0]` is `"knowledge"`, which triggers the prepend, producing `wiki/knowledge/wiki/foo.md`. Files are landing in `knowledge/wiki/knowledge/wiki/` on disk. The model does this because graph tools and search results return paths in `knowledge:wiki/foo.md` canonical form, and the model strips the colon but keeps the `knowledge/` prefix.

**Desired state:** `_normalize_wiki_path` strips `knowledge/` prefix before the tier check. If the model passes `knowledge/wiki/X`, `knowledge/memory/X`, or just `wiki/X`, all resolve to the same correct disk path. No double-nesting.

**Delta:** ~5-line fix in `_normalize_wiki_path`: strip leading `knowledge/` from parts before the tier check. Add test cases for `knowledge/wiki/foo.md`, `knowledge/memory/bar.md`, and bare `foo.md` inputs. Also add system prompt guidance telling the model to use flat filenames (e.g., `foo.md`) not full paths.

### R10. LLM summary query hints

**Current state:** Mechanical summaries are first ~250 chars of section content. LLM summaries (when enabled) are dense 2-3 sentence digests. Neither includes "this section would be relevant if you're asking about X, Y, Z" — the summary describes what the section *is*, not what questions it *answers*.

**Desired state:** Each section summary includes 3-5 example queries or phrases that would make this section a good retrieval hit. Stored in the summary field or a separate column. Improves recall for queries that use different vocabulary than the source text.

**Delta:** Extend the LLM summary prompt to request query hints. Parse them from the LLM output and store alongside the summary. Possibly embed the query hints as part of the chunk content for vector search, or as a separate embedding. Evaluate whether this actually improves recall vs BM25 (R-backlog-7).

### R11. Context counter shows rolling total

**Current state:** `token_usage` SSE events are emitted per-iteration. The UI displays the count from the most recent event. This appears to go up and down because each iteration's token count varies — the user sees the *last iteration's* count, not the accumulated context window usage.

**Desired state:** The context display shows the rolling total: all messages in the context window, including system prompt + history + all tool results accumulated this turn. Should monotonically increase within a turn (except when compaction fires, at which point it drops and that drop should be visually indicated).

**Delta:** Server-side: compute total tokens across `messages` list (not just the latest iteration's delta) and include in `token_usage` events. Client-side: display the total, not the per-iteration value. Possibly show compaction events as a visual indicator ("context compacted: 180K → 70K").

### R12. URL validation in saved KB pages

**Current state:** `save_knowledge` validates conversation citations (`[turn:N](#conv-id)`) against the ConversationStore. But it does not validate external URLs, and the model sometimes saves placeholder URLs like `https://example.com/study` or `https://relevant-paper.org` that look real but point nowhere.

**Desired state:** All URLs in saved content are validated: (a) `[[wiki-link]]` and relative canon paths are checked against the KB; (b) external `https://` URLs are checked for syntactic validity and optionally for liveness (HEAD request); (c) placeholder patterns (`example.com`, `placeholder`, `relevant-study.org`) are rejected with specific feedback. The model is also instructed to use the current chat thread as context when composing wiki pages, so content matches the preceding conversation.

**Delta:** URL extraction regex in `save_knowledge`. Placeholder-pattern blocklist. Optional async HEAD check for external URLs (with timeout, behind a flag since it requires network). System prompt addition: "when saving a wiki page, use the current conversation as your primary source — do not introduce URLs you haven't verified."

### R13. Raise `CLASS_BUDGETS["explore"]` (8 → 10) — DONE

**Resolved state:** `explore=10` per turn. After the orient class offloaded `list_knowledge`, `folder_tree`, and `graph_stats`, the effective explore budget is used for: `search_knowledge`, `graph_search`, `describe_node`, `graph_traverse`, `graph_neighbors`, `read_knowledge`, `read_knowledge_section`, `search_conversations`, `read_conversation`. Ten calls fits a real multi-hop research turn comfortably (search → 2× describe → 2× traverse → 2× read section → search_conversations → read_conversation → cross-check with search = 10) without forcing refusals mid-flight.

**Safeguards still in place:** The system prompt SUFFICIENCY rules ("after TWO explore calls, attempt an answer") and the playbook guidance continue to push toward early commit. `MAX_TOTAL_TOOL_EXECUTIONS = sum(CLASS_BUDGETS.values()) + 4` auto-tracks the raise so the pathological-loop backstop stays in sync. Integration + eval tests now use `explore_budget + 2` dynamically so they continue to assert the refusal contract regardless of the configured number.