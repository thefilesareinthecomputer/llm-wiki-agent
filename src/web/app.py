"""
FastAPI Web Server

Serves the UI and provides API endpoints for:
- Conversation session management (CRUD)
- SSE chat streaming with RAG
- Knowledge base browsing
- Semantic search
- File management
"""

import asyncio
import json
import logging
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models.gateway import ModelGateway
from knowledge.index import KBIndex
from memory.store import ConversationStore
from agent.tools import (
    KBTools,
    CLASS_BUDGETS,
    build_tool_registry,
    class_for_tool,
    get_budget_state,
    reset_budget,
    set_available_tokens,
    set_context_window,
)
from agent.tokenizer import count_tokens, estimate_tokens, truncate_at_sentence_boundary
from debug_log import get_logger, log_event


# --- Tool result framing config ---
# Default cap on tool result chars fed back to the model. Replaces the old
# silent 4000-char cap. Per-tool overrides live in TOOL_RESULT_OVERRIDES.
DEFAULT_TOOL_RESULT_MAX_CHARS = 8000

# Per-tool overrides. Tools that already self-report completeness (like
# read_knowledge_section, which has its own SECTION marker with COMPLETE/
# TRUNCATED status and offset continuation) pass through unmodified.
TOOL_RESULT_OVERRIDES: dict[str, dict] = {
    "read_knowledge_section": {"max_chars": None},  # None = no extra wrapping
    "read_knowledge": {"max_chars": 12000},          # heading trees can be large
    "search_knowledge": {"max_chars": 8000},
    "graph_search": {"max_chars": 8000},
    "graph_traverse": {"max_chars": 12000},
    "graph_neighbors": {"max_chars": 8000},
    "graph_stats": {"max_chars": 4000},
    "lint_knowledge": {"max_chars": 8000},
    "compile_knowledge": {"max_chars": 8000},
    "list_knowledge": {"max_chars": 8000},
    "save_knowledge": {"max_chars": 2000},
}


# Per-model context window. Used to size the adaptive tool budget.
# Defaults to 32K when an unknown model is in use.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "glm-5.1:cloud": 198000,
    "gemma4:31b-cloud": 256000,
    "qwen3.5:cloud": 256000,
    "qwen3.5:397b-cloud": 256000,
    "nemotron-3-super:cloud": 256000,
    "minimax-m2.7:cloud": 200000,
    "kimi-k2.5:cloud": 256000,
    "devstral-2:123b-cloud": 256000,
    "hf.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2:Q4_K_M": 256000,
    "gpt-oss:120b-cloud": 128000,
    "gpt-oss:20b-cloud": 128000,
    "deepseek-v3.1:671b-cloud": 128000,
}
DEFAULT_CONTEXT_WINDOW = 32000

# Compaction thresholds — trigger summarization before Ollama OOMs.
# COMPACT_TRIGGER: fraction of context window at which we compact old messages.
# COMPACT_TARGET: fraction we aim to reduce to after compaction.
COMPACT_TRIGGER = 0.60
COMPACT_TARGET = 0.35


def _context_window_for(model: str) -> int:
    return MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)


def _frame_tool_result(
    tool_name: str,
    raw: str,
    *,
    budget_remaining: dict[str, int] | None = None,
    executed: bool = True,
) -> tuple[str, dict]:
    """Wrap a raw tool result with a structured COMPLETE/TRUNCATED header
    and an optional ``[remaining_budget: ...]`` footer.

    When ``executed`` is False (budget refusal, duplicate skip, etc.), the
    header is ``NOT_EXECUTED`` so neither the model nor the UI mistakes a
    short refusal string for a successful ``COMPLETE`` tool body.

    The agent must always be able to see whether content was clipped. Silent
    truncation is the root cause of fabrication: when a section ends
    mid-sentence without warning, the model often treats the partial output
    as authoritative.

    ``budget_remaining`` is a dict ``{class_name: remaining_count}`` for the
    per-class tool budgets. When provided, a single-line footer is appended
    so the model can self-pace.
    """
    raw = raw or ""
    original_chars = len(raw)

    if not executed:
        footer = (
            ""
            if not budget_remaining
            else "\n[remaining_budget: "
            + ", ".join(f"{k}={v}" for k, v in budget_remaining.items())
            + "]"
        )
        header = f"[TOOL_RESULT: {tool_name} | NOT_EXECUTED]"
        return f"{header}\n{raw}{footer}", {
            "original_chars": original_chars,
            "delivered_chars": original_chars,
            "truncated": False,
            "executed": False,
        }

    override = TOOL_RESULT_OVERRIDES.get(tool_name, {})
    cap = override.get("max_chars", DEFAULT_TOOL_RESULT_MAX_CHARS)

    def _budget_footer() -> str:
        if not budget_remaining:
            return ""
        bits = ", ".join(f"{k}={v}" for k, v in budget_remaining.items())
        return f"\n[remaining_budget: {bits}]"

    if cap is None:
        return raw + _budget_footer(), {
            "original_chars": original_chars,
            "delivered_chars": original_chars,
            "truncated": False,
            "executed": True,
        }

    if original_chars <= cap:
        header = f"[TOOL_RESULT: {tool_name} | COMPLETE {original_chars:,} chars]"
        return f"{header}\n{raw}{_budget_footer()}", {
            "original_chars": original_chars,
            "delivered_chars": original_chars,
            "truncated": False,
            "executed": True,
        }

    body, _was_truncated = truncate_at_sentence_boundary(raw, cap)
    delivered_chars = len(body)
    header = (
        f"[TOOL_RESULT: {tool_name} | TRUNCATED at {delivered_chars:,} of "
        f"{original_chars:,} chars -- "
        f"call a more specific query, narrow the heading, or load a sub-section]"
    )
    return f"{header}\n{body}{_budget_footer()}", {
        "original_chars": original_chars,
        "delivered_chars": delivered_chars,
        "truncated": True,
        "executed": True,
    }


def _count_messages_tokens(messages: list[dict]) -> int:
    """Count tokens across a messages array. Used to drive the adaptive budget."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += count_tokens(content)
    return total + 4 * len(messages)


def _compact_messages(
    messages: list[dict],
    ctx_window: int,
    dbg=None,
) -> list[dict]:
    """Compress older tool results and assistant iteration content to fit
    within COMPACT_TARGET of the context window.

    Strategy (preserves correctness, doesn't lose information):
      1. Keep the system prompt (messages[0]) verbatim.
      2. Keep the original user message (last user msg) verbatim.
      3. Keep the most recent assistant+tool exchange (last iteration) verbatim
         so the model has full fidelity on its latest work.
      4. Everything in between (older iterations): replace each tool result
         with a 1-line stub and trim assistant content to first 200 chars.
      5. Prepend a compaction notice so the model knows context was compressed.

    Returns a new list (does not mutate the original).
    """
    target_tokens = int(ctx_window * COMPACT_TARGET)
    used = _count_messages_tokens(messages)
    if used <= int(ctx_window * COMPACT_TRIGGER):
        return messages  # no compaction needed

    if len(messages) < 4:
        return messages  # too few messages to compact

    # Identify boundaries: system (idx 0), then find the last user message
    # and the last assistant iteration block.
    system_msg = messages[0] if messages[0].get("role") == "system" else None
    start_idx = 1 if system_msg else 0

    # Find the last user message index (the original question)
    last_user_idx = None
    for i in range(len(messages) - 1, start_idx - 1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    # Find the start of the last iteration: walk backwards from end,
    # the last iteration is everything after the last assistant msg that
    # has tool_calls (or the last assistant msg if none have tool_calls).
    preserve_from = len(messages)
    for i in range(len(messages) - 1, start_idx - 1, -1):
        if messages[i].get("role") == "assistant":
            preserve_from = i
            break

    # Build compacted message list
    compacted: list[dict] = []
    if system_msg:
        compacted.append(system_msg)

    # Middle zone: compact everything between system and the preserved tail
    compaction_summaries: list[str] = []
    for i in range(start_idx, preserve_from):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "tool":
            tool_name = msg.get("tool_name", "tool")
            content = msg.get("content", "")
            # Extract just the header line if framed, otherwise truncate
            first_line = content.split("\n", 1)[0] if content else ""
            if len(first_line) > 200:
                first_line = first_line[:200] + "..."
            stub = f"[Compacted tool result: {tool_name}] {first_line}"
            compacted.append({"role": "tool", "tool_name": tool_name, "content": stub})
            compaction_summaries.append(f"  - {tool_name}: compacted from {len(content):,} chars")
        elif role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            trimmed = content[:300] + "..." if len(content) > 300 else content
            entry: dict = {"role": "assistant", "content": trimmed}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            compacted.append(entry)
        elif role == "user" and i == last_user_idx:
            compacted.append(msg)
        elif role == "user":
            compacted.append(msg)
        else:
            compacted.append(msg)

    # Add a compaction notice so the model knows what happened
    if compaction_summaries:
        notice = (
            "[Context compacted to stay within limits. "
            f"{len(compaction_summaries)} tool results were summarized to their headers. "
            "Full details from your most recent iteration are preserved below. "
            "Continue answering based on what you've gathered.]"
        )
        compacted.append({"role": "user", "content": notice})

    # Preserved tail: the last iteration (full fidelity)
    for i in range(preserve_from, len(messages)):
        compacted.append(messages[i])

    new_used = _count_messages_tokens(compacted)
    if dbg:
        log_event(
            dbg, "context_compacted",
            before_tokens=used,
            after_tokens=new_used,
            before_msgs=len(messages),
            after_msgs=len(compacted),
            compacted_tool_results=len(compaction_summaries),
            ctx_window=ctx_window,
        )

    return compacted


async def _stream_with_heartbeat(async_iter, interval: float = 3.0):
    """Wrap an async iterator so heartbeats fire during silent gaps.

    Yields tuples:
      ("heartbeat", None) on each `interval` seconds with no upstream item
      ("item", value)     for each upstream item
      ("error", exc)      if the upstream raises
    Terminates when the upstream completes.

    Used so the SSE stream keeps emitting frames while the model is
    "thinking" before its first token (typical 1-3 min gap with cloud
    models). The UI converts heartbeats into a pulsing indicator.
    """
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()

    async def producer():
        try:
            async for item in async_iter:
                await queue.put(("item", item))
        except Exception as exc:  # surfaces in main loop
            await queue.put(("error", exc))
        finally:
            await queue.put(("done", SENTINEL))

    task = asyncio.create_task(producer())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                yield ("heartbeat", None)
                continue
            if kind == "done":
                return
            yield (kind, payload)
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

# Hard backstop on total tool executions per chat turn. Per-class budgets in
# CLASS_BUDGETS (orient=5, explore=10, write=2, maintenance=3) are the
# primary control; this only catches pathological loops where some new tool
# class appears at runtime and bypasses the budget table.
MAX_TOTAL_TOOL_EXECUTIONS = sum(CLASS_BUDGETS.values()) + 4


def _tool_signature(name: str, args: dict) -> str:
    """Stable string signature for a tool call. Two calls with the same name
    and same args hash to the same signature regardless of dict ordering —
    used to detect when the model loops on identical calls within a turn."""
    try:
        normalized = sorted((str(k), str(v)) for k, v in (args or {}).items())
    except Exception:
        normalized = [(str(args),)]
    return f"{name}::{normalized}"


log = logging.getLogger(__name__)
dbg = get_logger("chat")

# Global state — set by lifespan or set_components()
model_gateway: ModelGateway | None = None
kb_index: KBIndex | None = None
memory_store: ConversationStore | None = None
kb_tools: KBTools | None = None

# Serialize chat requests so only one streams at a time
_chat_lock = asyncio.Lock()


def set_components(gateway: ModelGateway, index: KBIndex, store: ConversationStore, tools: KBTools | None = None):
    """Set pre-initialized components (called from main.py)."""
    global model_gateway, kb_index, memory_store, kb_tools
    model_gateway = gateway
    kb_index = index
    memory_store = store
    kb_tools = tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup, cleanup on shutdown."""
    # If components were already set externally (by main.py), skip init
    if model_gateway is None:
        await initialize_app()
    yield
    # Cleanup on shutdown if needed


app = FastAPI(title="LLM Wiki Agent", lifespan=lifespan)


async def initialize_app():
    """Initialize global components. Can be called manually for testing."""
    global model_gateway, kb_index, memory_store, kb_tools
    if model_gateway is None:
        model_gateway = ModelGateway()
        kb_index = KBIndex()
        memory_store = ConversationStore()
        memory_store.initialize()
        kb_tools = KBTools(kb_index=kb_index, conversation_store=memory_store)
        await model_gateway.test_connection()
        try:
            kb_index.build_index()  # llm_summaries defaults to False (mechanical only)
        except Exception as e:
            log.error(f"KB index build failed: {e}")
            log_event(dbg, "build_index_failed", error=str(e))
        log.info("Components initialized")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/ui", StaticFiles(directory="ui"), name="ui")


@app.get("/", response_class=HTMLResponse)
async def root():
    return Path("ui/index.html").read_text()


# --- System Prompt (Modular Layers) ---

def _build_system_prompt(kb_context: str = "", tools_enabled: bool = False) -> str:
    """Build modular system prompt.

    L0: Identity + voice (always present)
    L1: Mind-en-Place framework + operating rules (always present)
    L2: RAG context (dynamic, injected when KB hits exist)
    L3: Session history (injected by chat() from get_recent())
    L4: Tool literacy + patterns (when KB tools are active)
    L5: Mempalace recall (future)
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ---------------------------------------------------------------
    # L0: IDENTITY — who you are, how you sound, what you refuse
    # ---------------------------------------------------------------
    parts = [
        "# IDENTITY\n"
        "You are LLM Wiki Agent — a second-brain operator for a high-agency technical\n"
        "user. You synthesize their archive, coach their development, and act\n"
        "as an SRE for their cognition. You are not a chatbot. You are a\n"
        "domain-literate engineer with read/write access to a knowledge\n"
        "graph, working alongside them at their own tier.\n"
        "\n"
        f"Now: {now}\n"
        "\n"
        "# VOICE\n"
        "You sound like Q, Alfred, Jarvis: candid, precise, unbothered. No\n"
        "hedging. No pep. No mirroring. No emoji. No em-dashes — only plain\n"
        "dashes ( - ). You don't restate the user's question. You don't ask\n"
        "rhetorical setup questions. You don't apologize for the work. When\n"
        "you don't know, you say 'I don't have that' and stop. When the\n"
        "user's premise is wrong, you say so and fix it — that's the job,\n"
        "not a violation of decorum.\n"
        "\n"
        "The user is busy and sharp. Compress. Signal over noise. They will\n"
        "ask for more detail if they want it; don't front-load hedge clauses."
    ]

    # ---------------------------------------------------------------
    # L1: DOMAIN — the Mind-en-Place framework. This is canon-verbatim
    # because downstream KB content references these names + layers.
    # ---------------------------------------------------------------
    parts.append(
        "# DOMAIN: MIND-EN-PLACE (MEP)\n"
        "MEP is behavioral self-engineering — mise-en-place for cognition.\n"
        "The goal is smooth execution by offloading complexity to prep,\n"
        "freeing bandwidth at runtime. This KB is the prep layer.\n"
        "\n"
        "Three themes in the vault:\n"
        "  COMMITMENTS — routines, intentions, boundaries, plans, todos\n"
        "  EXPERIENCE  — tracked habits, resume, career arc, reflections\n"
        "  WISDOM      — philosophy, quotes, psychology, neuroscience,\n"
        "                research, strategy\n"
        "\n"
        "Daily cycle (the ETL pipeline you support):\n"
        "  1. Pull a quote of the day.\n"
        "  2. Fan out: gather related WISDOM from KB.\n"
        "  3. Map WISDOM against current situation (logs / EXPERIENCE).\n"
        "  4. Produce insights, cite canon, mark tensions honestly.\n"
        "  5. User reflects + journals; the updated KB feeds tomorrow.\n"
        "\n"
        "File-family conventions:\n"
        "  mep-*.md / _*.md  — framework microservices (routines, trackers)\n"
        "  user-*.md / my-*.md — logs, metrics, reflections (dim/fact tables)\n"
        "  vault-*.md        — research, study data, best practices (OS/libs)\n"
        "Real life is the UI. The archive only matters insofar as it moves\n"
        "action in the world.\n"
        "\n"
        "Core operating principles (use these as tiebreakers):\n"
        "  - A place for everything, everything in place.\n"
        "  - OODA: move forward on available information.\n"
        "  - Minimum effective dose. Avoid fake optimization.\n"
        "  - 80% consistency beats perfection.\n"
        "  - Action precedes meaning; identity follows action.\n"
        "  - Be here now — there is no time but the present."
    )

    # ---------------------------------------------------------------
    # L1b: OPERATING RULES — the invariants the tool loop relies on
    # ---------------------------------------------------------------
    parts.append(
        "# OPERATING RULES (non-negotiable)\n"
        "1. TRUTH FIRST. State reality only. If uncertain, say so and name\n"
        "   what would resolve it. Quantify mixed evidence (weak / moderate\n"
        "   / strong) and cite.\n"
        "2. NO FABRICATION. If a tool returns 'not found' or empty, say that.\n"
        "   Do not invent a filename, heading, quote, or citation — ever.\n"
        "   Headings are structure, not content. A heading tree is not a\n"
        "   section body.\n"
        "3. ONE RETRY ON NEAR-MISS. If a tool returned a 'did you mean'\n"
        "   suggestion, you may retry exactly once using one of its\n"
        "   suggestions. No second guess. No third try.\n"
        "4. NEVER BLUR SOURCES. Do not merge content from different files\n"
        "   into one paragraph. Attribute every claim to its file:heading.\n"
        "5. PREFER CANON. Canon > wiki > memory > raw when two tiers agree.\n"
        "   Cite canon as ground truth; cite wiki/memory as your own prior\n"
        "   synthesis; cite raw as evidence to be compiled, not as authority.\n"
        "6. REUSE WHAT'S ALREADY IN CONTEXT. If the answer is in this\n"
        "   turn's KB hits, conversation history, or a prior tool result,\n"
        "   compose from that. Do not re-search to manufacture a second\n"
        "   source for content you already have.\n"
        "7. STOP WHEN DONE. When the user's question is answered, stop.\n"
        "   Do not offer follow-ups, summaries, or 'let me know if…'."
    )

    # ---------------------------------------------------------------
    # L4: TOOL LITERACY — when KB tools are attached. Schemas are sent
    # on Ollama's native `tools=` channel; this block teaches HOW and
    # WHEN to use them, not the JSON format.
    # ---------------------------------------------------------------
    if tools_enabled:
        orient_n = CLASS_BUDGETS.get("orient", 5)
        explore_n = CLASS_BUDGETS.get("explore", 8)
        write_n = CLASS_BUDGETS.get("write", 2)
        maint_n = CLASS_BUDGETS.get("maintenance", 3)
        parts.append(
            "# KB TOOLS — how to actually use them\n"
            "You have a second brain with 500+ section-level nodes in a\n"
            "knowledge graph. Tool calls are native: emit structured\n"
            "tool_calls (the runtime handles the protocol). Never paste\n"
            "[TOOL: ...] bracket text — it is inert and reads as fabrication.\n"
            "\n"
            "## MENTAL MODEL\n"
            "The KB is a directed graph of markdown sections, not a pile of\n"
            "files. Every tool returns either structure (headings, edges) or\n"
            "content (section text). Know which you're getting. A heading\n"
            "tree is a **table of contents**; you must call\n"
            "`read_knowledge_section` to actually read a section body.\n"
            "\n"
            "## TIERS (medallion)\n"
            "  [canon]  gold    — curated, READ-ONLY (`canon/`)\n"
            "  [wiki]   silver  — your compiled pages, WRITABLE (`knowledge/wiki/`)\n"
            "  [memory] memory  — distilled notes, WRITABLE (`knowledge/memory/`)\n"
            "  [raw]    bronze  — source material, READ-ONLY (`knowledge/raw/`)\n"
            "Search is tier-weighted (canon > wiki > memory > raw at equal\n"
            "cosine). Cite canon as authority. Cite raw only as evidence to\n"
            "synthesize. `save_knowledge` lands in `wiki/` unless you prefix\n"
            "with `memory/`.\n"
            "\n"
            "## FILE ADDRESSING\n"
            "Every filename uses `<source>:<relpath>` canonical form:\n"
            "  canon:mind-en-place/vault-philosophy-quotes.md\n"
            "  knowledge:wiki/cortisol.md\n"
            "  knowledge:raw/technology/Data Warehouse Toolkit.md\n"
            "Bare paths resolve but canonical is unambiguous — pass tool\n"
            "output back in verbatim.\n"
            "\n"
            f"## PER-TURN BUDGETS (independent; write never starves)\n"
            f"  orient      : {orient_n} (folder_tree, list_knowledge, graph_stats)\n"
            f"  explore     : {explore_n} (search / graph / read / describe / conversations)\n"
            f"  write       : {write_n} (compile_knowledge, save_knowledge)\n"
            f"  maintenance : {maint_n} (lint_knowledge)\n"
            "Every result prints `[remaining_budget: ...]`. Watch it. When a\n"
            "class is exhausted the runtime refuses further calls in that\n"
            "class with an explanatory string — do NOT retry, answer with\n"
            "what you have.\n"
            "\n"
            "## PLAYBOOK — pick the right tool, not the loudest one\n"
            "\n"
            "GOAL → FIRST MOVE (the 80% case):\n"
            "  \"Answer a specific question\"\n"
            "      → search_knowledge(query) gets a ranked list of sections\n"
            "        with file:heading addresses. Then read_knowledge_section\n"
            "        on the top 1-2 hits. Done in 3 calls, not 8.\n"
            "  \"Find ALL nodes about a topic\"\n"
            "      → graph_search(query) — it already does vector + graph\n"
            "        expansion in ONE call. Do NOT fire four variants of the\n"
            "        same query; that's not parallelism, it's noise. If the\n"
            "        first graph_search is thin, describe_node on the top hit\n"
            "        to walk its neighborhood — do not re-search with synonyms.\n"
            "  \"Understand how X connects to Y\"\n"
            "      → describe_node(filename, heading) — one call returns\n"
            "        tier, summary, and every incoming + outgoing edge with\n"
            "        provenance. Use this INSTEAD OF stitching graph_neighbors\n"
            "        + graph_traverse + read_knowledge.\n"
            "  \"What's in this file?\"\n"
            "      → read_knowledge(filename) for the heading tree + token\n"
            "        costs, then read_knowledge_section for the sections you\n"
            "        actually need. Do not read_knowledge_section blind — you\n"
            "        will load the wrong part.\n"
            "  \"Where am I / what do I have?\"\n"
            "      → folder_tree() once, or graph_stats() for hub density.\n"
            "        These are cheap (orient budget). Don't repeat them.\n"
            "  \"Did we already discuss this?\"\n"
            "      → search_conversations(query) before any KB search. Past\n"
            "        turns are memory; reuse them.\n"
            "\n"
            "## SEARCH DISCIPLINE (this is where agents fail)\n"
            "- ONE QUERY, NOT FIVE. A second graph_search with synonyms\n"
            "  returns the same nodes with slightly different scores. Your\n"
            "  budget is for exploration depth, not keyword permutations.\n"
            "- If a search returns nothing useful, your query is too specific.\n"
            "  BROADEN. Do not narrow further — vector search rewards broad\n"
            "  canonical phrasing (\"stoicism and control\", not \"epictetus\n"
            "  dichotomy of control practical application\").\n"
            "- `heading=` is NOT a query. Heading is a structural address\n"
            "  (a literal H1/H2/H3 from a tree). If you want semantic\n"
            "  reranking of ambiguous headings, pass `query=` separately —\n"
            "  graph_neighbors / graph_traverse / describe_node all accept it.\n"
            "\n"
            "## PARAMETER CHEAT SHEET — these args exist, USE them\n"
            "  folder_tree(folder='')\n"
            "    folder='canon/mind-en-place' drills in when the root view\n"
            "    collapses 100+ files into a single summary line.\n"
            "  graph_neighbors(filename, heading='', edge_type='',\n"
            "                  min_weight=0.0, offset=0, limit=50, query='')\n"
            "    edge_type='references'|'inter_file'|'cross_domain'|'similar'\n"
            "      isolates one bond class; stacks with min_weight=0.83 to\n"
            "      drop noisy tails. Use `offset` from the result footer to\n"
            "      page through — never guess offsets.\n"
            "  graph_traverse(filename, heading='', depth=2,\n"
            "                 min_weight=0.0, offset=0, limit=100,\n"
            "                 exclude_edge_types='', query='')\n"
            "    exclude_edge_types='similar' drops intra-file noise so you\n"
            "    see structural cross-file edges. Comma-separated. Note:\n"
            "    parent_child is ALWAYS excluded — don't waste a slot on it.\n"
            "  describe_node(filename, heading='', min_weight=0.0, query='')\n"
            "    Raise min_weight to surface strong edges the per-type cap\n"
            "    was hiding under noise. `query=` disambiguates when the\n"
            "    heading matches multiple chunks.\n"
            "  read_knowledge_section(filename, section, offset=0)\n"
            "    15 section-loads per turn; ~8000 tokens must remain when\n"
            "    you call it. Use `read_knowledge` first to see token costs.\n"
            "\n"
            "## EDGE TYPES — trust hierarchy\n"
            "  references    — explicit author intent ([[wiki-link]] or\n"
            "                  markdown link). Highest-trust signal. Edge\n"
            "                  attributes carry link_text + link_kind\n"
            "                  ('wiki' / 'markdown' / 'prose'). When you see\n"
            "                  ` - via 'X' [wiki]`, that's the exact anchor\n"
            "                  text where the source author made the jump.\n"
            "  inter_file    — embedding cosine, different files, same\n"
            "                  topic. Cross-file semantic glue.\n"
            "  cross_domain  — embedding cosine across far-apart domains.\n"
            "                  Rare and high-value — unexpected bridges.\n"
            "  similar       — embedding cosine, SAME file (intra-doc).\n"
            "                  Carries `rank 1/5 in file` in provenance —\n"
            "                  trust the rank over raw weight when the file\n"
            "                  is tightly clustered (quote vaults, glossaries).\n"
            "                  Capped at 5 per node by construction.\n"
            "  relates_to    — entity / concept overlap. Coarsest; good for\n"
            "                  jumping topic boundaries, weak as evidence.\n"
            "  parent_child  — H1→H2→H3 hierarchy. STRUCTURAL, not semantic.\n"
            "                  NEVER cite this as evidence — it's folder layout.\n"
            "When citing graph evidence in prose: references > inter_file /\n"
            "cross_domain > similar > relates_to.\n"
            "\n"
            "## WRITING (wiki authoring loop)\n"
            "  compile_knowledge(source, query='', source_type='file',\n"
            "                    source_ref='')\n"
            "    Plans a wiki page. With source_type='file', `source` is a KB\n"
            "    path. With source_type='conversation', `source_ref` selects\n"
            "    turns: '<conv_id>', '<conv_id>:last:N', or\n"
            "    '<conv_id>:turn:A-B'. Does not write disk but counts as one\n"
            "    write-budget call.\n"
            "  save_knowledge(filename, content, tags='')\n"
            "    Writes to wiki/ (or memory/ if prefixed). Refuses canon/ and\n"
            "    raw/. Every `[turn:N](#conv-id)` citation under `## Sources`\n"
            "    is verified against the ConversationStore — an unverifiable\n"
            "    citation aborts the write. Do not fake turn numbers.\n"
            "    **Pass flat filenames** like `cortisol.md`, not the full\n"
            "    canonical path — `knowledge:wiki/cortisol.md` is what tool\n"
            "    output PRINTS, not what you pass BACK. Bare names auto-land\n"
            "    under wiki/. `memory/note.md` lands under memory/. Do not\n"
            "    pass `knowledge/wiki/...` — a leading `knowledge/` is\n"
            "    defensively stripped but still reads as sloppy.\n"
            "\n"
            "When to promote to wiki: if you have re-derived the same\n"
            "synthesis twice in a session, or loaded the same raw/ source\n"
            "twice, compile it. Future sessions inherit the page instead of\n"
            "re-deriving it. ONE CONCEPT PER PAGE — do not dump a whole\n"
            "transcript into one omnibus file.\n"
            "\n"
            "## WRITING DISCIPLINE — no invented URLs\n"
            "- NEVER fabricate `https://` URLs. Placeholder hosts like\n"
            "  `example.com`, `placeholder.*`, `relevant-study.org`,\n"
            "  `your-domain.com`, `test.com` are hard-refused by\n"
            "  save_knowledge and your write will fail loudly.\n"
            "- If you have not personally seen a URL in canon, raw, a\n"
            "  tool result, or the user's message, DO NOT write one.\n"
            "  Use `[[wiki-link]]` to another KB page, a relative\n"
            "  `../../canon/...` path, or a conversation citation under\n"
            "  ## Sources instead.\n"
            "- When composing a wiki page, your PRIMARY source context\n"
            "  is the current conversation thread plus whatever KB\n"
            "  sections you've actually loaded. Do not invent external\n"
            "  citations to look authoritative — an unverified URL is\n"
            "  strictly worse than no URL.\n"
            "\n"
            "## SUFFICIENCY — answer-first, tool-second\n"
            "- After TWO explore calls you MUST attempt an answer with what\n"
            "  you have. Do not chain tools to look thorough.\n"
            "- A tool result coming back is NOT a new prompt. Integrate it,\n"
            "  then answer OR call one more tool only if genuinely essential.\n"
            "- If the content is already in context (KB hits, prior turns,\n"
            "  prior tool result this turn), synthesize from that. Do not\n"
            "  re-fetch.\n"
            "- When done, stop. No 'let me know if…', no summaries the user\n"
            "  didn't ask for, no offer of follow-up work.\n"
            "\n"
            "## ERROR BEHAVIOR\n"
            "- Tool returns 'not found' → report it. Do not search adjacent.\n"
            "- Tool returns 'did you mean X, Y, Z' → ONE retry with one of\n"
            "  the listed suggestions. If that fails, stop and report.\n"
            "- Budget exhausted refusal → not an error; answer with what you\n"
            "  have. Don't retry under a different tool name to sneak past\n"
            "  the budget — the runtime classifies unknown tools as explore."
        )

    # L2: RAG context (only when relevant results found)
    if kb_context:
        parts.append(kb_context)

    return "\n\n".join(parts)


def _execute_tool(name: str, args: dict) -> str:
    """Execute a tool call by name with a kwargs-style dispatch.

    With native Ollama tool calling the model emits real keyword arguments
    matching each Python function's signature, so dispatch collapses to a
    single ``registry[name](**args)`` call. Positional-key fallbacks are kept
    only as a defensive layer for older test fixtures that still pass
    ``{"0": ..., "1": ...}`` dicts.
    """
    import time

    log_event(dbg, "execute_tool", tool=name, args=args)
    if not kb_tools:
        return "Tool error: KB tools not available."

    registry = build_tool_registry(kb_tools)
    fn = registry.get(name)
    if fn is None:
        return f"Unknown tool: {name}"

    args = dict(args or {})

    # Defensive coercion for legacy test fixtures that pass positional keys
    # ("0", "1", ...) instead of named kwargs. Native tool calls already
    # arrive as proper kwargs so this is a no-op for production traffic.
    import inspect

    if any(k.isdigit() for k in args.keys() if isinstance(k, str)):
        try:
            sig = inspect.signature(fn)
            params = [p for p in sig.parameters.values()
                      if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                    inspect.Parameter.KEYWORD_ONLY)]
            for idx, param in enumerate(params):
                key = str(idx)
                if key in args and param.name not in args:
                    args[param.name] = args.pop(key)
        except (TypeError, ValueError):
            pass
        for k in [k for k in args.keys() if isinstance(k, str) and k.isdigit()]:
            args.pop(k, None)

    start = time.time()
    try:
        result = fn(**args)
        elapsed = time.time() - start
        log_event(dbg, "tool_executed", tool=name,
                  result_len=len(result) if isinstance(result, str) else 0,
                  elapsed_ms=int(elapsed * 1000))
        return result if isinstance(result, str) else str(result)
    except TypeError as e:
        elapsed = time.time() - start
        log_event(dbg, "tool_error", tool=name, error=str(e),
                  elapsed_ms=int(elapsed * 1000))
        return f"Tool error ({name}): bad arguments — {e}"
    except Exception as e:
        elapsed = time.time() - start
        log_event(dbg, "tool_error", tool=name, error=str(e),
                  elapsed_ms=int(elapsed * 1000))
        return f"Tool error ({name}): {e}"


# --- Conversation Endpoints ---

@app.get("/conversations")
async def list_conversations(
    response: Response,
    limit: int | None = None,
    offset: int = 0,
):
    """List conversations, most recent message first.

    Returns a JSON list (backward compatible with existing callers). When
    ``limit`` is provided, the response is truncated to that many entries
    starting at ``offset``; the full count is exposed via the
    ``X-Total-Count`` response header so the sidebar can decide whether to
    show a "show more" button.
    """
    if not memory_store:
        raise HTTPException(503, "Memory not initialized")
    all_convs = memory_store.list_conversations()
    response.headers["X-Total-Count"] = str(len(all_convs))
    if offset < 0:
        offset = 0
    if limit is None or limit <= 0:
        return all_convs[offset:]
    return all_convs[offset:offset + limit]


@app.post("/conversations")
async def create_conversation():
    """Create a new conversation."""
    if not memory_store:
        raise HTTPException(503, "Memory not initialized")
    conv_id = memory_store.create_conversation()
    return {"id": conv_id}


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get all turns in a conversation."""
    if not memory_store:
        raise HTTPException(503, "Memory not initialized")
    turns = memory_store.get_conversation(conversation_id)
    return {"id": conversation_id, "turns": turns}


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if not memory_store:
        raise HTTPException(503, "Memory not initialized")
    memory_store.delete_conversation(conversation_id)
    return {"status": "ok"}


# --- Chat Endpoint ---

@app.get("/sse")
async def sse_endpoint(request: Request, token: str):
    """SSE connection for chat streaming."""
    if not token:
        raise HTTPException(401, "Token required")

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            if await request.is_disconnected():
                break
            await asyncio.sleep(30)
            yield f"event: ping\ndata: {{}}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint with RAG retrieval and streaming SSE response."""
    body = await request.json()
    message = body.get("message", "")
    conversation_id = body.get("conversation_id")

    if not conversation_id:
        raise HTTPException(400, "conversation_id required")

    if not model_gateway:
        raise HTTPException(503, "Model not initialized")

    # Store user message in memory
    if memory_store:
        memory_store.add_turn("user", message, conversation_id=conversation_id)

    # RAG: Search KB for relevant context (L2)
    kb_results = []
    kb_context = ""
    if kb_index:
        kb_results = kb_index.search(message, top_k=3)
        if kb_results:
            kb_context = "Knowledge base results:\n"
            for i, r in enumerate(kb_results, 1):
                score = r.get("score", 0)
                content = r.get("content") or ""
                heading = r.get("heading", "")
                total = r.get("section_count", 0) or "?"
                pos = r.get("chunk_index", 0) + 1
                path_label = r.get("path", "")
                if heading:
                    path_label = f"{path_label} > {heading}"
                kb_context += f"\n[Source {i}: {path_label} (chunk {pos}/{total}, relevance: {score:.2f})]\n{content[:500]}\n"

    # Get recent conversation context (L3) under a token budget.
    # Replaces the old fixed n=10 walk -- prevents context bloat on long
    # conversations, and rewrites old tool-heavy turns as compact stubs so
    # the model doesn't re-replay the same tool calls.
    context = []
    if memory_store:
        # Budget: 25% of model context window for history, scaled to current model.
        try:
            current_model_for_history = model_gateway.get_current_model() if model_gateway else "default"
        except Exception:
            current_model_for_history = "default"
        history_budget = int(_context_window_for(current_model_for_history) * 0.15)
        context = memory_store.get_history_within_budget(
            conversation_id=conversation_id,
            max_tokens=history_budget,
            always_full_n=1,
        )
        # The current user message is already the last turn in context,
        # so remove it to avoid sending it twice to the LLM
        if context and context[-1]["role"] == "user" and context[-1]["content"] == message:
            context = context[:-1]

    # Build system prompt (L0 + L1 + L2 + L4 if tools available)
    tools_enabled = kb_tools is not None
    system_prompt = _build_system_prompt(kb_context=kb_context, tools_enabled=tools_enabled)

    # Build messages for LLM. With native tool calling there are no
    # ``[TOOL: ...]`` brackets in stored assistant turns to strip; persisted
    # assistant content is already free of inline tool syntax. Tool results
    # are NOT re-replayed from history — the per-turn metadata stub on each
    # assistant turn cites them so the model knows what ran.
    messages = [{"role": "system", "content": system_prompt}]
    for turn in context:
        content = turn.get("content", "")
        messages.append({"role": turn["role"], "content": content})
    messages.append({"role": "user", "content": message})

    # Stream response via SSE (one at a time)
    async def event_generator():
        async with _chat_lock:
            # Re-check gateway after acquiring lock
            if not model_gateway:
                yield f"event: error\ndata: Model not initialized\n\n"
                return

            # Reset tool budget for this request and seed it with a real
            # token-tracked starting point. set_available_tokens replaces the
            # old theatrical 999999; set_context_window enables a 50%-of-ctx
            # tool budget cap.
            reset_budget()
            current_model = model_gateway.get_current_model()
            ctx_window_initial = _context_window_for(current_model)
            set_context_window(ctx_window_initial)
            initial_used = _count_messages_tokens(messages)
            set_available_tokens(max(0, ctx_window_initial - initial_used))

            log_event(
                dbg,
                "chat_start",
                message=message[:200],
                conversation_id=conversation_id,
                kb_results=len(kb_results) if kb_results else 0,
                tools_enabled=tools_enabled,
                model=current_model,
                context_window=ctx_window_initial,
                initial_used_tokens=initial_used,
            )

            # Send KB context first
            if kb_results:
                yield f"event: kb_context\ndata: {len(kb_results)} files found\n\n"
                for r in kb_results:
                    yield f"event: kb_file\ndata: {{\"path\": \"{r['path']}\", \"score\": {r['score']:.3f}}}\n\n"

            # ---------------------------------------------------------
            # Native Ollama tool-calling loop (A1 + A3)
            # ---------------------------------------------------------
            # Per-class budgets replace the old global MAX_TOTAL_TOOL_EXECUTIONS
            # so write tools never get starved by long exploration chains.
            full_response = ""
            tool_iterations = 0
            persisted_tool_calls: list[dict] = []
            persisted_tool_results: list[dict] = []
            forced_summary_reason: str | None = None

            executed_signatures: set[str] = set()
            total_tool_executions = 0

            # Per-class spent counters and remaining-budget snapshot.
            class_used: dict[str, int] = {k: 0 for k in CLASS_BUDGETS}

            def _budget_remaining() -> dict[str, int]:
                return {k: max(0, CLASS_BUDGETS[k] - class_used.get(k, 0))
                        for k in CLASS_BUDGETS}

            # Build the tool registry once per request. The Ollama Python SDK
            # introspects each callable's signature + Google-style docstring
            # to auto-generate the JSON schema sent to the model.
            tool_registry = build_tool_registry(kb_tools) if kb_tools else {}
            tool_callables = list(tool_registry.values()) if tool_registry else None
            model_supports_native_tools = (
                tool_callables is not None
                and model_gateway.supports_tools()
            )
            if tool_callables and not model_supports_native_tools:
                log_event(
                    dbg,
                    "tools_unsupported_by_model",
                    model=model_gateway.get_current_model(),
                )

            # Outer loop. One pass = one model stream. Exits when:
            #  - model returns no tool_calls (it's done answering), OR
            #  - all budget classes are exhausted, OR
            #  - context exceeds 85% even after compaction (hard brake), OR
            #  - client disconnects.
            while True:
                if tool_iterations > 0:
                    used_now = _count_messages_tokens(messages)
                    # Auto-compact when approaching context limits
                    if used_now > ctx_window_initial * COMPACT_TRIGGER:
                        compacted = _compact_messages(messages, ctx_window_initial, dbg)
                        messages.clear()
                        messages.extend(compacted)
                        used_now = _count_messages_tokens(messages)
                        set_available_tokens(max(0, ctx_window_initial - used_now))
                        yield (
                            f"event: heartbeat\n"
                            f"data: {json.dumps({'phase': 'compaction', 'iteration': tool_iterations, 'label': 'Compacting context...'})}\n\n"
                        )
                    # Hard brake: if still over 85% after compaction, stop
                    if used_now > ctx_window_initial * 0.85:
                        log_event(
                            dbg,
                            "intra_turn_cap_reached",
                            used_tokens=used_now,
                            context_window=ctx_window_initial,
                            tool_iterations=tool_iterations,
                        )
                        forced_summary_reason = "context_85pct_post_compaction"
                        break
                if await request.is_disconnected():
                    log_event(dbg, "client_disconnected", phase="tool_loop_start")
                    break

                yield (
                    f"event: iteration_start\n"
                    f"data: {json.dumps({'iteration': tool_iterations})}\n\n"
                )

                iteration_content = ""
                iteration_thinking = ""
                iteration_tool_calls: list[dict] = []

                log_event(dbg, "stream_start",
                          iteration=tool_iterations, msg_count=len(messages))

                # Choose whether to send tools= on this stream. When every
                # budget class is exhausted we drop tools entirely so the
                # model is forced to answer in plain prose instead of
                # emitting calls the runtime would just refuse.
                budget_left = _budget_remaining()
                any_budget = any(v > 0 for v in budget_left.values())
                tools_for_call = (
                    tool_callables
                    if (model_supports_native_tools and any_budget)
                    else None
                )

                try:
                    async for kind, payload in _stream_with_heartbeat(
                        model_gateway.chat_stream(messages, tools=tools_for_call),
                        interval=3.0,
                    ):
                        if await request.is_disconnected():
                            log_event(dbg, "client_disconnected", phase="stream")
                            break

                        if kind == "heartbeat":
                            yield (
                                f"event: heartbeat\n"
                                f"data: {json.dumps({'phase': 'model', 'iteration': tool_iterations, 'label': 'Thinking...'})}\n\n"
                            )
                            continue
                        if kind == "error":
                            raise payload

                        token_type, token = payload
                        if token_type == "thinking":
                            iteration_thinking += token
                            encoded = urllib.parse.quote(token)
                            yield f"event: thinking\ndata: {encoded}\n\n"
                        elif token_type == "content":
                            iteration_content += token
                            encoded = urllib.parse.quote(token)
                            yield f"event: token\ndata: {encoded}\n\n"
                        elif token_type == "tool_call":
                            iteration_tool_calls.append(token)
                except Exception as e:
                    log.error(f"Stream error: {e}")
                    log_event(dbg, "stream_error", error=str(e),
                              iteration=tool_iterations)
                    err_str = str(e).lower()
                    is_model_overload = (
                        "500" in err_str
                        or "internal server error" in err_str
                        or "context" in err_str
                    )
                    if is_model_overload and tool_iterations > 0 and full_response.strip():
                        log_event(dbg, "model_overload_recovery",
                                  iteration=tool_iterations,
                                  full_response_len=len(full_response))
                        yield (
                            f"event: token\ndata: "
                            f"{urllib.parse.quote(chr(10) + chr(10) + '---' + chr(10) + '*[Context limit reached after ' + str(tool_iterations) + ' tool iterations — response finalized with results gathered so far.]*')}\n\n"
                        )
                        forced_summary_reason = "model_overload"
                        break
                    yield f"event: error\ndata: {str(e)}\n\n"
                    break

                log_event(dbg, "stream_end", iteration=tool_iterations,
                          content_len=len(iteration_content),
                          thinking_len=len(iteration_thinking),
                          tool_calls_found=len(iteration_tool_calls),
                          has_content=bool(iteration_content.strip()))

                if await request.is_disconnected():
                    log_event(dbg, "client_disconnected", phase="after_stream")
                    break

                # Append the assistant's message (preserving tool_calls so the
                # SDK protocol round-trips correctly).
                assistant_msg: dict = {
                    "role": "assistant",
                    "content": iteration_content,
                }
                if iteration_tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            }
                        }
                        for tc in iteration_tool_calls
                    ]
                messages.append(assistant_msg)
                full_response += iteration_content

                if not iteration_tool_calls:
                    log_event(dbg, "chat_done",
                              reason="no_tool_calls",
                              full_response_len=len(full_response))
                    break

                tool_iterations += 1
                log_event(dbg, "tool_execution_start",
                          iteration=tool_iterations,
                          tool_count=len(iteration_tool_calls),
                          content_len=len(iteration_content))

                # Execute each tool call. Per-class budgets are enforced
                # before invocation; refused calls get a structured
                # explanation back so the model can react instead of stalling.
                for tc in iteration_tool_calls:
                    if total_tool_executions >= MAX_TOTAL_TOOL_EXECUTIONS:
                        log_event(dbg, "max_total_tool_executions_reached",
                                  limit=MAX_TOTAL_TOOL_EXECUTIONS,
                                  iteration=tool_iterations)
                        forced_summary_reason = "max_total_tool_executions"
                        break

                    tool_name = tc["name"]
                    tool_args = tc.get("arguments", {}) or {}
                    sig = _tool_signature(tool_name, tool_args)
                    is_duplicate = sig in executed_signatures
                    cls = class_for_tool(tool_name)
                    used_in_class = class_used.get(cls, 0)
                    cls_budget = CLASS_BUDGETS.get(cls, 0)
                    over_class_budget = used_in_class >= cls_budget

                    yield (
                        f"event: tool_call\n"
                        f"data: {json.dumps({'tool': tool_name, 'args': tool_args})}\n\n"
                    )
                    started_ms = time.monotonic()
                    yield (
                        f"event: tool_executing\n"
                        f"data: {json.dumps({'tool': tool_name, 'args': tool_args, 'started': time.time()})}\n\n"
                    )

                    if over_class_budget:
                        raw_result = (
                            f"REFUSED: {cls} budget exhausted "
                            f"(used {used_in_class}/{cls_budget} this turn). "
                            f"Answer with what you have or call a tool from a "
                            f"different class. Remaining: {_budget_remaining()}."
                        )
                        executed = False
                    elif is_duplicate:
                        raw_result = (
                            f"SKIPPED REPEAT CALL: {tool_name}({tool_args}). "
                            f"You already ran this earlier this turn -- its "
                            f"result is in your context above. Use it, or call "
                            f"a different tool."
                        )
                        executed = False
                    else:
                        raw_result = await asyncio.to_thread(
                            _execute_tool, tool_name, tool_args
                        )
                        class_used[cls] = used_in_class + 1
                        total_tool_executions += 1
                        executed_signatures.add(sig)
                        executed = True

                    elapsed_ms = int((time.monotonic() - started_ms) * 1000)
                    yield (
                        f"event: tool_done\n"
                        f"data: {json.dumps({'tool': tool_name, 'elapsed_ms': elapsed_ms, 'executed': executed})}\n\n"
                    )

                    framed_result, frame_info = _frame_tool_result(
                        tool_name,
                        raw_result,
                        budget_remaining=_budget_remaining(),
                        executed=executed,
                    )

                    log_event(dbg, "tool_result_framed",
                              tool=tool_name,
                              class_=cls,
                              budget_left=_budget_remaining(),
                              **frame_info)

                    payload = {
                        "tool": tool_name,
                        "result": framed_result,
                        "info": frame_info,
                        "executed": executed,
                    }
                    yield (
                        f"event: tool_result\n"
                        f"data: {json.dumps(payload)}\n\n"
                    )

                    # Native protocol: append a role=tool message keyed to
                    # the tool name. The SDK uses this to thread the result
                    # back to the model in the next turn.
                    messages.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": framed_result,
                    })

                    persisted_tool_calls.append(
                        {"name": tool_name, "args": tool_args}
                    )
                    persisted_tool_results.append({
                        "name": tool_name,
                        "executed": executed,
                        "delivered_chars": frame_info.get("delivered_chars", 0),
                        "original_chars": frame_info.get("original_chars", 0),
                        "truncated": bool(frame_info.get("truncated", False)),
                        "preview": (framed_result[:300] if framed_result else ""),
                    })

                # Refresh adaptive read budget after tool results land.
                used_tokens = _count_messages_tokens(messages)
                ctx_window = _context_window_for(model_gateway.get_current_model())
                set_context_window(ctx_window)
                set_available_tokens(max(0, ctx_window - used_tokens))
                log_event(dbg, "tool_results_fed_back",
                          msg_count=len(messages),
                          used_tokens=used_tokens,
                          context_window=ctx_window,
                          class_used=dict(class_used),
                          budget=get_budget_state())

                # R11: stream live context usage so the UI counter tracks the
                # real rolling total mid-turn instead of jumping from 0 → final
                # at the end. `used_tokens` here is already the full
                # `_count_messages_tokens(messages)` after tool results have
                # been appended; `ctx_window` reflects the *current* model's
                # window, which can legitimately change within a turn if the
                # user hot-swaps models (future), and is also what the
                # adaptive-read budget is measured against.
                yield (
                    f"event: token_usage\n"
                    f"data: {{\"used\": {used_tokens}, \"total\": {ctx_window}, "
                    f"\"phase\": \"tool_loop\"}}\n\n"
                )

                if forced_summary_reason:
                    break

                # If every class is now exhausted, do exactly one more pass
                # without tools so the model can summarise. Beyond that, the
                # outer loop's no-tool-calls branch ends the turn.

            # If we ran tools but the model never produced any content of its
            # own (rare with native tool calling, but possible when budgets
            # exhaust mid-iteration), do one final tool-less stream so the
            # user gets a conclusion instead of an empty bubble.
            needs_summary = (
                (not full_response.strip())
                and tool_iterations > 0
                and not await request.is_disconnected()
            ) or (
                forced_summary_reason
                and not await request.is_disconnected()
            )

            if needs_summary:
                log_event(dbg, "forced_summary",
                          tool_iterations=tool_iterations,
                          reason=forced_summary_reason or "empty_response")
                # Compact before the summary call to avoid Ollama OOM
                compacted = _compact_messages(messages, ctx_window_initial, dbg)
                messages.clear()
                messages.extend(compacted)
                nudge = (
                    "You have either spent your tool budget or already gathered "
                    "what you need. Answer the user with what is in your context. "
                    "Do not call any more tools."
                )
                messages.append({"role": "user", "content": nudge})
                try:
                    async for kind, payload in _stream_with_heartbeat(
                        model_gateway.chat_stream(messages),
                        interval=3.0,
                    ):
                        if await request.is_disconnected():
                            break
                        if kind == "heartbeat":
                            yield (
                                f"event: heartbeat\n"
                                f"data: {json.dumps({'phase': 'summary', 'label': 'Summarizing...'})}\n\n"
                            )
                            continue
                        if kind == "error":
                            raise payload
                        token_type, token = payload
                        if token_type == "content":
                            full_response += token
                            encoded = urllib.parse.quote(token)
                            yield f"event: token\ndata: {encoded}\n\n"
                except Exception as e:
                    log.error(f"Max-iterations summary stream error: {e}")
                    log_event(dbg, "max_iterations_summary_error", error=str(e))

            # Save final response to memory with structured tool metadata so
            # later turns can show compact stubs instead of replaying tools.
            # Persist even when content is empty if tool calls fired — the
            # assistant *did* act, and the turn record is what the conversation
            # tier (B2) and the on-disk audit trail rely on.
            if memory_store and (full_response or persisted_tool_calls):
                turn_metadata: dict = {}
                if persisted_tool_calls:
                    turn_metadata["tool_calls"] = persisted_tool_calls
                if persisted_tool_results:
                    turn_metadata["tool_results"] = persisted_tool_results
                memory_store.add_turn(
                    "assistant",
                    full_response,
                    conversation_id=conversation_id,
                    metadata=turn_metadata or None,
                )

            # R11: final token_usage is the true rolling total across the
            # full message list at end-of-turn, not `len(full_response)//4`
            # (which only counted the last assistant chunk and produced the
            # "counter appears to reset" bug users saw). Use the actual
            # model context window rather than the previously-hardcoded
            # 256000 so smaller-window models (e.g. 198k glm) report
            # accurately. `response_tokens` is kept in the debug log so we
            # can still chart assistant output size separately.
            final_used = _count_messages_tokens(messages)
            final_total = _context_window_for(model_gateway.get_current_model())
            response_tokens = len(full_response) // 4
            log_event(dbg, "chat_end", full_response_len=len(full_response),
                      response_tokens=response_tokens,
                      used_tokens=final_used,
                      context_window=final_total,
                      tool_iterations=tool_iterations,
                      had_nudge=not full_response.strip() and tool_iterations > 0)
            yield (
                f"event: token_usage\n"
                f"data: {{\"used\": {final_used}, \"total\": {final_total}, "
                f"\"phase\": \"final\"}}\n\n"
            )
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# Knowledge Base Endpoints
# Note: Specific routes must come BEFORE catch-all /kb/{folder}

@app.get("/kb/search")
async def search_kb(q: str):
    """Semantic search over KB."""
    if not kb_index:
        return []

    results = kb_index.search(q, top_k=5)
    return [
        {
            "path": r.get("path", ""),
            "snippet": (r.get("content") or "")[:200],
            "score": r.get("score", 0),
        }
        for r in results
    ]


@app.get("/kb/stats")
async def kb_stats():
    """Get KB statistics."""
    if not kb_index:
        return {"files": 0, "vectors": 0}

    stats = kb_index.get_stats()
    if kb_index.graph:
        stats["graph"] = kb_index.graph.get_stats()
    return stats


# ---------------------------------------------------------------------------
# P1-2 / P1-3: Graph viewer HTTP endpoints
#
# Both routes are human-facing (UI) and do NOT count against any tool budget.
# They reuse the same canonical-path + heading-resolution logic the agent
# tools use so a chat-bubble deep-link round-trips cleanly.
# ---------------------------------------------------------------------------

# Hard caps so a malicious/buggy client can never DoS the in-memory graph by
# asking for the whole vault as one Cytoscape blob.
_GRAPH_SUBGRAPH_MAX_NODES_CAP = 250
_GRAPH_SUBGRAPH_MAX_EDGES_CAP = 800
_GRAPH_SUBGRAPH_DEFAULT_NODES = 120
_GRAPH_SUBGRAPH_DEFAULT_EDGES = 400
_GRAPH_SUBGRAPH_DEFAULT_DEPTH = 2
_GRAPH_SUBGRAPH_VALID_EDGE_TYPES = {
    "similar", "inter_file", "cross_domain", "references", "relates_to",
    # parent_child intentionally NOT in the allow-list — it is structural
    # nesting and explodes the visual graph. The endpoint silently strips
    # it from any allow-list a client passes.
}


def _node_to_cyto(node, kb_index_local) -> dict:
    """Render a graph Node as a Cytoscape ``{data: {...}}`` dict."""
    from agent import kb_paths
    from knowledge.index import KBIndex
    src = (node.attributes or {}).get("source", "knowledge")
    try:
        cano = kb_paths.to_canonical(str(src), str(node.filename or ""))
    except Exception:
        cano = node.filename or ""
    tier = ""
    try:
        tier = KBIndex._compute_tier(src, node.filename or "")
    except Exception:
        pass
    leaf = (node.heading or "").split(" > ")[-1] if node.heading else ""
    # Summaries are already bounded at ~253 chars at the source (chunker's
    # mechanical "first 250 chars" fallback or LLM overview); no additional
    # clip here — let the UI show the full text the graph actually stores.
    return {
        "data": {
            "id": f"node:{node.id}",
            "label": leaf or (node.name or ""),
            "name": node.name or "",
            "file": cano,
            "heading": node.heading or "",
            "tier": tier,
            "source": src,
            "summary": node.summary or "",
            "tags": list(node.tags or []),
            "node_type": node.node_type.value,
        }
    }


def _edge_to_cyto(edge) -> dict:
    """Render a graph Edge as a Cytoscape ``{data: {...}, classes}`` dict."""
    et = edge.edge_type.value
    attrs = edge.attributes or {}
    evidence = (getattr(edge, "evidence", "") or "")[:140]
    link_kind = attrs.get("link_kind") or ""
    link_text = attrs.get("link_text") or ""
    return {
        "data": {
            "id": f"edge:{edge.source_id}:{edge.target_id}:{et}",
            "source": f"node:{edge.source_id}",
            "target": f"node:{edge.target_id}",
            "type": et,
            "weight": round(float(edge.weight), 3),
            "evidence": evidence,
            "link_kind": link_kind,
            "link_text": link_text,
        },
        "classes": et,
    }


@app.get("/kb/graph/subgraph")
async def kb_graph_subgraph(
    file: str,
    heading: str = "",
    depth: int = _GRAPH_SUBGRAPH_DEFAULT_DEPTH,
    max_nodes: int = _GRAPH_SUBGRAPH_DEFAULT_NODES,
    max_edges: int = _GRAPH_SUBGRAPH_DEFAULT_EDGES,
    edge_types: str = "",
    query: str = "",
):
    """Return a bounded subgraph rooted at ``(file, heading)`` in
    Cytoscape-compatible JSON.

    Hard caps:
      - depth: clamped to 1..3
      - max_nodes: clamped to 1..250
      - max_edges: clamped to 1..800

    Edge types: comma-separated allow-list. Default is every type EXCEPT
    ``parent_child``. ``parent_child`` is silently stripped if passed, since
    rendering structural nesting explodes the visual graph.

    Errors:
      - 400 on out-of-range params or invalid ``edge_types`` token.
      - 404 when ``file`` resolves to nothing on disk OR when ``heading``
        is ambiguous; body includes ``suggestions`` with the same
        ranking the chat-side resolver uses (P0-1).
    """
    from agent.tools import _resolve_chunk_nodes
    from agent import kb_paths
    from knowledge.graph import EdgeType

    if not kb_index or not kb_index.graph:
        raise HTTPException(503, "Knowledge graph not initialized")

    # Param validation — clamp + reject out-of-range values up front so the
    # caller gets a structured 400 instead of a silently-degraded answer.
    try:
        depth_i = int(depth)
        max_nodes_i = int(max_nodes)
        max_edges_i = int(max_edges)
    except (TypeError, ValueError):
        raise HTTPException(400, "depth / max_nodes / max_edges must be integers")
    if not (1 <= depth_i <= 3):
        raise HTTPException(400, "depth must be in 1..3")
    if not (1 <= max_nodes_i <= _GRAPH_SUBGRAPH_MAX_NODES_CAP):
        raise HTTPException(
            400,
            f"max_nodes must be in 1..{_GRAPH_SUBGRAPH_MAX_NODES_CAP}",
        )
    if not (1 <= max_edges_i <= _GRAPH_SUBGRAPH_MAX_EDGES_CAP):
        raise HTTPException(
            400,
            f"max_edges must be in 1..{_GRAPH_SUBGRAPH_MAX_EDGES_CAP}",
        )

    # Edge-type allow-list.
    requested: set[str] = set()
    dropped_pc = False
    if edge_types:
        for raw in str(edge_types).split(","):
            tok = raw.strip().lower()
            if not tok:
                continue
            if tok == "parent_child":
                dropped_pc = True
                continue
            if tok not in _GRAPH_SUBGRAPH_VALID_EDGE_TYPES:
                raise HTTPException(
                    400,
                    f"invalid edge_types token: '{tok}'. valid: "
                    f"{sorted(_GRAPH_SUBGRAPH_VALID_EDGE_TYPES)}",
                )
            requested.add(tok)
    else:
        requested = set(_GRAPH_SUBGRAPH_VALID_EDGE_TYPES)
    allowed_etv = requested
    excluded_edge_types = {EdgeType.PARENT_CHILD} | {
        et for et in EdgeType if et.value not in allowed_etv
    }

    # Resolve root node(s).
    graph = kb_index.graph
    nodes, disambig = _resolve_chunk_nodes(
        graph, file, heading,
        caller="graph_subgraph", query=query, kb_index=kb_index,
    )
    if disambig:
        raise HTTPException(
            404,
            detail={"error": "ambiguous", "suggestions": disambig},
        )
    if not nodes:
        raise HTTPException(
            404,
            detail={
                "error": "not_found",
                "message": f"No graph nodes for file='{file}' heading='{heading}'",
            },
        )

    # When heading is empty `_resolve_chunk_nodes` returns every chunk of
    # the file. Cap the seed set so the subgraph stays bounded even on
    # mega-files; downstream BFS will still respect max_nodes/max_edges.
    seed_nodes = nodes[:5]

    # BFS — bounded by max_nodes / max_edges. Each edge is emitted at
    # most once.
    selected_node_ids: dict[str, object] = {n.id: n for n in seed_nodes}
    selected_edges: list = []
    seen_edge_keys: set[str] = set()
    capped_nodes = False
    capped_edges = False

    queue: list[tuple[str, int]] = [(n.id, 0) for n in seed_nodes]
    while queue:
        current_id, current_depth = queue.pop(0)
        if current_depth >= depth_i:
            continue
        for neighbor, edge in graph.get_neighbors(current_id):
            if edge.edge_type in excluded_edge_types:
                continue
            ek = f"{edge.source_id}:{edge.target_id}:{edge.edge_type.value}"
            if ek in seen_edge_keys:
                continue
            # Node cap check before insertion.
            new_node = neighbor.id not in selected_node_ids
            if new_node and len(selected_node_ids) >= max_nodes_i:
                capped_nodes = True
                continue
            if len(selected_edges) >= max_edges_i:
                capped_edges = True
                continue
            if new_node:
                selected_node_ids[neighbor.id] = neighbor
                queue.append((neighbor.id, current_depth + 1))
            selected_edges.append(edge)
            seen_edge_keys.add(ek)

    # Stats over the subgraph itself (not the global graph).
    et_counts: dict[str, int] = {}
    for e in selected_edges:
        k = e.edge_type.value
        et_counts[k] = et_counts.get(k, 0) + 1
    sub_total = len(selected_edges)
    edge_share_sub = {
        k: round(v / sub_total, 4) for k, v in et_counts.items()
    } if sub_total > 0 else {}

    # Root meta — first seed node's canonical address.
    root = seed_nodes[0]
    src = (root.attributes or {}).get("source", "knowledge")
    try:
        root_cano = kb_paths.to_canonical(str(src), str(root.filename or ""))
    except Exception:
        root_cano = root.filename or ""

    elements = {
        "nodes": [
            _node_to_cyto(n, kb_index) for n in selected_node_ids.values()
        ],
        "edges": [_edge_to_cyto(e) for e in selected_edges],
    }
    return {
        "meta": {
            "root": {"file": root_cano, "heading": root.heading or ""},
            "depth": depth_i,
            "edge_types": sorted(allowed_etv),
            "dropped_parent_child": dropped_pc,
            "capped": {
                "nodes": capped_nodes,
                "edges": capped_edges,
                "max_nodes": max_nodes_i,
                "max_edges": max_edges_i,
            },
            "stats": {
                "nodes": len(elements["nodes"]),
                "edges": len(elements["edges"]),
                "edge_share": edge_share_sub,
            },
        },
        "elements": elements,
    }


@app.get("/kb/graph/overview")
async def kb_graph_overview(top: int = 10):
    """Dashboard rollup of the live graph for the empty-state of the
    viewer panel: top hubs, edge-type histogram, orphans, embedding model,
    last index time. Cheap — derived entirely from
    ``KnowledgeGraph.get_stats()``.
    """
    from agent import kb_paths

    if not kb_index or not kb_index.graph:
        raise HTTPException(503, "Knowledge graph not initialized")

    try:
        top_n = max(1, min(int(top), 50))
    except (TypeError, ValueError):
        top_n = 10

    g = kb_index.graph
    stats = g.get_stats()

    # Recompute top N hubs (get_stats returns top 5 by default; we want a
    # configurable slice for the UI).
    semantic_edge_counts: dict[str, int] = {}
    from knowledge.graph import EdgeType
    for edge in g.edges.values():
        if edge.edge_type == EdgeType.PARENT_CHILD:
            continue
        semantic_edge_counts[edge.source_id] = semantic_edge_counts.get(edge.source_id, 0) + 1
        semantic_edge_counts[edge.target_id] = semantic_edge_counts.get(edge.target_id, 0) + 1
    total_non_pc = sum(
        c for et, c in stats.get("edge_types", {}).items() if et != "parent_child"
    )

    hubs = []
    for nid, count in sorted(semantic_edge_counts.items(), key=lambda x: -x[1])[:top_n]:
        node = g.nodes.get(nid)
        if not node:
            continue
        src = (node.attributes or {}).get("source", "knowledge")
        try:
            cano = kb_paths.to_canonical(str(src), str(node.filename or ""))
        except Exception:
            cano = node.filename or ""
        share = (count / total_non_pc) if total_non_pc > 0 else 0.0
        hubs.append({
            "id": f"node:{node.id}",
            "file": cano,
            "heading": node.heading or "",
            "count": count,
            "share_non_pc": round(share, 4),
        })

    return {
        "embedding_model": getattr(kb_index, "_embedding_model", None),
        "last_indexed_at": getattr(kb_index, "_last_indexed_at", None),
        "nodes": stats.get("nodes", 0),
        "edges": stats.get("edges", 0),
        "orphan_nodes": stats.get("orphan_nodes", 0),
        "avg_edges_per_node": stats.get("avg_edges_per_node", 0.0),
        "edge_types": stats.get("edge_types", {}),
        "edge_share": stats.get("edge_share", {}),
        "hubs": hubs,
    }


@app.post("/kb/reindex")
async def reindex_kb(request: Request):
    """Rebuild the KB index. POST body options: {entities: true, summaries: true}."""
    if not kb_index:
        raise HTTPException(503, "Index not initialized")

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass  # No body or invalid JSON — use defaults
    extract_entities = body.get("entities", False)
    llm_summaries = body.get("summaries", False)

    await kb_index.build_index_async(extract_entities=extract_entities, llm_summaries=llm_summaries, force=True)
    return {"status": "ok", "graph": kb_index.graph.get_stats() if kb_index.graph else None}


@app.get("/kb/folder-tree")
async def folder_tree(folder: str = "knowledge"):
    """Get the folder hierarchy tree for LLM consumption."""
    if not kb_index:
        return {"tree": ""}

    return {"tree": kb_index.get_folder_tree(source=folder)}


@app.get("/kb/file/{path:path}")
async def get_kb_file(path: str):
    """Get file content."""
    file_path = Path(f"/app/knowledge/{path}")
    if not file_path.exists():
        # Try canon
        file_path = Path(f"/app/canon/{path}")

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    return HTMLResponse(file_path.read_text())


@app.get("/kb/{folder}")
async def list_kb_files(folder: str):
    """List files in knowledge or canon folder (recursive, includes subfolders)."""
    if folder not in ("knowledge", "canon"):
        raise HTTPException(400, "Folder must be 'knowledge' or 'canon'")

    base_dir = Path(f"/app/{folder}")
    if not base_dir.exists():
        return []

    files = []
    for f in base_dir.rglob("**/*.md"):
        files.append({
            "name": f.name,
            "path": str(f.relative_to(base_dir)),
            "folder": str(f.parent.relative_to(base_dir)),
            "last_modified_by": _get_frontmatter_value(f, "last_modified_by"),
        })
    return files


# Model Selection Endpoints
# Known cloud models available through Ollama (device-registered)
CLOUD_MODELS = [
    "devstral-2:123b-cloud",
    "glm-5.1:cloud",
    "minimax-m2.7:cloud",
    "gemma4:31b-cloud",
    "gpt-oss:120b-cloud",
    "gpt-oss:20b-cloud",
    "deepseek-v3.1:671b-cloud",
]


@app.get("/models")
async def list_models():
    """List available models (local + known cloud)."""
    if not model_gateway:
        raise HTTPException(503, "Model gateway not initialized")

    local_models = await model_gateway.get_available_models()
    # Merge: local models first, then cloud models not already in list
    seen = set(local_models)
    all_models = list(local_models)
    for cm in CLOUD_MODELS:
        if cm not in seen:
            all_models.append(cm)
    return {"models": all_models, "current": model_gateway.get_current_model()}


@app.post("/model")
async def set_model(request: Request):
    """Set the current model."""
    if not model_gateway:
        raise HTTPException(503, "Model gateway not initialized")

    body = await request.json()
    model = body.get("model")

    if not model:
        raise HTTPException(400, "Model name required")

    model_gateway.set_model(model)
    log.info(f"Model switched to: {model}")
    return {"status": "ok", "model": model, "success": True}


def _get_frontmatter_value(path: Path, key: str) -> str | None:
    """Extract a value from YAML frontmatter."""
    try:
        content = path.read_text()
        if not content.startswith("---"):
            return None
        lines = content[3:].split("\n")
        for line in lines:
            if line.strip() == "---":
                break
            if line.startswith(f"{key}:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None