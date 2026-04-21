"""Tests for tool result framing (Phase 1 honesty fixes).

Covers:
- _frame_tool_result: COMPLETE vs TRUNCATED markers, per-tool overrides
- _count_messages_tokens: token accounting across messages array
- Per-model context window resolution
- read_knowledge_section bypass (already self-truncates honestly)
"""

import pytest
from web.app import (
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    MODEL_CONTEXT_WINDOWS,
    DEFAULT_CONTEXT_WINDOW,
    TOOL_RESULT_OVERRIDES,
    _context_window_for,
    _count_messages_tokens,
    _frame_tool_result,
)


class TestFrameToolResult:
    """Structured COMPLETE/TRUNCATED markers around raw tool output."""

    def test_short_result_complete_marker(self):
        framed, info = _frame_tool_result("graph_stats", "Nodes: 4\nEdges: 2")
        first = framed.splitlines()[0]
        assert first.startswith("[TOOL_RESULT: graph_stats | COMPLETE")
        assert info["truncated"] is False
        assert info.get("executed") is True
        assert info["original_chars"] == info["delivered_chars"]

    def test_long_result_truncated_marker(self):
        big = "Lorem ipsum dolor sit amet. " * 1000  # ~28k chars
        framed, info = _frame_tool_result("search_knowledge", big)
        first = framed.splitlines()[0]
        assert first.startswith("[TOOL_RESULT: search_knowledge | TRUNCATED")
        assert info["truncated"] is True
        assert info["delivered_chars"] < info["original_chars"]
        # Header must show both numbers (honest accounting).
        assert f"{info['original_chars']:,}" in first
        assert f"{info['delivered_chars']:,}" in first

    def test_section_tool_passes_through_unmodified(self):
        """read_knowledge_section already self-reports completeness; no extra framing."""
        raw = (
            "[SECTION: foo.md | 1/3 | Heading | LOADED 100 of 100 tokens (COMPLETE)]\n\n"
            "Real content here."
        )
        framed, info = _frame_tool_result("read_knowledge_section", raw)
        assert framed == raw  # Pass-through
        assert info["truncated"] is False

    def test_per_tool_override_applies(self):
        """graph_traverse uses 12000-char cap, not the default 8000."""
        assert TOOL_RESULT_OVERRIDES["graph_traverse"]["max_chars"] == 12000
        body = "x" * 10000  # over default cap, under graph_traverse cap
        framed, info = _frame_tool_result("graph_traverse", body)
        assert info["truncated"] is False  # would be True under default cap

    def test_default_cap_used_for_unknown_tool(self):
        body = "x" * (DEFAULT_TOOL_RESULT_MAX_CHARS + 100)
        framed, info = _frame_tool_result("nonexistent_tool", body)
        assert info["truncated"] is True

    def test_truncation_at_sentence_boundary(self):
        """Truncated output ends at sentence/word boundary, not mid-token."""
        big = ("This is a sentence. " * 800)  # ~16k chars
        framed, info = _frame_tool_result("search_knowledge", big)
        body = framed.split("\n", 1)[1]
        # Should not end mid-word
        assert body.endswith(".") or body.endswith(" ") or body.endswith("\n")

    def test_empty_result(self):
        framed, info = _frame_tool_result("graph_stats", "")
        assert info["original_chars"] == 0
        assert "COMPLETE 0 chars" in framed

    def test_not_executed_uses_distinct_header(self):
        """Budget / duplicate skips must not claim COMPLETE on a refusal string."""
        raw = "REFUSED: write budget exhausted (used 2/2 this turn)."
        framed, info = _frame_tool_result("save_knowledge", raw, executed=False)
        first = framed.splitlines()[0]
        assert first == "[TOOL_RESULT: save_knowledge | NOT_EXECUTED]"
        assert "COMPLETE" not in first
        assert info["executed"] is False
        assert info["truncated"] is False
        assert raw in framed


class TestContextWindowResolution:
    """Per-model context window lookup."""

    def test_known_model(self):
        assert _context_window_for("glm-5.1:cloud") == 198000

    def test_unknown_model_default(self):
        assert _context_window_for("some-random-model") == DEFAULT_CONTEXT_WINDOW

    def test_default_is_at_least_32k(self):
        assert DEFAULT_CONTEXT_WINDOW >= 32000


class TestMessageTokenCounting:
    """_count_messages_tokens drives the adaptive budget."""

    def test_empty_messages(self):
        assert _count_messages_tokens([]) == 0

    def test_growing_messages_grow_count(self):
        a = [{"role": "user", "content": "hello"}]
        b = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        assert _count_messages_tokens(b) > _count_messages_tokens(a)

    def test_handles_missing_content(self):
        # Should not crash on malformed messages.
        msgs = [{"role": "user"}]
        # Just shouldn't throw; result is small but defined.
        n = _count_messages_tokens(msgs)
        assert isinstance(n, int)
        assert n >= 0


class TestChatLoopFraming:
    """E2E: tool loop with oversized result feeds correct TRUNCATED marker
    into next iteration's messages.
    """

    def test_oversized_search_result_is_framed_in_sse(self, monkeypatch, tmp_path):
        """When a tool returns >8000 chars, the SSE stream contains a TRUNCATED marker."""
        from fastapi.testclient import TestClient
        import lancedb
        from web.app import app, set_components, _execute_tool
        from knowledge.index import KBIndex
        from models.gateway import ModelGateway
        from memory.store import ConversationStore
        from agent.tools import KBTools, reset_budget
        from tests.conftest import FakeEmbeddingFunction

        # Patch _execute_tool to return a giant string for graph_stats
        big = "x. " * 5000  # 15,000 chars

        gateway = ModelGateway()

        async def fake_chat_stream(messages, tools=None, think=True):
            yield ("tool_call", {"name": "graph_stats", "arguments": {}})

        monkeypatch.setattr(gateway, "chat_stream", fake_chat_stream)
        monkeypatch.setattr(gateway, "get_current_model", lambda: "glm-5.1:cloud")
        monkeypatch.setattr(gateway, "supports_tools", lambda model=None: True)

        kb_index = KBIndex()
        kb_index.db = lancedb.connect(str(tmp_path / "lancedb"))
        kb_index._embedding_fn = FakeEmbeddingFunction()
        store = ConversationStore(sessions_dir=tmp_path / "sessions")
        store.initialize()
        tools = KBTools(kb_index=kb_index)

        # Patch graph_stats on the tools instance to return our big string
        monkeypatch.setattr(tools, "graph_stats", lambda: big)
        set_components(gateway, kb_index, store, tools)

        with TestClient(app, raise_server_exceptions=True) as client:
            # Create a conversation, then chat
            conv = client.post("/conversations", json={}).json()
            conv_id = conv["id"]
            with client.stream(
                "POST", "/chat",
                json={"message": "test", "conversation_id": conv_id, "tools_enabled": True}
            ) as resp:
                body = "".join(chunk for chunk in resp.iter_text())

        # The SSE stream must include the TRUNCATED marker for the oversized result.
        assert "TOOL_RESULT: graph_stats" in body
        assert "TRUNCATED" in body or "COMPLETE" in body
        # In this test the result is 15,000 chars > 4,000 cap (graph_stats override),
        # so it must be TRUNCATED specifically.
        assert "TRUNCATED" in body


class TestPhase2LifecycleEvents:
    """Phase 2: SSE lifecycle events for the UI.

    iteration_start fires before any tokens of a new tool-loop pass.
    tool_executing fires before _execute_tool, tool_done after.
    heartbeat fires when the model is silent for >3s (not exercised here
    because the fake stream returns immediately, but the wiring is tested
    in test_heartbeat_emits_during_silent_gap).
    """

    def test_iteration_lifecycle_event_ordering(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        import lancedb
        from web.app import app, set_components
        from knowledge.index import KBIndex
        from models.gateway import ModelGateway
        from memory.store import ConversationStore
        from agent.tools import KBTools
        from tests.conftest import FakeEmbeddingFunction

        gateway = ModelGateway()

        # First call → emits a tool call so the loop iterates.
        # Second call → no tool call → loop exits.
        call_count = {"n": 0}

        async def fake_chat_stream(messages, tools=None, think=True):
            call_count["n"] += 1
            if call_count["n"] == 1:
                yield ("content", "Calling now.")
                yield ("tool_call", {"name": "graph_stats", "arguments": {}})
            else:
                yield ("content", "All done.")

        monkeypatch.setattr(gateway, "chat_stream", fake_chat_stream)
        monkeypatch.setattr(gateway, "get_current_model", lambda: "glm-5.1:cloud")
        monkeypatch.setattr(gateway, "supports_tools", lambda model=None: True)

        kb_index = KBIndex()
        kb_index.db = lancedb.connect(str(tmp_path / "lancedb"))
        kb_index._embedding_fn = FakeEmbeddingFunction()
        store = ConversationStore(sessions_dir=tmp_path / "sessions")
        store.initialize()
        tools = KBTools(kb_index=kb_index)
        monkeypatch.setattr(tools, "graph_stats", lambda: "Nodes: 0\nEdges: 0")
        set_components(gateway, kb_index, store, tools)

        with TestClient(app, raise_server_exceptions=True) as client:
            conv = client.post("/conversations", json={}).json()
            with client.stream(
                "POST", "/chat",
                json={"message": "go", "conversation_id": conv["id"], "tools_enabled": True},
            ) as resp:
                body = "".join(chunk for chunk in resp.iter_text())

        # Required lifecycle events present
        assert "event: iteration_start" in body
        assert "event: tool_executing" in body
        assert "event: tool_done" in body
        assert "event: tool_result" in body

        # Ordering: iteration_start → tool_executing → tool_done → tool_result
        idx_iter = body.index("event: iteration_start")
        idx_exec = body.index("event: tool_executing")
        idx_done = body.index("event: tool_done")
        idx_res = body.index("event: tool_result")
        assert idx_iter < idx_exec < idx_done
        # tool_result fires after tool_done (server emits done immediately
        # after the tool returns, then frames + emits the result)
        assert idx_done < idx_res

    def test_heartbeat_emits_during_silent_gap(self, monkeypatch, tmp_path):
        """When the model stalls for >3s before first token, a heartbeat fires."""
        from fastapi.testclient import TestClient
        import lancedb
        from web.app import app, set_components
        from knowledge.index import KBIndex
        from models.gateway import ModelGateway
        from memory.store import ConversationStore
        from agent.tools import KBTools
        from tests.conftest import FakeEmbeddingFunction

        gateway = ModelGateway()

        import asyncio

        async def slow_stream(messages, tools=None, think=True):
            await asyncio.sleep(3.5)  # > 3s heartbeat interval
            yield ("content", "ok")

        monkeypatch.setattr(gateway, "chat_stream", slow_stream)
        monkeypatch.setattr(gateway, "get_current_model", lambda: "glm-5.1:cloud")
        monkeypatch.setattr(gateway, "supports_tools", lambda model=None: True)

        kb_index = KBIndex()
        kb_index.db = lancedb.connect(str(tmp_path / "lancedb"))
        kb_index._embedding_fn = FakeEmbeddingFunction()
        store = ConversationStore(sessions_dir=tmp_path / "sessions")
        store.initialize()
        tools = KBTools(kb_index=kb_index)
        set_components(gateway, kb_index, store, tools)

        with TestClient(app, raise_server_exceptions=True) as client:
            conv = client.post("/conversations", json={}).json()
            with client.stream(
                "POST", "/chat",
                json={"message": "go", "conversation_id": conv["id"], "tools_enabled": True},
            ) as resp:
                body = "".join(chunk for chunk in resp.iter_text())

        assert "event: heartbeat" in body
        # Heartbeat must precede the eventual content token
        idx_hb = body.index("event: heartbeat")
        idx_token = body.index("event: token")
        assert idx_hb < idx_token


# NOTE: A1 deleted src/agent/tool_parser.py and the matching JS strip regex
# in ui/app.js. With Ollama native tool calling the model emits structured
# tool_call objects instead of `[TOOL: ...]` text, so there is nothing to
# strip on either side. The end-to-end framing of tool results is now
# covered by tests/integration/test_native_tool_loop.py.
