"""Phase 3: context management tests (post-A1 native tool calling).

Covers:
- Tool metadata persisted on assistant turns after a chat with tool calls.
- Token-budgeted history walk (compact stubs for old tool turns).
- Native tool calling protocol -- assistant history fed to the model never
  contains the legacy [TOOL: ...] brackets.
- Context-aware intra-turn cap (>50% of ctx forces summary mode).
- E2E: simulated repeated tool-heavy turns stay bounded in token cost.
"""

import lancedb
from fastapi.testclient import TestClient

from agent.tools import KBTools
from knowledge.index import KBIndex
from memory.store import ConversationStore
from models.gateway import ModelGateway
from web.app import (
    _count_messages_tokens,
    app,
    set_components,
)
from tests.conftest import FakeEmbeddingFunction


def _make_components(tmp_path, fake_chat_stream, graph_stats_text="Nodes: 0\nEdges: 0",
                     model_name="glm-5.1:cloud"):
    gateway = ModelGateway()
    gateway.chat_stream = fake_chat_stream  # type: ignore[assignment]
    gateway.get_current_model = lambda: model_name  # type: ignore[assignment]
    # Force tool support so the chat loop attaches our tools to fake_chat_stream.
    gateway.supports_tools = lambda model=None: True  # type: ignore[assignment]

    kb_index = KBIndex()
    kb_index.db = lancedb.connect(str(tmp_path / "lancedb"))
    kb_index._embedding_fn = FakeEmbeddingFunction()
    store = ConversationStore(sessions_dir=tmp_path / "sessions")
    store.initialize()
    tools = KBTools(kb_index=kb_index)
    tools.graph_stats = lambda: graph_stats_text  # type: ignore[assignment]
    set_components(gateway, kb_index, store, tools)
    return gateway, kb_index, store, tools


class TestToolMetadataPersisted:
    """After a chat that ran tools, the assistant turn carries structured metadata."""

    def test_assistant_turn_has_tool_calls_and_results(self, tmp_path):
        call_count = {"n": 0}

        async def fake_chat_stream(messages, tools=None, think=True):
            call_count["n"] += 1
            if call_count["n"] == 1:
                yield ("content", "Looking it up.")
                yield ("tool_call", {"name": "graph_stats", "arguments": {}})
            else:
                yield ("content", "Final answer based on results.")

        _, _, store, _ = _make_components(tmp_path, fake_chat_stream)

        with TestClient(app, raise_server_exceptions=True) as client:
            conv_id = client.post("/conversations", json={}).json()["id"]
            with client.stream(
                "POST", "/chat",
                json={"message": "stats?", "conversation_id": conv_id, "tools_enabled": True},
            ) as resp:
                _ = "".join(resp.iter_text())

        turns = store.get_conversation(conv_id)
        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        assert assistant_turns, "no assistant turn persisted"
        last = assistant_turns[-1]
        assert "tool_calls" in last
        assert last["tool_calls"][0]["name"] == "graph_stats"
        assert "tool_results" in last
        assert last["tool_results"][0]["name"] == "graph_stats"
        assert "delivered_chars" in last["tool_results"][0]
        assert "truncated" in last["tool_results"][0]


class TestNoLegacyToolBracketsInHistory:
    """A1: native tool calling means stored assistant turns never carry the
    legacy [TOOL: ...] inline syntax. The chat loop must not invent it on
    re-feed either."""

    def test_assistant_messages_in_chat_call_are_clean(self, tmp_path):
        seen_messages: list[list[dict]] = []

        async def fake_chat_stream(messages, tools=None, think=True):
            seen_messages.append([dict(m) for m in messages])
            yield ("content", "Reply with no tool call.")

        _, _, store, _ = _make_components(tmp_path, fake_chat_stream)

        with TestClient(app, raise_server_exceptions=True) as client:
            conv_id = client.post("/conversations", json={}).json()["id"]
            store.add_turn("user", "first", conversation_id=conv_id)
            store.add_turn(
                "assistant",
                "Sure, here you go.",
                conversation_id=conv_id,
            )

            with client.stream(
                "POST", "/chat",
                json={"message": "next", "conversation_id": conv_id, "tools_enabled": True},
            ) as resp:
                _ = "".join(resp.iter_text())

        assert seen_messages, "model never called"
        msgs = seen_messages[0]
        assistant_in_history = [m for m in msgs if m["role"] == "assistant"]
        assert assistant_in_history, "history did not feed the assistant turn"
        for m in assistant_in_history:
            assert "[TOOL:" not in m["content"], (
                "raw tool brackets must never appear under native tool calling"
            )


class TestBudgetedHistoryWalk:
    """get_history_within_budget bounds context regardless of conversation length."""

    def test_long_conversation_stays_bounded(self, tmp_path):
        store = ConversationStore(sessions_dir=tmp_path / "sessions")
        store.initialize()
        conv_id = store.create_conversation()
        big_chunk = "lorem ipsum dolor sit amet " * 200
        for i in range(20):
            store.add_turn("user", f"q{i}", conversation_id=conv_id)
            store.add_turn(
                "assistant",
                f"answer {i}: {big_chunk}",
                conversation_id=conv_id,
                metadata={
                    "tool_calls": [{"name": "graph_stats", "args": {}}],
                    "tool_results": [{
                        "name": "graph_stats",
                        "delivered_chars": len(big_chunk),
                        "original_chars": len(big_chunk),
                        "truncated": False,
                        "preview": big_chunk[:100],
                    }],
                },
            )

        budget = 2000
        out = store.get_history_within_budget(conv_id, max_tokens=budget, always_full_n=2)
        assert out[-1]["content"].startswith("answer 19:")
        older_stubs = [t for t in out[:-2] if t["role"] == "assistant"]
        for stub in older_stubs:
            assert stub["content"].startswith("[earlier turn:"), (
                f"older assistant turn should be a stub, got: {stub['content'][:80]!r}"
            )


class TestIntraTurnCap:
    """When tool messages exceed 50% of ctx, force summary mode."""

    def test_tool_loop_breaks_when_context_over_50pct(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            __import__("web.app", fromlist=["MODEL_CONTEXT_WINDOWS"]).MODEL_CONTEXT_WINDOWS,
            "tiny-test-model",
            1000,
        )

        big_blob = "x. " * 1000  # ~3000 chars per call -- ~750 tokens

        async def fake_chat_stream(messages, tools=None, think=True):
            # Always emit one tool_call so the loop keeps going until the
            # 50%-of-ctx safety brake triggers.
            yield ("tool_call", {"name": "graph_stats", "arguments": {}})

        _, _, store, _ = _make_components(
            tmp_path, fake_chat_stream,
            graph_stats_text=big_blob, model_name="tiny-test-model",
        )

        with TestClient(app, raise_server_exceptions=True) as client:
            conv_id = client.post("/conversations", json={}).json()["id"]
            with client.stream(
                "POST", "/chat",
                json={"message": "blow it up", "conversation_id": conv_id, "tools_enabled": True},
            ) as resp:
                _ = "".join(resp.iter_text())

        turns = store.get_conversation(conv_id)
        assistant_turns = [t for t in turns if t["role"] == "assistant"]
        assert assistant_turns, "expected an assistant turn to be saved"


class TestE2ENoReplay:
    """E2E: a follow-up turn does NOT replay every prior tool result.

    Repro of the deadlock the user reported: in the old impl, history
    grew unbounded with full tool transcripts. After Phase 3, older
    tool-bearing turns become compact stubs.
    """

    def test_followup_turn_does_not_repeat_tool_results(self, tmp_path):
        seen_messages: list[list[dict]] = []
        call_count = {"n": 0}

        async def fake_chat_stream(messages, tools=None, think=True):
            seen_messages.append([dict(m) for m in messages])
            call_count["n"] += 1
            if call_count["n"] == 1:
                yield ("content", "Checking")
                yield ("tool_call", {"name": "graph_stats", "arguments": {}})
            else:
                yield ("content", "Done.")

        _, _, store, _ = _make_components(
            tmp_path, fake_chat_stream,
            graph_stats_text="Nodes: 99\nEdges: 42\n" + ("info " * 1000),
        )

        with TestClient(app, raise_server_exceptions=True) as client:
            conv_id = client.post("/conversations", json={}).json()["id"]
            with client.stream(
                "POST", "/chat",
                json={"message": "stats?", "conversation_id": conv_id, "tools_enabled": True},
            ) as resp:
                _ = "".join(resp.iter_text())

            seen_messages.clear()

            with client.stream(
                "POST", "/chat",
                json={"message": "thanks!", "conversation_id": conv_id, "tools_enabled": True},
            ) as resp:
                _ = "".join(resp.iter_text())

        assert seen_messages, "no chat_stream calls captured for turn 2"
        first_call_msgs = seen_messages[0]
        history_assistant = [
            m for m in first_call_msgs
            if m["role"] == "assistant"
        ]
        assert history_assistant, "expected the prior assistant turn in history"
        for m in history_assistant:
            assert "Nodes: 99" not in m["content"], (
                "tool result body leaked back into model context on follow-up"
            )

        total_tokens = _count_messages_tokens(first_call_msgs)
        assert total_tokens < 5000, (
            f"context for follow-up turn exploded to {total_tokens} tokens; "
            "Phase 3 budget walk is leaking the prior tool transcript"
        )
