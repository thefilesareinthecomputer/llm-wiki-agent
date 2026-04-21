"""B2 — search_conversations + read_conversation tools.

These tests exercise the conversation memory tier without touching the chat
loop. They verify:

1. ``search_conversations`` returns ranked turns from prior threads using the
   same embedding pipeline as KB search.
2. ``read_conversation`` honors ``last:N`` and ``range:A:B`` slicing,
   refusing malformed inputs with a clear error.
3. Both tools are registered in the explore budget class so the chat loop
   counts them against the same cap as ``search_knowledge``.
4. The tool registry only exposes them when a ConversationStore is wired,
   so models hosted by KBTools instances without conversation access never
   see a tool they can't actually run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import FakeEmbeddingFunction


@pytest.fixture
def conv_env(tmp_path):
    """Build a KBTools wired to an in-tmp ConversationStore + fake embeddings."""
    from memory.store import ConversationStore
    from agent.tools import KBTools

    class _StubKBIndex:
        _embedding_fn = FakeEmbeddingFunction()

    store = ConversationStore(sessions_dir=tmp_path / "sessions")
    store.initialize()

    # Seed three conversations with distinct content.
    conv_a = store.create_conversation()
    store.add_turn("user", "How do I set up tea ceremony tools?", conversation_id=conv_a)
    store.add_turn("assistant", "Start by gathering chasen and chawan.", conversation_id=conv_a)

    conv_b = store.create_conversation()
    store.add_turn("user", "Tell me about Kimball dimensional modeling and SCD Type 2.",
                   conversation_id=conv_b)
    store.add_turn(
        "assistant",
        "Slowly Changing Dimensions Type 2 preserve history by adding rows.",
        conversation_id=conv_b,
    )

    conv_c = store.create_comversation = None  # noqa: just to silence linters
    conv_c = store.create_conversation()
    store.add_turn("user", "What does my morning routine look like?", conversation_id=conv_c)
    store.add_turn(
        "assistant",
        "It includes journaling, hydration, and 30 minutes of focused work.",
        conversation_id=conv_c,
    )

    tools = KBTools(
        kb_index=_StubKBIndex(),
        kb_dir=tmp_path / "knowledge",
        canon_dir=tmp_path / "canon",
        conversation_store=store,
    )
    return {
        "tools": tools,
        "store": store,
        "conv_a": conv_a,
        "conv_b": conv_b,
        "conv_c": conv_c,
        "sessions_dir": tmp_path / "sessions",
    }


def test_search_conversations_returns_ranked_hits(conv_env):
    out = conv_env["tools"].search_conversations(
        "Tell me about Kimball dimensional modeling and SCD Type 2.",
        limit=3,
    )
    assert "Conversation matches" in out
    assert conv_env["conv_b"] in out, (
        "the conversation that exactly matches the query must rank in top 3; "
        f"got:\n{out}"
    )


def test_search_conversations_caches_embeddings_on_disk(conv_env):
    """Second call should reuse cached embeddings — verified by checking
    that ``embedding`` keys land in the persisted session JSON.
    """
    conv_env["tools"].search_conversations("morning routine", limit=2)
    sessions_dir: Path = conv_env["sessions_dir"]
    cached = False
    for path in sessions_dir.glob("*.json"):
        data = json.loads(path.read_text())
        for turn in data.get("turns", []):
            if isinstance(turn.get("embedding"), list) and turn["embedding"]:
                cached = True
                break
        if cached:
            break
    assert cached, (
        "search_conversations must persist per-turn embeddings to the session "
        "JSON so subsequent searches don't re-embed every turn."
    )


def test_search_conversations_requires_query(conv_env):
    out = conv_env["tools"].search_conversations("   ", limit=3)
    assert "query is required" in out


def test_search_conversations_clear_error_when_unwired(tmp_path):
    """KBTools instances built without a ConversationStore must not pretend
    the tool works — the model needs an actionable failure message.
    """
    from agent.tools import KBTools

    class _StubKBIndex:
        _embedding_fn = FakeEmbeddingFunction()

    tools = KBTools(
        kb_index=_StubKBIndex(),
        kb_dir=tmp_path / "knowledge",
        canon_dir=tmp_path / "canon",
        conversation_store=None,
    )
    out = tools.search_conversations("anything")
    assert "not wired" in out


def test_read_conversation_last_n(conv_env):
    out = conv_env["tools"].read_conversation(conv_env["conv_b"], "last:1")
    assert f"Conversation {conv_env['conv_b']}" in out
    assert "showing 1" in out
    assert "Slowly Changing Dimensions" in out


def test_read_conversation_range_inclusive(conv_env):
    out = conv_env["tools"].read_conversation(conv_env["conv_b"], "range:0:1")
    assert "turn[0]" in out
    assert "turn[1]" in out


def test_read_conversation_rejects_bad_spec(conv_env):
    out = conv_env["tools"].read_conversation(conv_env["conv_a"], "garbage")
    assert "must be 'last:N' or 'range:A:B'" in out


def test_read_conversation_unknown_id_is_safe(conv_env):
    out = conv_env["tools"].read_conversation("does-not-exist", "last:5")
    assert "no turns" in out


def test_conversation_tools_belong_to_explore_class():
    """B2 + A3: the conversation tools must inherit the explore budget so
    the model can't bypass per-class caps by routing reads through them.
    """
    from agent.tools import TOOL_CLASSES, class_for_tool

    assert TOOL_CLASSES["search_conversations"] == "explore"
    assert TOOL_CLASSES["read_conversation"] == "explore"
    assert class_for_tool("search_conversations") == "explore"
    assert class_for_tool("read_conversation") == "explore"


def test_registry_only_exposes_conversation_tools_when_store_wired(conv_env, tmp_path):
    from agent.tools import KBTools, build_tool_registry

    class _StubKBIndex:
        _embedding_fn = FakeEmbeddingFunction()

    wired_registry = build_tool_registry(conv_env["tools"])
    assert "search_conversations" in wired_registry
    assert "read_conversation" in wired_registry

    bare = KBTools(
        kb_index=_StubKBIndex(),
        kb_dir=tmp_path / "k",
        canon_dir=tmp_path / "c",
        conversation_store=None,
    )
    bare_registry = build_tool_registry(bare)
    assert "search_conversations" not in bare_registry, (
        "exposing search_conversations without a ConversationStore would "
        "let the model call a tool that always returns 'not wired'."
    )
    assert "read_conversation" not in bare_registry
