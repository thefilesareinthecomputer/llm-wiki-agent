"""
Tests for conversation memory store.
JSON file-based conversation session management.
"""

import json
import pytest
from pathlib import Path
from memory.store import ConversationStore


class TestConversationStore:
    """Test JSON-backed conversation session store."""

    def test_initialize_creates_sessions_dir(self, tmp_path):
        """Initialize creates the sessions directory."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        assert sessions_dir.exists()

    def test_initialize_idempotent(self, tmp_path):
        """Initialize is safe to call multiple times."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        store.initialize()  # Should not raise
        assert sessions_dir.exists()

    def test_create_conversation_returns_uuid(self, tmp_path):
        """create_conversation returns a valid UUID4."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_id = store.create_conversation()
        assert isinstance(conv_id, str)
        assert len(conv_id) == 36  # UUID4 format: 8-4-4-4-12
        assert conv_id.count("-") == 4

    def test_create_conversation_creates_json_file(self, tmp_path):
        """Creating a conversation writes a JSON file."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_id = store.create_conversation()
        json_path = sessions_dir / f"{conv_id}.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["id"] == conv_id
        assert data["title"] == "New Chat"
        assert data["turns"] == []

    def test_add_turn(self, tmp_path):
        """Adding turns stores them in the conversation JSON."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_id = store.create_conversation()
        store.add_turn("user", "Hello", conversation_id=conv_id)
        store.add_turn("assistant", "Hi there!", conversation_id=conv_id)

        turns = store.get_recent(conversation_id=conv_id, n=10)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Hello"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "Hi there!"

    def test_conversation_isolation(self, tmp_path):
        """Turns from different conversations don't bleed."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_a = store.create_conversation()
        conv_b = store.create_conversation()
        store.add_turn("user", "A message", conversation_id=conv_a)
        store.add_turn("assistant", "A reply", conversation_id=conv_a)
        store.add_turn("user", "B message", conversation_id=conv_b)

        a_turns = store.get_recent(conversation_id=conv_a, n=10)
        b_turns = store.get_recent(conversation_id=conv_b, n=10)
        assert len(a_turns) == 2
        assert len(b_turns) == 1
        assert any("A message" in t["content"] for t in a_turns)
        assert not any("B message" in t["content"] for t in a_turns)

    def test_get_recent_limits_and_returns_last_n(self, tmp_path):
        """get_recent returns the last N turns, not more."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_id = store.create_conversation()
        for i in range(20):
            store.add_turn("user", f"msg {i}", conversation_id=conv_id)

        recent = store.get_recent(conversation_id=conv_id, n=5)
        assert len(recent) == 5
        # Should be the LAST 5 (most recent)
        assert "msg 19" in recent[-1]["content"]
        assert "msg 15" in recent[0]["content"]

    def test_list_conversations(self, tmp_path):
        """list_conversations returns all conversations with metadata."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_id = store.create_conversation()
        store.add_turn("user", "Hello there world", conversation_id=conv_id)

        convs = store.list_conversations()
        assert isinstance(convs, list)
        assert len(convs) >= 1
        found = [c for c in convs if c["id"] == conv_id]
        assert len(found) == 1
        assert "title" in found[0]
        assert "created_at" in found[0]
        assert "turn_count" in found[0]

    def test_list_sorted_by_updated_at_desc(self, tmp_path):
        """Most recently updated conversation appears first."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        old_id = store.create_conversation()
        store.add_turn("user", "old message", conversation_id=old_id)

        newer_id = store.create_conversation()
        store.add_turn("user", "newer message", conversation_id=newer_id)

        convs = store.list_conversations()
        assert convs[0]["id"] == newer_id

    def test_delete_conversation_removes_file(self, tmp_path):
        """delete_conversation removes the JSON file."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_id = store.create_conversation()
        store.add_turn("user", "To be deleted", conversation_id=conv_id)

        json_path = sessions_dir / f"{conv_id}.json"
        assert json_path.exists()
        store.delete_conversation(conv_id)
        assert not json_path.exists()

    def test_get_conversation_all_turns(self, tmp_path):
        """get_conversation returns all turns (no limit)."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_id = store.create_conversation()
        for i in range(15):
            store.add_turn("user", f"msg {i}", conversation_id=conv_id)

        all_turns = store.get_conversation(conv_id)
        assert len(all_turns) == 15

    def test_empty_conversation_in_list(self, tmp_path):
        """New conversations with no turns appear in list."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_id = store.create_conversation()

        convs = store.list_conversations()
        found = [c for c in convs if c["id"] == conv_id]
        assert len(found) == 1
        assert found[0]["turn_count"] == 0
        assert found[0]["title"] == "New Chat"

    def test_add_turn_persists_tool_metadata(self, tmp_path):
        """Tool metadata (tool_calls, tool_results) is stored on the turn."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_id = store.create_conversation()
        store.add_turn(
            "assistant",
            "I checked the docs.",
            conversation_id=conv_id,
            metadata={
                "tool_calls": [
                    {"name": "read_knowledge_section",
                     "args": {"0": "foo.md", "1": "Intro"}},
                ],
                "tool_results": [
                    {"name": "read_knowledge_section",
                     "delivered_chars": 1234, "original_chars": 5000,
                     "truncated": True, "preview": "[SECTION: ...]"},
                ],
            },
        )
        json_path = sessions_dir / f"{conv_id}.json"
        data = json.loads(json_path.read_text())
        turn = data["turns"][0]
        assert turn["tool_calls"][0]["name"] == "read_knowledge_section"
        assert turn["tool_results"][0]["truncated"] is True
        assert turn["tool_results"][0]["delivered_chars"] == 1234


class TestHistoryWithinBudget:
    """Token-budget-aware history walk replaces fixed-N get_recent."""

    def test_returns_chronological_order(self, tmp_path):
        store = ConversationStore(sessions_dir=tmp_path / "s")
        store.initialize()
        conv_id = store.create_conversation()
        for i in range(5):
            store.add_turn("user", f"m{i}", conversation_id=conv_id)
        out = store.get_history_within_budget(conv_id, max_tokens=10000)
        assert [t["content"] for t in out] == ["m0", "m1", "m2", "m3", "m4"]

    def test_always_full_n_keeps_recent_turns_intact(self, tmp_path):
        """The last N turns are returned in full even if they have tool metadata."""
        store = ConversationStore(sessions_dir=tmp_path / "s")
        store.initialize()
        conv_id = store.create_conversation()
        store.add_turn("user", "hi", conversation_id=conv_id)
        store.add_turn(
            "assistant", "Full recent reply with details",
            conversation_id=conv_id,
            metadata={
                "tool_calls": [{"name": "search_knowledge", "args": {"0": "q"}}],
                "tool_results": [{"name": "search_knowledge",
                                  "delivered_chars": 10, "original_chars": 10,
                                  "truncated": False, "preview": ""}],
            },
        )
        out = store.get_history_within_budget(conv_id, max_tokens=100, always_full_n=2)
        assert out[-1]["content"] == "Full recent reply with details"
        assert out[0]["content"] == "hi"

    def test_old_tool_turn_compacted_to_stub(self, tmp_path):
        """Older assistant turns with tool metadata become a [earlier turn: ...] stub."""
        store = ConversationStore(sessions_dir=tmp_path / "s")
        store.initialize()
        conv_id = store.create_conversation()
        store.add_turn(
            "assistant",
            "A very long reply with raw [TOOL: brackets] that should be replaced.",
            conversation_id=conv_id,
            metadata={
                "tool_calls": [
                    {"name": "read_knowledge_section",
                     "args": {"0": "doc.md", "1": "Sec A"}},
                    {"name": "graph_stats", "args": {}},
                ],
                "tool_results": [
                    {"name": "read_knowledge_section", "delivered_chars": 100,
                     "original_chars": 100, "truncated": False, "preview": ""},
                    {"name": "graph_stats", "delivered_chars": 50,
                     "original_chars": 50, "truncated": False, "preview": ""},
                ],
            },
        )
        store.add_turn("user", "follow up", conversation_id=conv_id)
        store.add_turn("assistant", "recent", conversation_id=conv_id)

        out = store.get_history_within_budget(conv_id, max_tokens=10000, always_full_n=2)
        assert len(out) == 3
        stub = out[0]
        assert stub["role"] == "assistant"
        assert "[earlier turn:" in stub["content"]
        assert "read_knowledge_section" in stub["content"]
        assert "graph_stats" in stub["content"]
        assert "Sec A from doc.md" in stub["content"]
        assert "[TOOL:" not in stub["content"]

    def test_budget_drops_older_turns(self, tmp_path):
        """Turns older than the budget allows are dropped (recent ones kept)."""
        store = ConversationStore(sessions_dir=tmp_path / "s")
        store.initialize()
        conv_id = store.create_conversation()
        big = "word " * 500
        for i in range(6):
            store.add_turn("user", f"{i}: {big}", conversation_id=conv_id)
        out = store.get_history_within_budget(conv_id, max_tokens=200, always_full_n=2)
        assert len(out) < 6
        assert out[-1]["content"].startswith("5:")

    def test_empty_conversation_returns_empty_list(self, tmp_path):
        store = ConversationStore(sessions_dir=tmp_path / "s")
        store.initialize()
        conv_id = store.create_conversation()
        assert store.get_history_within_budget(conv_id, max_tokens=1000) == []


class TestAutoTitle:
    def test_auto_title_from_first_user_message(self, tmp_path):
        """Title auto-updates from the first user message."""
        sessions_dir = tmp_path / "sessions"
        store = ConversationStore(sessions_dir=sessions_dir)
        store.initialize()
        conv_id = store.create_conversation()
        store.add_turn(
            "user",
            "This is a very long first message that should be truncated to fit in the title",
            conversation_id=conv_id,
        )

        convs = store.list_conversations()
        found = [c for c in convs if c["id"] == conv_id]
        assert len(found) == 1
        assert found[0]["title"].startswith("This is a very long first message")
        assert len(found[0]["title"]) <= 53  # 50 chars + "..."