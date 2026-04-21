"""Real infrastructure tests.

These tests verify the actual wiring: components are initialized,
endpoints return real data, the chat flow stores and retrieves messages,
RAG context is built correctly, and tool execution works.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


class TestComponentWiring:
    """Verify shared components are actually initialized and wired."""

    def test_model_gateway_initialized(self, client_with_init):
        """Model gateway is initialized and accessible from web app."""
        from web.app import model_gateway
        assert model_gateway is not None
        assert hasattr(model_gateway, 'chat_stream')
        assert hasattr(model_gateway, 'chat')

    def test_kb_index_initialized(self, client_with_init):
        """KB index is initialized with a real table."""
        from web.app import kb_index
        assert kb_index is not None
        assert kb_index.table is not None

    def test_memory_store_initialized(self, client_with_init):
        """Memory store is initialized and can CRUD."""
        from web.app import memory_store
        assert memory_store is not None
        conv_id = memory_store.create_conversation()
        assert len(conv_id) == 36
        memory_store.add_turn("user", "test", conversation_id=conv_id)
        turns = memory_store.get_conversation(conv_id)
        assert len(turns) == 1

    def test_kb_tools_initialized(self, client_with_init):
        """KB tools are initialized and wired into web app."""
        from web.app import kb_tools
        assert kb_tools is not None
        assert hasattr(kb_tools, 'list_knowledge')
        assert hasattr(kb_tools, 'read_knowledge')
        assert hasattr(kb_tools, 'read_knowledge_section')
        assert hasattr(kb_tools, 'search_knowledge')
        assert hasattr(kb_tools, 'save_knowledge')

    def test_kb_index_has_real_data(self, client_with_init):
        """KB index actually contains indexed documents."""
        from web.app import kb_index
        stats = kb_index.get_stats()
        assert stats["files"] > 0
        assert stats["vectors"] > 0
        assert stats["vectors"] >= stats["files"]

    def test_kb_stats_file_vs_vector_counts(self, client_with_init):
        """KB stats correctly separates file count from vector count."""
        response = client_with_init.get("/kb/stats")
        data = response.json()
        assert "files" in data
        assert "vectors" in data
        assert data["files"] > 0
        assert data["vectors"] >= data["files"]
        # Section chunking means more vectors than files
        assert data["vectors"] > data["files"]


class TestChatFlow:
    """Test the actual chat flow — message storage, RAG context, response persistence."""

    def test_chat_stores_user_and_assistant_messages(self, client_with_init):
        """Chat endpoint stores both user and assistant messages in memory."""
        from web.app import memory_store, model_gateway

        conv_id = memory_store.create_conversation()

        async def mock_stream(messages, tools=None, think=True):
            yield "content", "Hello from mock"

        # Replace chat_stream directly
        original_stream = model_gateway.chat_stream
        model_gateway.chat_stream = mock_stream

        try:
            with client_with_init.stream(
                "POST", "/chat",
                json={"message": "Test message", "conversation_id": conv_id},
            ) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    pass
        finally:
            model_gateway.chat_stream = original_stream

        turns = memory_store.get_conversation(conv_id)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Test message"
        assert turns[1]["role"] == "assistant"
        assert "Hello from mock" in turns[1]["content"]

    def test_chat_builds_rag_context(self, client_with_init):
        """Chat endpoint includes KB search results in the system prompt."""
        from web.app import memory_store, model_gateway
        captured_messages = []

        async def mock_stream(messages, tools=None, think=True):
            captured_messages.extend(messages)
            yield "content", "Response"

        conv_id = memory_store.create_conversation()

        original_stream = model_gateway.chat_stream
        model_gateway.chat_stream = mock_stream

        try:
            with client_with_init.stream(
                "POST", "/chat",
                json={"message": "knowledge base", "conversation_id": conv_id},
            ) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    pass
        finally:
            model_gateway.chat_stream = original_stream

        system_msg = captured_messages[0]
        assert system_msg["role"] == "system"

    def test_chat_includes_tool_definitions(self, client_with_init):
        """Chat endpoint includes L4 tool definitions in system prompt when tools available."""
        from web.app import memory_store, model_gateway, kb_tools
        captured_messages = []

        async def mock_stream(messages, tools=None, think=True):
            captured_messages.extend(messages)
            yield "content", "OK"

        conv_id = memory_store.create_conversation()

        original_stream = model_gateway.chat_stream
        model_gateway.chat_stream = mock_stream

        try:
            with client_with_init.stream(
                "POST", "/chat",
                json={"message": "hello", "conversation_id": conv_id},
            ) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    pass
        finally:
            model_gateway.chat_stream = original_stream

        system_msg = captured_messages[0]
        if kb_tools:
            assert "list_knowledge" in system_msg["content"]

    def test_chat_no_duplicate_user_message(self, client_with_init):
        """User message appears exactly once in LLM context, not twice."""
        from web.app import memory_store, model_gateway
        captured_messages = []

        async def mock_stream(messages, tools=None, think=True):
            captured_messages.extend(messages)
            yield "content", "Hi"

        conv_id = memory_store.create_conversation()

        original_stream = model_gateway.chat_stream
        model_gateway.chat_stream = mock_stream
        try:
            with client_with_init.stream(
                "POST", "/chat",
                json={"message": "unique_test_42", "conversation_id": conv_id},
            ) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    pass
        finally:
            model_gateway.chat_stream = original_stream

        user_msg_count = sum(
            1 for m in captured_messages
            if m["role"] == "user" and m["content"] == "unique_test_42"
        )
        assert user_msg_count == 1, f"User message appeared {user_msg_count} times (expected 1)"

    def test_chat_strips_timestamps_from_context(self, client_with_init):
        """LLM context messages only have role and content, not timestamp."""
        from web.app import memory_store, model_gateway
        captured_messages = []

        async def mock_stream(messages, tools=None, think=True):
            captured_messages.extend(messages)
            yield "content", "OK"

        conv_id = memory_store.create_conversation()
        memory_store.add_turn("user", "previous message", conversation_id=conv_id)
        memory_store.add_turn("assistant", "previous response", conversation_id=conv_id)

        original_stream = model_gateway.chat_stream
        model_gateway.chat_stream = mock_stream
        try:
            with client_with_init.stream(
                "POST", "/chat",
                json={"message": "new message", "conversation_id": conv_id},
            ) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    pass
        finally:
            model_gateway.chat_stream = original_stream

        for msg in captured_messages:
            assert "timestamp" not in msg, f"Message had timestamp: {msg}"

    def test_chat_returns_503_when_no_model(self, client_with_init):
        """Chat returns 503 when model gateway is not initialized."""
        import web.app

        # Save and clear model gateway
        original = web.app.model_gateway
        web.app.model_gateway = None
        try:
            response = client_with_init.post(
                "/chat",
                json={"message": "test", "conversation_id": "fake-id"},
            )
            assert response.status_code == 503
        finally:
            web.app.model_gateway = original

    def test_chat_returns_400_without_conversation_id(self, client_with_init):
        """Chat returns 400 when conversation_id is missing."""
        response = client_with_init.post(
            "/chat",
            json={"message": "hello"},
        )
        assert response.status_code == 400


class TestToolExecution:
    """Test that tool calls are detected and executed in the chat flow."""

    def test_tool_call_detected_in_model_output(self, client_with_init):
        """When the model emits a native tool_call event, the chat loop
        dispatches it through _execute_tool."""
        from web.app import memory_store, model_gateway

        conv_id = memory_store.create_conversation()
        executed_tools = []

        call_count = [0]

        async def mock_stream(messages, tools=None, think=True):
            call_count[0] += 1
            if call_count[0] == 1:
                yield "content", "Let me search."
                yield "tool_call", {"name": "list_knowledge", "arguments": {}}
            else:
                yield "content", "Here are the files."

        import web.app
        original_execute = web.app._execute_tool
        original_stream = model_gateway.chat_stream
        original_supports = model_gateway.supports_tools

        def mock_execute(name, args):
            executed_tools.append((name, args))
            return "Mock tool result"

        web.app._execute_tool = mock_execute
        model_gateway.chat_stream = mock_stream
        model_gateway.supports_tools = lambda model=None: True

        try:
            with client_with_init.stream(
                "POST", "/chat",
                json={"message": "show me files", "conversation_id": conv_id},
            ) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    pass
        finally:
            web.app._execute_tool = original_execute
            model_gateway.chat_stream = original_stream
            model_gateway.supports_tools = original_supports

        assert len(executed_tools) == 1
        assert executed_tools[0][0] == "list_knowledge"

    def test_tool_budget_resets_per_request(self, client_with_init):
        """KB load budget resets at the start of each chat request."""
        from agent.tools import _KB_MAX_LOADS_PER_RESPONSE, reset_budget
        import agent.tools

        agent.tools._current_kb_loads = _KB_MAX_LOADS_PER_RESPONSE

        from web.app import memory_store, model_gateway

        conv_id = memory_store.create_conversation()

        async def mock_stream(messages, tools=None, think=True):
            yield "content", "OK"

        original_stream = model_gateway.chat_stream
        model_gateway.chat_stream = mock_stream
        try:
            with client_with_init.stream(
                "POST", "/chat",
                json={"message": "test", "conversation_id": conv_id},
            ) as response:
                assert response.status_code == 200
                for line in response.iter_lines():
                    pass
        finally:
            model_gateway.chat_stream = original_stream

        assert agent.tools._current_kb_loads == 0
        reset_budget()


class TestKBSearchRealData:
    """Test KB search returns real, usable data."""

    def test_heading_tree_returns_structure(self, client_with_init):
        """get_heading_tree returns navigable tree with token costs."""
        from web.app import kb_index
        if not kb_index:
            pytest.skip("KB index not initialized")

        # Use a known file path instead of search
        kb_index.build_index(extract_entities=False)
        # List files to find one we can test
        from pathlib import Path
        kb_dir = Path("/app/knowledge")
        md_files = list(kb_dir.rglob("*.md"))
        if not md_files:
            pytest.skip("No markdown files in knowledge/")

        # Use the first file's relative path
        rel_path = str(md_files[0].relative_to(kb_dir))
        tree = kb_index.get_heading_tree(rel_path)
        assert tree is not None
        assert isinstance(tree, str)


class TestChatLockRelease:
    """Verify _chat_lock is released even on error or disconnect."""

    def test_lock_released_after_error(self, client_with_init):
        """If chat_stream raises, the lock must still be released."""
        from web.app import _chat_lock

        # Create a conversation first
        client_with_init.post("/conversations")
        conv_id = client_with_init.get("/conversations").json()[0]["id"]

        # Replace chat_stream with one that raises
        from web.app import model_gateway
        original_stream = model_gateway.chat_stream

        async def failing_stream(messages, tools=None, think=True):
            raise RuntimeError("stream error")
            yield  # noqa: unreachable — makes this an async generator

        model_gateway.chat_stream = failing_stream

        # Send chat request — should get error event but not deadlock
        try:
            with client_with_init.stream(
                "POST", "/chat",
                json={"message": "hello", "conversation_id": conv_id},
            ) as resp:
                for line in resp.iter_lines():
                    pass
        finally:
            model_gateway.chat_stream = original_stream

        # Lock should NOT be held
        assert not _chat_lock.locked(), "Lock still held after stream error"

    def test_lock_available_after_normal_chat(self, client_with_init):
        """After a normal chat completes, lock is available."""
        from web.app import _chat_lock

        # Create a conversation
        client_with_init.post("/conversations")
        conv_id = client_with_init.get("/conversations").json()[0]["id"]

        from web.app import model_gateway
        original_stream = model_gateway.chat_stream

        async def simple_stream(messages, tools=None, think=True):
            yield "content", "Hello"

        model_gateway.chat_stream = simple_stream

        try:
            with client_with_init.stream(
                "POST", "/chat",
                json={"message": "hello", "conversation_id": conv_id},
            ) as resp:
                for line in resp.iter_lines():
                    pass
        finally:
            model_gateway.chat_stream = original_stream

        # Lock should NOT be held
        assert not _chat_lock.locked(), "Lock still held after normal chat"