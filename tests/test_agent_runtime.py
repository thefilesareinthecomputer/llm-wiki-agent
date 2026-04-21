"""
Tests for agent runtime.
"""

import pytest
import asyncio
from pathlib import Path
from agent.runtime import AgentRuntime


class TestAgentRuntime:
    """Test agent runtime component."""

    def test_runtime_init(self, mock_model_gateway, temp_kb_dir):
        """Test runtime initialization."""
        kb_dir, canon_dir = temp_kb_dir
        runtime = AgentRuntime(mock_model_gateway, kb_dir, canon_dir)

        assert runtime.model == mock_model_gateway
        assert runtime.kb_dir == kb_dir
        assert runtime.canon_dir == canon_dir
        assert runtime.running is False

    @pytest.mark.asyncio
    async def test_runtime_start_stop(self, mock_model_gateway, temp_kb_dir):
        """Test runtime start and stop."""
        kb_dir, canon_dir = temp_kb_dir
        runtime = AgentRuntime(mock_model_gateway, kb_dir, canon_dir)

        # Start runtime in background task
        task = asyncio.create_task(runtime.run())

        await asyncio.sleep(0.1)  # Let it run briefly
        assert runtime.running is True

        runtime.stop()
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_runtime_with_memory(self, mock_model_gateway, temp_kb_dir):
        """Test runtime with memory store."""
        from memory.store import ConversationStore
        from pathlib import Path

        kb_dir, canon_dir = temp_kb_dir
        memory = ConversationStore(sessions_dir=Path("/tmp/test_sessions"))

        runtime = AgentRuntime(mock_model_gateway, kb_dir, canon_dir, memory)
        assert runtime.memory == memory
