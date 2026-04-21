"""
Tests for model gateway.
"""

import pytest
from unittest.mock import AsyncMock, patch
from models.gateway import ModelGateway


class TestModelGateway:
    """Test ModelGateway LLM abstraction."""

    def test_init_default_config(self):
        """Test default initialization."""
        gateway = ModelGateway()
        assert "host.docker.internal" in gateway.base_url
        assert gateway.model == "devstral-2:123b-cloud"

    def test_init_custom_env(self, monkeypatch):
        """Test initialization with custom env vars."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5")

        gateway = ModelGateway()
        assert gateway.base_url == "http://custom:11434"
        assert gateway.model == "qwen2.5"

    @pytest.mark.asyncio
    async def test_chat_success(self, mock_model_gateway):
        """Test successful chat completion."""
        response = await mock_model_gateway.chat([{"role": "user", "content": "test"}])
        assert response == "Mock response"
        mock_model_gateway.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test connection test with available models."""
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.return_value = AsyncMock(
                raise_for_status=lambda: None,
                json=lambda: {"models": [{"name": "llama3.2"}]}
            )

            gateway = ModelGateway()
            result = await gateway.test_connection()
            assert result is True

    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test connection test failure."""
        with patch('httpx.AsyncClient.get') as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            gateway = ModelGateway()
            result = await gateway.test_connection()
            assert result is False
