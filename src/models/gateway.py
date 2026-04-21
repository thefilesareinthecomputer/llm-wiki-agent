"""
Model Gateway

Unified interface for LLM access via the Ollama Python SDK.

Supports native tool calling: pass `tools=[python_fn1, python_fn2, ...]` (or
JSON-schema dicts) to ``chat_stream`` and the gateway will yield structured
tool_call events alongside thinking/content tokens. The Python SDK introspects
function type hints + Google-style docstrings to build the tool schema, so
KBTools methods are passed in directly — no separate schema file.

Reference: https://docs.ollama.com/capabilities/tool-calling
"""

import logging
import os
from typing import Any, Callable, Iterable

import httpx
import ollama

log = logging.getLogger(__name__)


# Models known to support Ollama's `tools` parameter on /api/chat.
# When a model not in this set is selected, the agent must run chat-only
# (no native tool calling). There is no homegrown bracket-parser fallback.
#
# Source: https://ollama.com/search?c=tools and
# https://ollama.com/library/glm-5.1 (per-model tool-support flag).
SUPPORTS_TOOLS_MODELS: set[str] = {
    "glm-5.1:cloud",
    "devstral-2:123b-cloud",
    "qwen3.5:397b-cloud",
    "qwen3-coder:30b",
    "qwen3:0.6b",  # used by tests/e2e
    "qwen3:1.7b",
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:14b",
    "qwen3:32b",
    "gpt-oss:120b-cloud",
    "gpt-oss:20b-cloud",
    "minimax-m2.7:cloud",
    "deepseek-v3.1:671b-cloud",
    "llama3.1",
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.2",
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.3",
    "mistral",
    "mistral-nemo",
    "mistral-small",
    "command-r-plus",
    "firefunction-v2",
}


# Preferred fallback order when the user-selected model lacks tool support.
TOOL_CAPABLE_FALLBACKS: list[str] = [
    "devstral-2:123b-cloud",
    "glm-5.1:cloud",
    "qwen3.5:397b-cloud",
    "gpt-oss:120b-cloud",
    "qwen3-coder:30b",
]


def model_supports_tools(model: str) -> bool:
    """Return True when the model is on the verified tool-capable allowlist.

    Used by the chat loop to decide whether to attach the tool schema and
    by the UI to show a "no tools" warning when an incompatible model is
    selected.
    """
    if not model:
        return False
    if model in SUPPORTS_TOOLS_MODELS:
        return True
    # Family-prefix tolerance — "qwen3:0.6b-instruct" should match "qwen3:0.6b".
    base = model.split("-")[0]
    return base in SUPPORTS_TOOLS_MODELS


class ModelGateway:
    """Gateway to Ollama-served LLMs (local or cloud).

    Wraps the official ``ollama`` Python SDK so the chat loop can pass tools
    as Python callables and receive structured ``tool_call`` events back in
    the streaming protocol.
    """

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.model = os.getenv("OLLAMA_MODEL", "devstral-2:123b-cloud")
        self._client: ollama.AsyncClient | None = None
        # Kept around for endpoints that still use bare HTTP (eg. /api/tags).
        self._http: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @property
    def client(self) -> ollama.AsyncClient:
        if self._client is None:
            self._client = ollama.AsyncClient(host=self.base_url)
        return self._client

    @property
    def _http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=60.0)
        return self._http

    async def test_connection(self) -> bool:
        """Test connection to Ollama."""
        try:
            resp = await self._http_client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            log.info(f"Connected to Ollama. Available models: {[m['name'] for m in models]}")
            return True
        except Exception as e:
            log.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            return False

    async def get_available_models(self) -> list[str]:
        """Get list of available models from Ollama."""
        try:
            resp = await self._http_client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
        except Exception as e:
            log.error(f"Failed to fetch models: {e}")
            return []

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def set_model(self, model: str) -> bool:
        """Set the current model."""
        self.model = model
        log.info(f"Model set to: {model}")
        return True

    def get_current_model(self) -> str:
        """Get the current model."""
        return self.model

    def supports_tools(self, model: str | None = None) -> bool:
        """Return True when the current (or named) model can use native tool calling."""
        return model_supports_tools(model or self.model)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    async def chat(self, messages: list[dict], stream: bool = False) -> str:
        """Send a non-tool, single-shot chat completion. Used by integration tests.

        For the production tool loop see :meth:`chat_stream`.
        """
        response = await self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
        )
        return response.message.content or ""

    async def chat_stream(
        self,
        messages: list[dict],
        tools: Iterable[Callable[..., Any] | dict] | None = None,
        think: bool = True,
    ):
        """Stream a chat completion, yielding events the tool loop can consume.

        Yields tuples of ``(event_type, payload)`` where ``event_type`` is one of:

        - ``"thinking"`` — a thinking-token chunk (str). Reasoning models only.
        - ``"content"``  — a content-token chunk (str).
        - ``"tool_call"`` — a structured tool invocation (dict with keys
          ``{"name": str, "arguments": dict}``). Emitted once per accumulated
          tool call when the model has finished thinking.

        ``tools`` is forwarded to the Ollama SDK as-is. Passing Python callables
        lets the SDK auto-generate the JSON schema from each function's type
        hints and Google-style docstring. See
        https://docs.ollama.com/capabilities/tool-calling for the contract.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": think,
        }
        if tools:
            # Materialize iterables (some tests pass generators).
            kwargs["tools"] = list(tools)

        try:
            stream = await self.client.chat(**kwargs)
        except TypeError:
            # Older ollama-python versions don't accept ``think=`` — retry without.
            kwargs.pop("think", None)
            stream = await self.client.chat(**kwargs)

        async for chunk in stream:
            msg = getattr(chunk, "message", None)
            if msg is None:
                continue

            thinking = getattr(msg, "thinking", None)
            if thinking:
                yield ("thinking", thinking)

            content = getattr(msg, "content", None)
            if content:
                yield ("content", content)

            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if fn is None:
                    continue
                name = getattr(fn, "name", "") or ""
                arguments = getattr(fn, "arguments", None) or {}
                # Ollama returns arguments as a dict already, but be defensive.
                if not isinstance(arguments, dict):
                    try:
                        import json as _json
                        arguments = _json.loads(arguments)
                    except Exception:
                        arguments = {"_raw": str(arguments)}
                if name:
                    yield ("tool_call", {"name": name, "arguments": arguments})

            if getattr(chunk, "done", False):
                break
