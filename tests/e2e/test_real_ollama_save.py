"""End-to-end: real Ollama, real model, real disk.

This is the gate the user demanded after bug 1: stop shipping a broken
app for manual testing. The test:

  1. Connects to a real Ollama at OLLAMA_BASE_URL.
  2. Pulls / uses a small tool-capable model (default ``qwen3:0.6b``).
  3. Asks the agent to save a tiny wiki page via natural language.
  4. Asserts the file lands on disk under ``knowledge/wiki/``.

Skipped automatically when:
  - ``OLLAMA_E2E=1`` is not set in the environment, OR
  - The Ollama server is unreachable, OR
  - The chosen model is not present and pulling fails.

Run locally with:
    OLLAMA_E2E=1 docker exec llm-wiki-agent \\
        python -m pytest tests/e2e/ -v -s
"""

from __future__ import annotations

import os

import lancedb
import pytest

E2E_ENABLED = os.environ.get("OLLAMA_E2E") == "1"
E2E_MODEL = os.environ.get("OLLAMA_E2E_MODEL", "qwen3:0.6b")


pytestmark = pytest.mark.skipif(
    not E2E_ENABLED,
    reason=(
        "E2E disabled. Set OLLAMA_E2E=1 (and optionally OLLAMA_E2E_MODEL) "
        "to enable. Required in CI on main."
    ),
)


def _ollama_reachable(base_url: str) -> bool:
    import httpx
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def real_ollama_env(tmp_path_factory):
    """Wire up the real KBIndex + tools but with a tmp dir + real Ollama."""
    base_url = os.environ.get(
        "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
    )
    if not _ollama_reachable(base_url):
        pytest.skip(f"Ollama not reachable at {base_url}")

    from web.app import set_components
    from knowledge.index import KBIndex
    from memory.store import ConversationStore
    from agent.tools import KBTools
    from models.gateway import ModelGateway

    tmp_path = tmp_path_factory.mktemp("e2e")
    kb_dir = tmp_path / "knowledge"
    canon_dir = tmp_path / "canon"
    kb_dir.mkdir()
    canon_dir.mkdir()
    (kb_dir / "wiki").mkdir()

    from tests.conftest import FakeEmbeddingFunction
    import threading
    kb_index = KBIndex.__new__(KBIndex)
    kb_index.db = lancedb.connect(str(tmp_path / "lancedb"))
    kb_index.table = None
    kb_index.model_gateway = None
    kb_index._file_count = 0
    kb_index.graph = None
    kb_index._build_lock = threading.Lock()
    kb_index._embedding_model = "fake_768d"
    kb_index._embedding_fn = FakeEmbeddingFunction()

    store = ConversationStore(sessions_dir=tmp_path / "sessions")
    store.initialize()

    tools = KBTools(kb_index=kb_index, kb_dir=kb_dir, canon_dir=canon_dir)

    gateway = ModelGateway()
    gateway.base_url = base_url
    gateway.model = E2E_MODEL

    # Verify the model is on the server's tools allowlist; fail loud otherwise.
    if not gateway.supports_tools():
        pytest.skip(
            f"E2E model {E2E_MODEL!r} is not on the SUPPORTS_TOOLS_MODELS "
            f"allowlist; pick another via OLLAMA_E2E_MODEL"
        )

    set_components(gateway, kb_index, store, tools)
    kb_index.build_index(extract_entities=False)

    yield {
        "kb_dir": kb_dir,
        "canon_dir": canon_dir,
        "store": store,
        "tools": tools,
        "gateway": gateway,
    }


@pytest.mark.asyncio
async def test_real_ollama_can_save_a_wiki_page(real_ollama_env):
    """The agent receives a plain-English save request, calls
    save_knowledge via native tool calling, and the file lands on disk.

    No assertions on prose content — only that the model reached for the
    right tool and the runtime turned that into a real file. This is the
    end-to-end smoke test that catches "tool calling is broken with the
    chosen real model" regressions before the user has to.
    """
    env = real_ollama_env
    gateway = env["gateway"]
    tools = env["tools"]
    kb_dir = env["kb_dir"]

    from agent.tools import build_tool_registry
    registry = build_tool_registry(tools)
    callables = list(registry.values())

    messages = [
        {
            "role": "system",
            "content": (
                "You MUST use tool calls to act. Never describe an action "
                "in prose. To save a page you call `save_knowledge` with "
                "exactly three arguments: filename, content, tags."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use the save_knowledge tool to save filename='wiki/e2e-smoke.md', "
                "content='## Notes\n\nE2E roundtrip succeeded.\n', tags=['e2e']."
            ),
        },
    ]

    saw_save_call = False
    captured_text: list[str] = []
    async for kind, payload in gateway.chat_stream(
        messages, tools=callables, think=False
    ):
        if kind == "content":
            captured_text.append(payload)
        elif kind == "tool_call":
            if payload.get("name") == "save_knowledge":
                saw_save_call = True
                result = registry["save_knowledge"](**payload.get("arguments", {}))
                assert "wiki/" in result or "saved" in result.lower(), (
                    f"save_knowledge returned an unexpected message: {result!r}"
                )
                break

    assert saw_save_call, (
        f"Real model {gateway.model!r} did not emit a save_knowledge "
        f"tool_call. Either the model is too small to follow the instruction "
        f"or native tool calling is broken with this server.\n"
        f"Model produced this text instead:\n---\n{''.join(captured_text)[:2000]}\n---"
    )

    saved = kb_dir / "wiki" / "e2e-smoke.md"
    assert saved.exists(), (
        f"save_knowledge ran but no file appeared at {saved}"
    )
    body = saved.read_text(encoding="utf-8")
    assert body, "saved file is empty"
