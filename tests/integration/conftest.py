"""Integration test harness for the chat -> tool -> disk pipeline.

These tests differ from the unit and eval suites in two ways:

1. They wire up a *real* KBIndex (FakeEmbeddings, tmp LanceDB) plus a real
   on-disk knowledge directory under ``tmp_path``. ``save_knowledge`` writes
   actual files so tests can ``Path.read_text`` them and assert on bytes.
2. They mock only ``model_gateway.chat_stream`` — the gateway emits
   structured ``("tool_call", {"name", "arguments"})`` events that match the
   real Ollama Python SDK contract. Everything from the FastAPI tool loop
   onward (dispatch, budget enforcement, tool result framing, conversation
   persistence) runs unmodified.

This is the layer that A1's bug 1 (parser silently dropped saves) and bug 3
(forced summary swallowed writes) would have been caught at if it existed
before. Every change touching the tool loop, KB writer, or graph MUST come
with at least one scenario here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock

import lancedb
import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeEmbeddingFunction


@dataclass
class IntegrationTurn:
    """Structured record of a single /chat turn for assertions."""

    raw_sse: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    tool_done: list[dict] = field(default_factory=list)
    text_tokens: list[str] = field(default_factory=list)
    token_usage: list[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.text_tokens)

    def results_for(self, tool_name: str) -> list[dict]:
        return [r for r in self.tool_results if r.get("tool") == tool_name]

    def executed(self, tool_name: str) -> list[dict]:
        return [
            d for d in self.tool_done
            if d.get("tool") == tool_name and d.get("executed", False)
        ]


def _parse_sse(body: str) -> IntegrationTurn:
    turn = IntegrationTurn(raw_sse=body)
    for frame in body.split("\n\n"):
        event = None
        data_lines: list[str] = []
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: "):])
        data = "\n".join(data_lines)
        if not event:
            continue
        if event == "tool_call":
            try:
                turn.tool_calls.append(json.loads(data))
            except json.JSONDecodeError:
                pass
        elif event == "tool_result":
            try:
                turn.tool_results.append(json.loads(data))
            except json.JSONDecodeError:
                pass
        elif event == "tool_done":
            try:
                turn.tool_done.append(json.loads(data))
            except json.JSONDecodeError:
                pass
        elif event == "token":
            from urllib.parse import unquote
            turn.text_tokens.append(unquote(data))
        elif event == "token_usage":
            try:
                turn.token_usage.append(json.loads(data))
            except json.JSONDecodeError:
                pass
    return turn


# Helpers so each scenario reads as a clear sequence of model events.

def content(text: str) -> tuple[str, str]:
    return ("content", text)


def tool_call(name: str, **arguments) -> tuple[str, dict]:
    return ("tool_call", {"name": name, "arguments": dict(arguments)})


@pytest.fixture
def integration_env(tmp_path, monkeypatch):
    """Build a fully-wired integration environment.

    Returns a dict with:
      ``client``      — FastAPI TestClient bound to the app
      ``kb_dir``      — Path to the on-disk knowledge tier (writable)
      ``canon_dir``   — Path to the on-disk canon tier (read-only seed)
      ``store``       — ConversationStore for asserting on persistence
      ``run``         — callable(message, scripted) -> IntegrationTurn
      ``last_messages``— list[dict] of messages the gateway last received
                        (mutated each ``run`` call so tests can introspect)
    """
    from web.app import app, set_components
    from knowledge.index import KBIndex
    from memory.store import ConversationStore
    from agent.tools import KBTools
    import web.app as app_mod

    kb_dir = tmp_path / "knowledge"
    canon_dir = tmp_path / "canon"
    kb_dir.mkdir()
    canon_dir.mkdir()
    (kb_dir / "wiki").mkdir()
    (kb_dir / "raw").mkdir()

    # Steer module-level dir constants the index/tools fall back to so an
    # accidental write under /app/knowledge can never sneak through.
    import agent.tools as tools_mod
    import knowledge.index as index_mod
    monkeypatch.setattr(tools_mod, "_KNOWLEDGE_DIR", kb_dir, raising=False)
    monkeypatch.setattr(tools_mod, "_CANON_DIR", canon_dir, raising=False)
    monkeypatch.setattr(index_mod, "KB_DIR", kb_dir, raising=False)
    monkeypatch.setattr(index_mod, "CANON_DIR", canon_dir, raising=False)
    monkeypatch.setattr(index_mod, "LANCEDB_DIR", tmp_path / "lancedb", raising=False)

    kb_index = KBIndex.__new__(KBIndex)
    kb_index.db = lancedb.connect(str(tmp_path / "lancedb"))
    kb_index.table = None
    kb_index.model_gateway = None
    kb_index._file_count = 0
    kb_index.graph = None
    import threading
    kb_index._build_lock = threading.Lock()
    kb_index._embedding_model = "fake_768d"
    kb_index._embedding_fn = FakeEmbeddingFunction()

    store = ConversationStore(sessions_dir=tmp_path / "sessions")
    store.initialize()

    tools = KBTools(kb_index=kb_index, kb_dir=kb_dir, canon_dir=canon_dir)

    scripted: list = []
    last_messages: list[list[dict]] = []

    async def fake_chat_stream(messages, tools=None, think=True):
        last_messages.append(list(messages))
        if not scripted:
            return
        step = scripted.pop(0)
        if isinstance(step, str):
            if step:
                yield ("content", step)
            return
        for event in step:
            yield event

    fake_gateway = MagicMock()
    fake_gateway.chat_stream = fake_chat_stream
    fake_gateway.get_current_model = MagicMock(return_value="qwen3:0.6b")
    # Tool calling on for the loop.
    fake_gateway.supports_tools = MagicMock(return_value=True)
    fake_gateway.base_url = "http://stub"

    set_components(fake_gateway, kb_index, store, tools)
    monkeypatch.setattr(app_mod, "model_gateway", fake_gateway)

    kb_index.build_index(extract_entities=False)

    client = TestClient(app, raise_server_exceptions=True)

    def run(message: str, responses: Iterable) -> IntegrationTurn:
        scripted.clear()
        scripted.extend(list(responses))
        conv = client.post("/conversations", json={}).json()
        resp = client.post(
            "/chat",
            json={"message": message, "conversation_id": conv["id"]},
        )
        body = b"".join(resp.iter_bytes()).decode("utf-8", errors="replace")
        turn = _parse_sse(body)
        turn.conversation_id = conv["id"]  # type: ignore[attr-defined]
        return turn

    yield {
        "client": client,
        "kb_dir": kb_dir,
        "canon_dir": canon_dir,
        "store": store,
        "kb_index": kb_index,
        "tools": tools,
        "run": run,
        "last_messages": last_messages,
    }
