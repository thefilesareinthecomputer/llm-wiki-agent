"""Eval harness — pytest-based behavioral evals for the agent.

These evals are NOT model-quality benchmarks. They verify that the
prompt scaffolding, tool loop, and tool-result framing behave the way
the agent prompts say they should, GIVEN a model that follows the
script. We mock ``model_gateway.chat_stream`` with a scripted sequence
per iteration; the rest of the stack (KB index, graph, tools, system
prompt assembly) runs unmodified.

A1 migrated the runtime to native Ollama tool calling. The harness
mirrors that protocol: each scripted iteration is a list of
``(kind, payload)`` tuples drawn from the same event vocabulary that
``ModelGateway.chat_stream`` yields:

  ("content", "...")
  ("thinking", "...")
  ("tool_call", {"name": "...", "arguments": {...}})

Helpers ``content(...)`` and ``tool_call(name, **kwargs)`` let scenarios
spell that out tersely. A bare string is still accepted as a shortcut
for a single content event.

A single helper, ``EvalTurn``, parses the SSE stream from POST /chat
into a structured turn record so each scenario can assert on:
  - which tools were executed
  - tool args
  - tool results delivered to the model
  - final assistant text shown to the user

To add a new scenario: drop a ``test_eval_*.py`` file next to this one
and use the ``eval_run`` fixture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable
from unittest.mock import MagicMock

import pytest


@dataclass
class EvalTurn:
    """Structured record of a single /chat turn."""

    raw_sse: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    text_tokens: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.text_tokens)


def _parse_sse(body: str) -> EvalTurn:
    """Parse an SSE response body into an EvalTurn."""
    turn = EvalTurn(raw_sse=body)
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
        elif event == "token":
            from urllib.parse import unquote
            turn.text_tokens.append(unquote(data))
    return turn


# --- Scripting helpers ----------------------------------------------------

def content(text: str) -> tuple[str, str]:
    """Script a content token chunk."""
    return ("content", text)


def thinking(text: str) -> tuple[str, str]:
    """Script a thinking token chunk."""
    return ("thinking", text)


def tool_call(name: str, **arguments) -> tuple[str, dict]:
    """Script a native tool_call event.

    Native tool calling means the model emits ``{"name": ..., "arguments": {...}}``
    instead of inline ``[TOOL: ...]`` brackets. Use this helper in
    scripted_responses lists so each step reads as a clear sequence of
    events.
    """
    return ("tool_call", {"name": name, "arguments": dict(arguments)})


@pytest.fixture
def eval_run(client_with_init, monkeypatch):
    """Returns a callable: ``eval_run(message, scripted_responses) -> EvalTurn``.

    ``scripted_responses`` is a list of "iterations". Each iteration is one
    of:

    - ``str`` — single content chunk shortcut.
    - ``list[tuple[str, Any]]`` — full sequence of events for that
      iteration. Use ``content("...")`` and ``tool_call("name", ...)`` to
      build entries.

    Past the scripted length the model "says nothing" (empty iteration).
    """
    import web.app as app_mod

    scripted: list = []

    async def _fake_chat_stream(messages, tools=None, think=True):
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
    fake_gateway.chat_stream = _fake_chat_stream
    fake_gateway.get_current_model = MagicMock(return_value="stub-model")
    # The stub model isn't on the SUPPORTS_TOOLS_MODELS allowlist, but we
    # want the loop to attach tools so we can exercise tool_call events.
    fake_gateway.supports_tools = MagicMock(return_value=True)
    fake_gateway.base_url = "http://stub"

    monkeypatch.setattr(app_mod, "model_gateway", fake_gateway)

    def run(message: str, responses: Iterable) -> EvalTurn:
        scripted.clear()
        scripted.extend(list(responses))
        conv = client_with_init.post("/conversations", json={}).json()
        resp = client_with_init.post(
            "/chat",
            json={"message": message, "conversation_id": conv["id"]},
        )
        body = b"".join(resp.iter_bytes()).decode("utf-8", errors="replace")
        return _parse_sse(body)

    return run
