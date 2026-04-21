"""Tests for the tool-loop dedup guard in the chat endpoint (post-A1).

After the A1 native-tool-calling refactor, the dedup guard works on the
streamed ``tool_call`` events instead of inline ``[TOOL: ...]`` text.
Three layers are still verified:

1. ``_tool_signature`` produces stable, order-independent signatures.
2. Within a single iteration, identical calls collapse to one execution.
3. Across iterations, an already-executed (name, args) is short-circuited
   and the model gets a SKIPPED REPEAT note instead of a redundant
   tool result.
"""

import json
from unittest.mock import MagicMock

import pytest


# ---------- Pure helper: _tool_signature ----------

class TestToolSignature:
    def test_identical_args_same_signature(self):
        from web.app import _tool_signature
        s1 = _tool_signature("search_knowledge", {"query": "daily log"})
        s2 = _tool_signature("search_knowledge", {"query": "daily log"})
        assert s1 == s2

    def test_arg_order_does_not_matter(self):
        from web.app import _tool_signature
        s1 = _tool_signature("graph_neighbors", {"filename": "a.md", "heading": "x"})
        s2 = _tool_signature("graph_neighbors", {"heading": "x", "filename": "a.md"})
        assert s1 == s2, "dict order must not change the signature"

    def test_different_args_different_signature(self):
        from web.app import _tool_signature
        s1 = _tool_signature("search_knowledge", {"query": "daily log"})
        s2 = _tool_signature("search_knowledge", {"query": "philosophy"})
        assert s1 != s2

    def test_different_tools_different_signature(self):
        from web.app import _tool_signature
        s1 = _tool_signature("search_knowledge", {"query": "x"})
        s2 = _tool_signature("graph_search", {"query": "x"})
        assert s1 != s2

    def test_empty_args(self):
        from web.app import _tool_signature
        assert _tool_signature("graph_stats", {}) == _tool_signature("graph_stats", {})


# ---------- End-to-end: chat endpoint dedup ----------

@pytest.fixture
def chat_with_stub_gateway(client_with_init, monkeypatch):
    """Wrap client_with_init so the model gateway is replaced by a stub
    whose chat_stream emits a scripted sequence of native events per call.

    Yields ``(client, set_script)`` where each script entry is a list of
    ``(kind, payload)`` tuples (see ``ModelGateway.chat_stream``)."""
    import web.app as app_mod

    scripted: list = []

    async def _fake_chat_stream(messages, tools=None, think=True):
        if not scripted:
            return
        events = scripted.pop(0)
        for ev in events:
            yield ev

    fake_gateway = MagicMock()
    fake_gateway.chat_stream = _fake_chat_stream
    fake_gateway.get_current_model = MagicMock(return_value="stub-model")
    fake_gateway.supports_tools = MagicMock(return_value=True)
    fake_gateway.base_url = "http://stub"

    monkeypatch.setattr(app_mod, "model_gateway", fake_gateway)

    def set_script(responses: list):
        scripted.clear()
        scripted.extend(responses)

    return client_with_init, set_script


def _read_sse(response) -> str:
    chunks = []
    for chunk in response.iter_bytes():
        chunks.append(chunk.decode("utf-8", errors="replace"))
    return "".join(chunks)


def _count_tool_done_executed(body: str) -> int:
    """How many tools actually ran (``tool_done`` with executed=true).

    Do not count raw ``\"executed\": true`` — ``tool_result`` also carries
    ``executed`` for the UI and would double-count.
    """
    n = 0
    pos = 0
    while True:
        ev = body.find("event: tool_done", pos)
        if ev < 0:
            break
        data_mark = body.find("data: ", ev)
        if data_mark < 0:
            break
        line_start = data_mark + len("data: ")
        line_end = body.find("\n", line_start)
        if line_end < 0:
            break
        try:
            payload = json.loads(body[line_start:line_end])
            if payload.get("executed") is True:
                n += 1
        except json.JSONDecodeError:
            pass
        pos = line_end
    return n


def _tc(name: str, **arguments):
    return ("tool_call", {"name": name, "arguments": dict(arguments)})


def _content(text: str):
    return ("content", text)


class TestChatDedup:
    def test_same_call_in_one_response_runs_once(self, chat_with_stub_gateway):
        """If the model emits the same tool_call three times in one
        iteration, the chat loop must execute it ONCE."""
        client, set_script = chat_with_stub_gateway

        set_script([
            [_tc("list_knowledge"), _tc("list_knowledge"), _tc("list_knowledge")],
            [_content("Final answer based on the result.")],
        ])

        conv = client.post("/conversations", json={}).json()
        resp = client.post("/chat", json={
            "message": "list files",
            "conversation_id": conv["id"],
        })
        body = _read_sse(resp)

        # Each tool_call still surfaces a tool_executing event (with
        # executed=False for the dupes), so we count actual executions
        # using the tool_done event payload instead.
        executed_done_events = _count_tool_done_executed(body)
        assert executed_done_events == 1, (
            f"Expected within-iteration dedup to collapse 3 identical calls "
            f"to 1 execution; saw {executed_done_events}.\n"
            f"Body preview:\n{body[:600]}"
        )

    def test_repeat_across_iterations_is_blocked(self, chat_with_stub_gateway):
        """If the model emits the same call in two separate iterations,
        the second one must NOT re-execute - the dedup guard converts it
        to a SKIPPED REPEAT result."""
        client, set_script = chat_with_stub_gateway

        set_script([
            [_tc("list_knowledge")],
            [_tc("list_knowledge")],
            [_content("I should never reach here.")],
        ])

        conv = client.post("/conversations", json={}).json()
        resp = client.post("/chat", json={
            "message": "loopy",
            "conversation_id": conv["id"],
        })
        body = _read_sse(resp)

        executed_done_events = _count_tool_done_executed(body)
        assert executed_done_events == 1, (
            f"Repeated (name,args) across iterations must be skipped; "
            f"saw {executed_done_events} executions.\n"
            f"Body preview:\n{body[:600]}"
        )

    def test_different_args_still_run(self, chat_with_stub_gateway):
        """Dedup is keyed on (name, args). Same tool with different args
        must still execute - we are not blocking the tool, just blocking
        identical re-runs."""
        client, set_script = chat_with_stub_gateway

        set_script([
            [_tc("search_knowledge", query="alpha"),
             _tc("search_knowledge", query="beta")],
            [_content("Done.")],
        ])

        conv = client.post("/conversations", json={}).json()
        resp = client.post("/chat", json={
            "message": "two queries",
            "conversation_id": conv["id"],
        })
        body = _read_sse(resp)

        executed_done_events = _count_tool_done_executed(body)
        assert executed_done_events == 2, (
            f"Different args must each execute; saw {executed_done_events}.\n"
            f"Body preview:\n{body[:600]}"
        )
