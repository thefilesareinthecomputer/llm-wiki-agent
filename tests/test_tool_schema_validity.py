"""Regression test: every registered KB tool must produce a JSON Schema
that Ollama will accept.

Ollama's server-side validation rejects tool definitions whose parameter
``type`` field is a comma-joined string like ``"string, array"`` — which
is what the Ollama Python SDK emits when a method signature uses
``str | list[str] | None`` or other union annotations. The failure only
surfaces mid-chat (``400 Invalid tool schema: 'string, array' is not
valid under any of the given schemas``) so guard it here.

Rules enforced per parameter:
  1. ``type`` must be a string (single JSON Schema primitive) or absent
     in favor of ``anyOf`` / ``oneOf`` (which Ollama tolerates) — never
     a comma-joined string.
  2. If present, ``type`` must be one of the seven JSON Schema types.
"""
from __future__ import annotations

import pytest

from agent.tools import KBTools, build_tool_registry


VALID_JSON_SCHEMA_TYPES = {
    "string", "number", "integer", "boolean", "array", "object", "null",
}


def _convert_function_to_tool(fn):
    """Mirror of the path Ollama's SDK takes inside ``ollama.chat(tools=...)``."""
    try:
        from ollama._utils import convert_function_to_tool as _c
        return _c(fn)
    except Exception:  # pragma: no cover - SDK shape changed upstream
        from ollama._types import Tool
        return Tool(type="function", function=fn)


@pytest.fixture
def registry(tmp_path):
    class _StubKBIndex:
        graph = None

    tools = KBTools(_StubKBIndex(), tmp_path / "k", tmp_path / "c")
    return build_tool_registry(tools)


def test_every_tool_parameter_type_is_valid_json_schema(registry):
    """No tool parameter may advertise a comma-joined or unknown type."""
    offenders: list[str] = []

    for name, fn in registry.items():
        tool = _convert_function_to_tool(fn)
        dumped = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
        params = (dumped.get("function") or {}).get("parameters") or {}
        props = params.get("properties") or {}

        for pname, spec in props.items():
            t = spec.get("type")
            if isinstance(t, str):
                if "," in t:
                    offenders.append(
                        f"{name}.{pname}: type={t!r} is a comma-joined "
                        "string — Ollama will 400 on this (check for union "
                        "type hints like `str | list[str] | None`)."
                    )
                elif t not in VALID_JSON_SCHEMA_TYPES:
                    offenders.append(
                        f"{name}.{pname}: type={t!r} is not a JSON Schema "
                        f"primitive ({sorted(VALID_JSON_SCHEMA_TYPES)})."
                    )
            elif t is None and not any(k in spec for k in ("anyOf", "oneOf")):
                offenders.append(
                    f"{name}.{pname}: missing `type` and no anyOf/oneOf; "
                    "add a single type hint to the tool method signature."
                )

    assert not offenders, (
        "Ollama-incompatible tool schemas:\n  - " + "\n  - ".join(offenders)
    )


def test_save_knowledge_tags_is_plain_string(registry):
    """Guard: ``save_knowledge.tags`` must advertise as a plain string so
    the auto-generated schema is ``{"type": "string"}`` — not a union.
    Internal callers can still pass lists; ``_normalize_tags`` flattens at
    runtime. Do not re-introduce a union annotation on a tool parameter."""
    tool = _convert_function_to_tool(registry["save_knowledge"])
    dumped = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
    params = (dumped.get("function") or {}).get("parameters") or {}
    tags_spec = (params.get("properties") or {}).get("tags") or {}
    assert tags_spec.get("type") == "string", (
        f"save_knowledge.tags advertised as {tags_spec!r}; "
        "Ollama rejects non-primitive / union types in tool schemas."
    )
