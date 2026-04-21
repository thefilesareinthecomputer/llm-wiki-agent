"""B3 — compile_knowledge from conversations + ## Sources validation.

Covers:

1. ``compile_knowledge(source_type="conversation", source_ref=...)`` accepts
   the documented selector grammar (``<id>``, ``<id>:last:N``,
   ``<id>:turn:N``, ``<id>:turn:A-B``) and returns a compilation prompt
   built from the right turns.
2. Malformed selectors and unknown conversation ids return a clear error
   instead of silently compiling from nothing.
3. ``save_knowledge`` validates ``## Sources`` lines that look like
   ``conversation:<id>:turn:N``. A bad citation aborts the save; a
   verifiable one writes through.
4. Validation is skipped when no ConversationStore is wired (so existing
   tests + non-chat callers keep working unchanged).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import FakeEmbeddingFunction


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Build KBTools with a wired ConversationStore + KB on disk.

    Uses the same monkeypatch pattern as test_save_compilation.py — KBIndex
    reads its KB_DIR/CANON_DIR/LANCEDB_DIR from module globals, so we
    redirect those at tmp_path before instantiating.
    """
    import lancedb
    import knowledge.index as kbi
    import agent.tools as agent_tools
    from agent.tools import KBTools
    from knowledge.index import KBIndex
    from memory.store import ConversationStore

    kb_dir = tmp_path / "knowledge"
    canon_dir = tmp_path / "canon"
    sessions_dir = tmp_path / "sessions"
    (kb_dir / "wiki").mkdir(parents=True, exist_ok=True)
    (kb_dir / "raw").mkdir(parents=True, exist_ok=True)
    canon_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(kbi, "KB_DIR", kb_dir)
    monkeypatch.setattr(kbi, "CANON_DIR", canon_dir)
    monkeypatch.setattr(kbi, "LANCEDB_DIR", tmp_path / "lancedb")
    monkeypatch.setattr(agent_tools, "_KNOWLEDGE_DIR", kb_dir, raising=False)
    monkeypatch.setattr(agent_tools, "_CANON_DIR", canon_dir, raising=False)

    (kb_dir / "wiki" / "kimball.md").write_text(
        "---\ntags: dimensional-modeling\n---\n# Kimball\n\n"
        "## Star Schemas\nKimball star schemas are the workhorse.\n"
    )
    (kb_dir / "raw" / "notes.md").write_text(
        "# Raw Notes\n\nSome raw text about data warehouses.\n"
    )

    kb_index = KBIndex()
    kb_index.db = lancedb.connect(str(tmp_path / "lancedb"))
    kb_index._embedding_fn = FakeEmbeddingFunction()
    kb_index.build_index(extract_entities=False, llm_summaries=False, force=True)

    store = ConversationStore(sessions_dir=sessions_dir)
    store.initialize()
    conv_id = store.create_conversation()
    store.add_turn("user", "What is SCD Type 2?", conversation_id=conv_id)
    store.add_turn(
        "assistant",
        "Slowly Changing Dimensions Type 2 add a new row when an attribute "
        "changes, preserving full history with effective-from / effective-to "
        "columns.",
        conversation_id=conv_id,
    )
    store.add_turn("user", "Anything else I should know?", conversation_id=conv_id)
    store.add_turn(
        "assistant",
        "Pair SCD2 with surrogate keys and a current_flag column for fast "
        "point-in-time queries.",
        conversation_id=conv_id,
    )

    tools = KBTools(
        kb_index=kb_index,
        kb_dir=kb_dir,
        canon_dir=canon_dir,
        conversation_store=store,
    )
    return {
        "tools": tools,
        "store": store,
        "conv_id": conv_id,
        "kb_dir": kb_dir,
    }


# ----- compile_knowledge from conversation ----------------------------------

def test_compile_from_conversation_whole_thread(env):
    out = env["tools"].compile_knowledge(
        source_type="conversation",
        source_ref=env["conv_id"],
    )
    assert "Compile Knowledge from conversation" in out
    assert env["conv_id"] in out
    assert "[tier=memory]" in out
    assert "turn[0]" in out
    assert "turn[3]" in out
    # Plan should teach the model the right citation format.
    assert f"conversation:{env['conv_id']}:turn:0" in out


def test_compile_from_conversation_last_n(env):
    out = env["tools"].compile_knowledge(
        source_type="conversation",
        source_ref=f"{env['conv_id']}:last:2",
    )
    assert "turn[2]" in out
    assert "turn[3]" in out
    assert "turn[0]" not in out


def test_compile_from_conversation_turn_range(env):
    out = env["tools"].compile_knowledge(
        source_type="conversation",
        source_ref=f"{env['conv_id']}:turn:1-2",
    )
    assert "turn[1]" in out
    assert "turn[2]" in out
    assert "turn[3]" not in out


def test_compile_from_conversation_single_turn(env):
    out = env["tools"].compile_knowledge(
        source_type="conversation",
        source_ref=f"{env['conv_id']}:turn:1",
    )
    assert "turn[1]" in out
    assert "turn[0]" not in out
    assert "turn[2]" not in out


def test_compile_rejects_unknown_conversation(env):
    out = env["tools"].compile_knowledge(
        source_type="conversation",
        source_ref="not-a-real-conversation",
    )
    assert "no turns" in out


def test_compile_rejects_malformed_selector(env):
    out = env["tools"].compile_knowledge(
        source_type="conversation",
        source_ref=f"{env['conv_id']}:turn:bad",
    )
    assert "Invalid turn selector" in out


def test_compile_rejects_missing_source_ref(env):
    out = env["tools"].compile_knowledge(
        source_type="conversation",
        source_ref="",
    )
    assert "source_ref is required" in out


def test_compile_rejects_unknown_source_type(env):
    out = env["tools"].compile_knowledge(
        source_type="garbage",
        source_ref=env["conv_id"],
    )
    assert "Invalid source_type" in out


def test_compile_file_path_still_works(env):
    """File-source path must remain backward compatible."""
    out = env["tools"].compile_knowledge(source="raw/notes.md")
    assert "Compile Knowledge from: raw/notes.md" in out


# ----- save_knowledge sources validation ------------------------------------

def _body_with_sources(conv_id: str, sources_lines: list[str]) -> str:
    return (
        "## Body\n\nSome content.\n\n"
        "## Sources\n"
        + "\n".join(sources_lines)
        + "\n"
    )


def test_save_knowledge_accepts_verifiable_conversation_citation(env):
    body = _body_with_sources(
        env["conv_id"],
        [
            f"- conversation:{env['conv_id']}:turn:0",
            f"- conversation:{env['conv_id']}:turn:1-2",
        ],
    )
    result = env["tools"].save_knowledge("scd-type-2.md", body, tags="dw")
    assert result.startswith("Saved:")
    saved = (env["kb_dir"] / "wiki" / "scd-type-2.md").read_text()
    assert "Sources" in saved


def test_save_knowledge_refuses_unknown_conversation(env):
    body = _body_with_sources(
        env["conv_id"],
        [
            "- conversation:totally-fake-id:turn:0",
        ],
    )
    result = env["tools"].save_knowledge("bad-cite.md", body, tags="dw")
    assert "refusing to write" in result
    assert "totally-fake-id" in result
    assert not (env["kb_dir"] / "wiki" / "bad-cite.md").exists()


def test_save_knowledge_refuses_out_of_range_turn(env):
    # The fixture conversation has 4 turns (0..3); 99 must trip validation.
    body = _body_with_sources(
        env["conv_id"],
        [
            f"- conversation:{env['conv_id']}:turn:99",
        ],
    )
    result = env["tools"].save_knowledge("oor.md", body, tags="dw")
    assert "refusing to write" in result
    assert "out of range" in result
    assert not (env["kb_dir"] / "wiki" / "oor.md").exists()


def test_save_knowledge_refuses_inverted_range(env):
    body = _body_with_sources(
        env["conv_id"],
        [
            f"- conversation:{env['conv_id']}:turn:3-1",
        ],
    )
    result = env["tools"].save_knowledge("inv.md", body, tags="dw")
    assert "refusing to write" in result
    assert "B >= A" in result


def test_save_knowledge_ignores_non_conversation_sources(env):
    """Non-conversation Source lines must not trigger validation.

    R12 note: we swap the previously-used `https://example.com/article`
    for a real-looking URL. The placeholder-URL validator (R12) catches
    `example.com` specifically, and this test is about the CONVERSATION-
    citation validator, not the URL validator — using a real URL keeps
    the test scope clean.
    """
    body = (
        "## Body\n\nfoo\n\n"
        "## Sources\n"
        "- canon:mind-en-place/vault.md\n"
        "- knowledge:wiki/kimball.md\n"
        "- https://en.wikipedia.org/wiki/Data_warehouse\n"
    )
    result = env["tools"].save_knowledge("plain.md", body, tags="dw")
    assert result.startswith("Saved:")


def test_save_knowledge_skips_validation_when_store_unwired(tmp_path):
    """Without a ConversationStore the validator can't verify; it must
    not block the save (otherwise non-chat callers break)."""
    from agent.tools import KBTools

    kb_dir = tmp_path / "knowledge"
    canon_dir = tmp_path / "canon"
    (kb_dir / "wiki").mkdir(parents=True, exist_ok=True)
    canon_dir.mkdir(parents=True, exist_ok=True)

    tools = KBTools(
        kb_index=None,
        kb_dir=kb_dir,
        canon_dir=canon_dir,
        conversation_store=None,
    )
    body = (
        "## Body\n\nfoo\n\n"
        "## Sources\n"
        "- conversation:made-up:turn:0\n"
    )
    result = tools.save_knowledge("nostore.md", body, tags="x")
    assert result.startswith("Saved:")


# ----- Selector parser unit tests -------------------------------------------

@pytest.mark.parametrize("ref,expected", [
    ("abc", ("abc", "")),
    ("abc:last:5", ("abc", "last:5")),
    ("abc:turn:7", ("abc", "range:7:7")),
    ("abc:turn:0-3", ("abc", "range:0:3")),
])
def test_parse_conversation_ref_happy(ref, expected):
    from agent.tools import KBTools

    conv_id, spec, err = KBTools._parse_conversation_ref(ref)
    assert err == ""
    assert (conv_id, spec) == expected


@pytest.mark.parametrize("ref,fragment", [
    ("", "source_ref is required"),
    (":turn:1", "missing conversation id"),
    ("abc:last:0", "N >= 1"),
    ("abc:last:bad", "Invalid last:N"),
    ("abc:turn:foo", "Invalid turn selector"),
    ("abc:turn:5-1", "B >= A"),
    ("abc:weird:1", "Invalid source_ref"),
])
def test_parse_conversation_ref_errors(ref, fragment):
    from agent.tools import KBTools

    _, _, err = KBTools._parse_conversation_ref(ref)
    assert fragment in err
