"""Tests for P3.2 — compile_knowledge tool.

The tool stages a new wiki page from a raw/ source file. It must:
  - Resolve the source via the standard filename waterfall
  - Surface tier badge + a tier-specific note
  - Search for related wiki + canon material and bucket by tier
  - Suggest a wiki/<slug>.md output filename derived from the source
  - Render a step-by-step compilation plan (read → synthesize → link → cite → save)
  - NEVER write anything (read-only planning tool)
"""

import pytest


@pytest.fixture
def tools_with_kb(tmp_path, monkeypatch):
    """Real KBIndex with a small mixed-tier corpus: raw source, two wiki
    pages, and one canon anchor — enough for compile_knowledge to exercise
    every output section."""
    import lancedb
    import knowledge.index as kbi
    from knowledge.index import KBIndex
    from agent.tools import KBTools
    from tests.conftest import FakeEmbeddingFunction

    kb = tmp_path / "knowledge"
    canon = tmp_path / "canon"
    (kb / "wiki").mkdir(parents=True)
    (kb / "raw").mkdir(parents=True)
    canon.mkdir()
    monkeypatch.setattr(kbi, "KB_DIR", kb)
    monkeypatch.setattr(kbi, "CANON_DIR", canon)
    monkeypatch.setattr(kbi, "LANCEDB_DIR", tmp_path / "lancedb")

    (kb / "raw" / "stress-transcript.md").write_text(
        "# stress transcript\n\n"
        "Notes from a session about cortisol release, the HPA axis, "
        "and the body's stress response.\n"
    )
    (kb / "wiki" / "cortisol.md").write_text(
        "# cortisol\n\nStress hormone produced in the adrenal glands.\n"
    )
    (kb / "wiki" / "stress.md").write_text(
        "# stress\n\nThe body's response to perceived threat.\n"
    )
    (canon / "endocrinology.md").write_text(
        "# endocrinology\n\nCanonical reference for hormone systems.\n"
    )

    idx = KBIndex()
    idx.db = lancedb.connect(str(tmp_path / "lancedb"))
    idx._embedding_fn = FakeEmbeddingFunction()
    idx.build_index(extract_entities=False, llm_summaries=False, force=True)

    tools = KBTools(kb_index=idx, kb_dir=kb, canon_dir=canon)
    return tools, kb, idx


# ---------------------------------------------------------------------------
# Degraded modes
# ---------------------------------------------------------------------------

class TestDegradedModes:

    def test_no_kb_index(self):
        from agent.tools import KBTools
        out = KBTools(kb_index=None).compile_knowledge("foo.md")
        assert "No knowledge index available" in out

    def test_unknown_source(self, tools_with_kb):
        tools, _, _ = tools_with_kb
        out = tools.compile_knowledge("raw/does-not-exist.md")
        assert "FILE NOT FOUND" in out


# ---------------------------------------------------------------------------
# Source rendering + tier badging
# ---------------------------------------------------------------------------

class TestSourceRendering:

    def test_raw_source_shows_raw_tier(self, tools_with_kb):
        tools, _, _ = tools_with_kb
        out = tools.compile_knowledge("raw/stress-transcript.md")
        assert "# Compile Knowledge from: raw/stress-transcript.md" in out
        assert "[tier=raw]" in out
        # No tier-warning note for the canonical raw use case
        assert "NOTE:" not in out

    def test_wiki_source_shows_warning_note(self, tools_with_kb):
        tools, _, _ = tools_with_kb
        out = tools.compile_knowledge("wiki/cortisol.md")
        assert "[tier=wiki]" in out
        assert "already a wiki page" in out

    def test_canon_source_shows_canon_note(self, tools_with_kb):
        tools, _, _ = tools_with_kb
        out = tools.compile_knowledge("endocrinology.md")
        assert "[tier=canon]" in out
        assert "ground truth" in out

    def test_token_count_present(self, tools_with_kb):
        tools, _, _ = tools_with_kb
        out = tools.compile_knowledge("raw/stress-transcript.md")
        assert "size:" in out
        assert "tokens" in out


# ---------------------------------------------------------------------------
# Suggested output slug
# ---------------------------------------------------------------------------

class TestSuggestedSlug:

    def test_suggested_filename_lives_in_wiki(self, tools_with_kb):
        tools, _, _ = tools_with_kb
        out = tools.compile_knowledge("raw/stress-transcript.md")
        assert "## Suggested Output" in out
        assert "filename: wiki/stress-transcript.md" in out

    def test_special_chars_normalised_in_slug(self, tools_with_kb):
        tools, kb, idx = tools_with_kb
        # Source with messy stem — slug should still come out clean
        (kb / "raw" / "Weird File NAME (v2).md").write_text(
            "# weird\n\nbody.\n"
        )
        idx.build_index(extract_entities=False, llm_summaries=False, force=True)
        out = tools.compile_knowledge("raw/Weird File NAME (v2).md")
        # No spaces, no parens, no uppercase in the suggested slug
        assert "filename: wiki/weird-file-name-v2.md" in out


# ---------------------------------------------------------------------------
# Search bucket rendering
# ---------------------------------------------------------------------------

class TestRelatedSections:

    def test_related_wiki_and_canon_sections_rendered(self, tools_with_kb):
        tools, _, _ = tools_with_kb
        out = tools.compile_knowledge("raw/stress-transcript.md")
        assert "## Related Wiki Pages" in out
        assert "## Canon Anchors" in out
        assert "## Compilation Plan" in out

    def test_canon_citation_examples_use_relative_paths(self, tools_with_kb):
        tools, _, _ = tools_with_kb
        out = tools.compile_knowledge("raw/stress-transcript.md")
        assert "## Canon citation examples" in out
        assert "../../canon/" in out
        assert "canon/path.md" not in out

    def test_query_override_used_when_provided(self, tools_with_kb, monkeypatch):
        tools, _, idx = tools_with_kb
        seen = {}

        original = idx.search

        def spy(q, top_k=10):
            seen["q"] = q
            seen["top_k"] = top_k
            return original(q, top_k=top_k)

        monkeypatch.setattr(idx, "search", spy)
        tools.compile_knowledge(
            "raw/stress-transcript.md", query="explicit override query",
        )
        assert seen.get("q") == "explicit override query"

    def test_derived_query_used_when_omitted(self, tools_with_kb, monkeypatch):
        tools, _, idx = tools_with_kb
        seen = {}
        original = idx.search

        def spy(q, top_k=10):
            seen["q"] = q
            return original(q, top_k=top_k)

        monkeypatch.setattr(idx, "search", spy)
        tools.compile_knowledge("raw/stress-transcript.md")
        # Derived query should at minimum mention the source stem (sans hyphens)
        assert "stress" in (seen.get("q") or "").lower()

    def test_source_self_excluded_from_related(self, tools_with_kb, monkeypatch):
        """The source file's own chunks must never appear as 'related'."""
        tools, _, idx = tools_with_kb

        # Force every search hit to be the source itself — a degenerate case
        # that proves the self-filter is active.
        from knowledge.index import KBIndex
        tier_for = lambda s, fn: KBIndex._compute_tier(s, fn)

        def fake_search(q, top_k=10):
            return [{
                "filename": "raw/stress-transcript.md",
                "heading": "stress transcript",
                "summary": "x",
                "score": 0.99,
                "weighted_score": 0.99,
                "source": "knowledge",
                "tier": tier_for("knowledge", "raw/stress-transcript.md"),
            }]

        monkeypatch.setattr(idx, "search", fake_search)
        out = tools.compile_knowledge("raw/stress-transcript.md")
        # Every related-section bucket should be empty
        wiki_block = out.split("## Related Wiki Pages")[1].split("##")[0]
        canon_block = out.split("## Canon Anchors")[1].split("##")[0]
        assert "stress-transcript" not in wiki_block
        assert "stress-transcript" not in canon_block


# ---------------------------------------------------------------------------
# Side-effect contract — read-only
# ---------------------------------------------------------------------------

class TestReadOnly:

    def test_does_not_write_any_file(self, tools_with_kb):
        tools, kb, _ = tools_with_kb
        before = sorted(p.name for p in (kb / "wiki").rglob("*.md"))
        tools.compile_knowledge("raw/stress-transcript.md")
        after = sorted(p.name for p in (kb / "wiki").rglob("*.md"))
        assert before == after

    def test_does_not_create_suggested_file(self, tools_with_kb):
        tools, kb, _ = tools_with_kb
        tools.compile_knowledge("raw/stress-transcript.md")
        assert not (kb / "wiki" / "stress-transcript.md").exists()


# ---------------------------------------------------------------------------
# Tool registry membership (A1: native Ollama tool calling)
# ---------------------------------------------------------------------------

class TestToolRegistryMembership:

    def test_compile_knowledge_in_registry(self, tmp_path):
        from unittest.mock import MagicMock
        from agent.tools import KBTools, build_tool_registry, TOOL_CLASSES

        kb_tools = KBTools(MagicMock(), tmp_path, tmp_path)
        registry = build_tool_registry(kb_tools)
        assert "compile_knowledge" in registry
        assert callable(registry["compile_knowledge"])
        assert TOOL_CLASSES["compile_knowledge"] == "write"
