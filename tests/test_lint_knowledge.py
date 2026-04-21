"""Tests for P3.2 — lint_knowledge tool.

Covers:
  - orphan detection (chunks with no semantic edges)
  - broken wiki / markdown links
  - heading collisions across files
  - oversized chunks above the token threshold
  - scope filter (lint a single file)
  - parser whitelist (lint_knowledge accepted as a bare call)
  - degraded modes (no index, empty corpus, scope misses)
"""

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_kb(tmp_path, monkeypatch):
    """A real KBIndex over an empty temp KB; no chunks yet, caller seeds."""
    import lancedb
    import knowledge.index as kbi
    from knowledge.index import KBIndex
    from agent.tools import KBTools
    from tests.conftest import FakeEmbeddingFunction

    kb = tmp_path / "knowledge"
    canon = tmp_path / "canon"
    (kb / "wiki").mkdir(parents=True)
    canon.mkdir()
    monkeypatch.setattr(kbi, "KB_DIR", kb)
    monkeypatch.setattr(kbi, "CANON_DIR", canon)
    monkeypatch.setattr(kbi, "LANCEDB_DIR", tmp_path / "lancedb")

    idx = KBIndex()
    idx.db = lancedb.connect(str(tmp_path / "lancedb"))
    idx._embedding_fn = FakeEmbeddingFunction()
    idx.build_index(extract_entities=False, llm_summaries=False, force=True)

    tools = KBTools(kb_index=idx, kb_dir=kb, canon_dir=canon)
    return tools, kb, idx


def _seed_and_reindex(idx, kb, files: dict[str, str]):
    """Write files into the wiki dir and rebuild the index from scratch."""
    for name, body in files.items():
        path = kb / "wiki" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    idx.build_index(extract_entities=False, llm_summaries=False, force=True)


# ---------------------------------------------------------------------------
# Degraded modes
# ---------------------------------------------------------------------------

class TestDegradedModes:

    def test_no_kb_index(self):
        from agent.tools import KBTools
        tools = KBTools(kb_index=None)
        assert "No knowledge index available" in tools.lint_knowledge()

    def test_empty_corpus(self, fresh_kb):
        tools, _, _ = fresh_kb
        out = tools.lint_knowledge()
        assert "Knowledge base is empty" in out

    def test_scope_to_unknown_file(self, fresh_kb):
        tools, kb, idx = fresh_kb
        _seed_and_reindex(idx, kb, {"present.md": "# present\n\nbody.\n"})
        out = tools.lint_knowledge("wiki/missing.md")
        assert "no chunks indexed for 'wiki/missing.md'" in out


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

class TestOrphanDetection:

    def test_isolated_files_are_orphans(self, fresh_kb):
        tools, kb, idx = fresh_kb
        # Three completely unrelated single-section files. Random-hash fake
        # embeddings produce ~0 cosine similarity, so no SIMILAR /
        # INTER_FILE edges form → every chunk is an orphan.
        _seed_and_reindex(idx, kb, {
            "alpha.md": "# alpha\n\nUnique content A.\n",
            "beta.md": "# beta\n\nUnique content B.\n",
            "gamma.md": "# gamma\n\nUnique content C.\n",
        })
        out = tools.lint_knowledge()
        assert "## Orphans" in out
        assert "wiki/alpha.md" in out
        assert "wiki/beta.md" in out
        assert "wiki/gamma.md" in out

    def test_chunks_with_explicit_references_are_not_orphans(self, fresh_kb):
        tools, kb, idx = fresh_kb
        # alpha references beta via explicit wiki link → REFERENCES edge →
        # both chunks have a semantic edge → neither is an orphan.
        _seed_and_reindex(idx, kb, {
            "alpha.md": "# alpha\n\nLinks to [[beta]] for related material.\n",
            "beta.md": "# beta\n\nReferenced from alpha.\n",
        })
        out = tools.lint_knowledge()
        # Section may exist but neither file should be listed under it
        orphans_block = out.split("## Orphans")[1].split("##")[0]
        assert "wiki/alpha.md" not in orphans_block
        assert "wiki/beta.md" not in orphans_block


class TestBrokenLinkDetection:

    def test_wiki_link_to_missing_target(self, fresh_kb):
        tools, kb, idx = fresh_kb
        _seed_and_reindex(idx, kb, {
            "alpha.md": (
                "# alpha\n\nThis links to [[ghost-page]] which does not exist.\n"
            ),
        })
        out = tools.lint_knowledge()
        assert "## Broken Wiki Links" in out
        assert "ghost-page" in out
        assert "wiki/alpha.md" in out

    def test_markdown_link_to_missing_target(self, fresh_kb):
        tools, kb, idx = fresh_kb
        _seed_and_reindex(idx, kb, {
            "alpha.md": (
                "# alpha\n\nSee [the page](wiki/missing-md-target.md) for more.\n"
            ),
        })
        out = tools.lint_knowledge()
        assert "## Broken Wiki Links" in out
        assert "missing-md-target" in out

    def test_resolved_links_are_not_broken(self, fresh_kb):
        tools, kb, idx = fresh_kb
        _seed_and_reindex(idx, kb, {
            "alpha.md": "# alpha\n\nReferences [[beta]].\n",
            "beta.md": "# beta\n\nbody.\n",
        })
        out = tools.lint_knowledge()
        broken = out.split("## Broken Wiki Links")[1].split("##")[0]
        assert "(none)" in broken


class TestHeadingCollisions:

    def test_heading_in_3_or_more_files_flagged(self, fresh_kb):
        tools, kb, idx = fresh_kb
        # The chunker collapses H2-H5 into the parent H1 unless the file
        # exceeds the embed-token cap, so we use distinct H1 names that
        # share a common LEAF segment to provoke collision detection.
        # `chapter > overview` in three files → leaf "overview" wins.
        _seed_and_reindex(idx, kb, {
            "alpha.md": "# overview\n\nFirst page covering an overview topic.\n",
            "beta.md": "# overview\n\nSecond page also titled overview.\n",
            "gamma.md": "# overview\n\nThird page also titled overview.\n",
        })
        out = tools.lint_knowledge()
        assert "## Heading Collisions" in out
        collisions = out.split("## Heading Collisions")[1].split("##")[0]
        assert "overview" in collisions.lower()
        assert "(none)" not in collisions

    def test_unique_headings_not_flagged(self, fresh_kb):
        tools, kb, idx = fresh_kb
        _seed_and_reindex(idx, kb, {
            "alpha.md": "# history\n\nA.\n",
            "beta.md": "# etymology\n\nB.\n",
        })
        out = tools.lint_knowledge()
        collisions = out.split("## Heading Collisions")[1].split("##")[0]
        assert "(none)" in collisions


class TestOversizedChunks:

    def test_oversized_chunk_flagged(self, fresh_kb, monkeypatch):
        tools, kb, idx = fresh_kb
        # Push the threshold way down so a small synthetic file trips it
        monkeypatch.setattr(tools, "LINT_OVERSIZED_TOKENS", 5)
        _seed_and_reindex(idx, kb, {
            "alpha.md": "# alpha\n\nThis body has more than five tokens easily.\n",
        })
        out = tools.lint_knowledge()
        assert "## Oversized Chunks" in out
        assert "wiki/alpha.md" in out

    def test_small_chunk_not_flagged(self, fresh_kb):
        tools, kb, idx = fresh_kb
        _seed_and_reindex(idx, kb, {
            "alpha.md": "# alpha\n\nshort.\n",
        })
        out = tools.lint_knowledge()
        oversized = out.split("## Oversized Chunks")[1].split("##")[0]
        assert "(none)" in oversized


class TestScopeFilter:

    def test_scope_limits_orphan_detection_to_file(self, fresh_kb):
        tools, kb, idx = fresh_kb
        _seed_and_reindex(idx, kb, {
            "alpha.md": "# alpha\n\nA.\n",
            "beta.md": "# beta\n\nB.\n",
        })
        out = tools.lint_knowledge("wiki/alpha.md")
        # Header reflects the scope
        assert "Scope: wiki/alpha.md" in out
        # Beta should not appear in any finding
        assert "wiki/beta.md" not in out

    def test_scope_omits_heading_collisions_section(self, fresh_kb):
        """Collisions are corpus-wide signal — single-file scope hides
        the section entirely (would be misleading on n=1)."""
        tools, kb, idx = fresh_kb
        _seed_and_reindex(idx, kb, {
            "alpha.md": "# alpha\n\n## overview\n\nA.\n",
            "beta.md": "# beta\n\n## overview\n\nB.\n",
            "gamma.md": "# gamma\n\n## overview\n\nC.\n",
        })
        out = tools.lint_knowledge("wiki/alpha.md")
        assert "## Heading Collisions" not in out


# ---------------------------------------------------------------------------
# Wiki frontmatter + markdown href hygiene
# ---------------------------------------------------------------------------

class TestWikiHygieneLint:

    def test_flags_nested_tags_in_frontmatter(self, fresh_kb):
        tools, kb, idx = fresh_kb
        bad = (
            "---\n"
            "tags:\n"
            "  - [philosophy, neuroscience]\n"
            "---\n"
            "# bad\n\nbody.\n"
        )
        _seed_and_reindex(idx, kb, {"bad.md": bad})
        out = tools.lint_knowledge()
        assert "## Nested-tag" in out
        assert "wiki/bad.md" in out

    def test_flags_canon_id_in_markdown_href(self, fresh_kb):
        tools, kb, idx = fresh_kb
        body = "# link\n\nSee [quote](canon:mind-en-place/foo.md#bar).\n"
        _seed_and_reindex(idx, kb, {"link.md": body})
        out = tools.lint_knowledge()
        assert "## Markdown hrefs" in out
        assert "wiki/link.md" in out


# ---------------------------------------------------------------------------
# Tool registry membership
# ---------------------------------------------------------------------------
# A1 migrated tool dispatch from the old [TOOL: ...] regex parser to Ollama's
# native tool calling. The registry built by build_tool_registry is now the
# authoritative source of truth for which tools the agent can call. These
# tests pin lint_knowledge into that registry so a future refactor can't
# silently drop it.

class TestToolRegistryMembership:

    def test_lint_knowledge_in_registry(self, tmp_path):
        from unittest.mock import MagicMock
        from agent.tools import KBTools, build_tool_registry, TOOL_CLASSES

        kb_tools = KBTools(MagicMock(), tmp_path, tmp_path)
        registry = build_tool_registry(kb_tools)
        assert "lint_knowledge" in registry
        assert callable(registry["lint_knowledge"])
        assert TOOL_CLASSES["lint_knowledge"] == "maintenance"
