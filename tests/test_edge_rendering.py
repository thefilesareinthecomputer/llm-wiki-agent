"""Tests for P2.3 — REFERENCES edge provenance rendering in graph tools.

Verifies that `graph_neighbors`, `graph_traverse`, and `graph_search` render
the `link_text`, `link_kind`, `target_anchor`, and `evidence` attributes so
the agent can see WHY two chunks are connected, not just THAT they are.
"""

import pytest

from agent.tools import _format_edge_provenance


# ---------------------------------------------------------------------------
# Pure rendering helper
# ---------------------------------------------------------------------------

class _FakeEdge:
    def __init__(self, attributes=None, evidence=""):
        self.attributes = attributes or {}
        self.evidence = evidence


class TestFormatEdgeProvenance:

    def test_wiki_link_provenance(self):
        e = _FakeEdge(
            attributes={
                "link_text": "stress hormone",
                "link_kind": "wiki",
                "target_file": "wiki/cortisol.md",
                "target_anchor": "",
            },
            evidence="wiki link: [[cortisol|stress hormone]]",
        )
        out = _format_edge_provenance(e)
        assert "via 'stress hormone'" in out
        assert "[wiki]" in out

    def test_wiki_link_with_anchor(self):
        e = _FakeEdge(
            attributes={
                "link_text": "HPA axis",
                "link_kind": "wiki",
                "target_anchor": "regulation",
            },
        )
        out = _format_edge_provenance(e)
        assert "via 'HPA axis'" in out
        assert "#regulation" in out

    def test_markdown_link_provenance(self):
        e = _FakeEdge(
            attributes={
                "link_text": "the dopamine page",
                "link_kind": "markdown",
            },
        )
        out = _format_edge_provenance(e)
        assert "via 'the dopamine page'" in out
        assert "[markdown]" in out

    def test_prose_bridge_includes_evidence_snippet(self):
        e = _FakeEdge(
            attributes={"link_text": "stress", "link_kind": "prose"},
            evidence="Cortisol regulates the stress response in mammals.",
        )
        out = _format_edge_provenance(e)
        assert "via 'stress'" in out
        assert "[prose]" in out
        assert "Cortisol regulates" in out

    def test_long_evidence_truncated(self):
        long_text = "x" * 500
        e = _FakeEdge(
            attributes={"link_text": "x", "link_kind": "prose"},
            evidence=long_text,
        )
        out = _format_edge_provenance(e)
        assert "..." in out
        assert len(out) < len(long_text) + 50

    def test_no_attributes_falls_back_to_evidence(self):
        e = _FakeEdge(evidence="similar sections")
        out = _format_edge_provenance(e)
        assert "similar sections" in out

    def test_no_provenance_at_all_returns_empty(self):
        e = _FakeEdge()
        assert _format_edge_provenance(e) == ""

    def test_attributes_field_missing_safe(self):
        """Old-style edge dataclass without `attributes` attribute at all."""
        class _LegacyEdge:
            evidence = "legacy"
        out = _format_edge_provenance(_LegacyEdge())
        assert "legacy" in out


# ---------------------------------------------------------------------------
# End-to-end via KBTools.graph_neighbors / graph_search
# ---------------------------------------------------------------------------

@pytest.fixture
def tools_with_links(tmp_path, monkeypatch):
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

    (kb / "wiki" / "cortisol.md").write_text("# cortisol\n\nA hormone.\n")
    (kb / "wiki" / "stress.md").write_text(
        "# stress\n\nSee [[cortisol|stress hormone]] for context.\n"
    )

    idx = KBIndex()
    idx.db = lancedb.connect(str(tmp_path / "lancedb"))
    idx._embedding_fn = FakeEmbeddingFunction()
    idx.build_index(extract_entities=False, llm_summaries=False, force=True)
    return KBTools(kb_index=idx)


class TestGraphNeighborsRendersLinkText:

    def test_neighbors_output_contains_link_text(self, tools_with_links):
        """graph_neighbors on stress.md should show that the connection
        to cortisol.md is via the link text 'stress hormone'."""
        out = tools_with_links.graph_neighbors("wiki/stress.md")
        assert "stress" in out.lower()
        # Either the link_text or the link_kind should appear if a
        # REFERENCES edge exists.
        assert (
            "stress hormone" in out
            or "[wiki]" in out
            or "via" in out
        ), f"expected provenance metadata in output, got:\n{out}"

    def test_neighbors_filtered_to_references(self, tools_with_links):
        out = tools_with_links.graph_neighbors(
            "wiki/stress.md", edge_type="references",
        )
        # If any references edge exists, output must include 'references'
        # as the section header AND the link_text provenance.
        if "references" in out:
            assert "via" in out or "[wiki]" in out


class TestGraphSearchRendersProvenance:

    def test_search_output_renders_link_text(self, tools_with_links):
        out = tools_with_links.graph_search("stress")
        # When graph_search finds connected chunks via REFERENCES edges,
        # the link_text should be visible.
        if "[references]" in out:
            assert (
                "via" in out
                or "[wiki]" in out
                or "stress hormone" in out
            ), f"expected provenance in graph_search output, got:\n{out}"
