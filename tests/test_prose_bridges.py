"""Tests for the prose-bridge heuristic and edge builder (P2.2).

Layers under test:
  1. find_bridges  — pure heuristic on a body string
  2. _build_prose_bridge_edges — end-to-end: indexed chunks → REFERENCES
                                 edges with link_kind="prose" + evidence
"""

from pathlib import Path

import pytest

from knowledge.prose_bridges import find_bridges


# ---------------------------------------------------------------------------
# Layer 1: find_bridges (pure heuristic)
# ---------------------------------------------------------------------------

class TestFindBridgesBasic:

    def _pages(self, *names):
        """Build minimal known_pages dicts from filenames."""
        out = []
        for n in names:
            out.append({"filename": n, "source": "knowledge", "aliases": []})
        return out

    def test_simple_two_page_bridge(self):
        body = "Cortisol regulates the stress response in mammals."
        bridges = find_bridges(
            body, self._pages("wiki/cortisol.md", "wiki/stress.md"),
        )
        assert len(bridges) == 1
        b = bridges[0]
        assert b["subject_file"] == "wiki/cortisol.md"
        assert b["object_file"] == "wiki/stress.md"
        assert "Cortisol regulates" in b["evidence"]
        assert b["subject_match"].lower() == "cortisol"
        assert b["object_match"].lower() == "stress"

    def test_no_bridge_when_only_one_page_mentioned(self):
        body = "Cortisol is a steroid hormone produced by the adrenal glands."
        bridges = find_bridges(
            body, self._pages("wiki/cortisol.md", "wiki/stress.md"),
        )
        assert bridges == []

    def test_enumeration_skipped(self):
        """Pure list/enumeration with no verb between mentions → no bridge."""
        body = "Hormones to study: cortisol, dopamine, serotonin."
        bridges = find_bridges(
            body,
            self._pages("wiki/cortisol.md", "wiki/dopamine.md", "wiki/serotonin.md"),
        )
        # Comma-list with no verb between mentions has no connector
        # → conservative rule rejects it.
        assert bridges == []

    def test_verb_between_mentions_creates_bridge(self):
        body = "Stoicism deeply influenced Marcus Aurelius throughout his reign."
        bridges = find_bridges(
            body,
            self._pages("wiki/stoicism.md", "wiki/marcus-aurelius.md"),
        )
        # The slug 'marcus-aurelius' should match the prose 'Marcus Aurelius'
        assert len(bridges) == 1
        assert bridges[0]["subject_file"] == "wiki/stoicism.md"
        assert bridges[0]["object_file"] == "wiki/marcus-aurelius.md"

    def test_alias_matches_h1_heading(self):
        """Pages can declare aliases — H1 headings should bridge too."""
        pages = [
            {"filename": "wiki/marcus.md", "source": "knowledge",
             "aliases": ["Marcus Aurelius"]},
            {"filename": "wiki/stoicism.md", "source": "knowledge",
             "aliases": ["Stoicism"]},
        ]
        body = "Stoicism shaped the philosophy of Marcus Aurelius profoundly."
        bridges = find_bridges(body, pages)
        assert len(bridges) == 1
        assert bridges[0]["object_file"] == "wiki/marcus.md"


class TestFindBridgesGuards:
    """Conservative guards must hold."""

    def _pages(self, *names):
        return [{"filename": n, "source": "knowledge", "aliases": []} for n in names]

    def test_self_reference_not_emitted(self):
        body = "Cortisol affects how cortisol gets reabsorbed in the kidney."
        bridges = find_bridges(body, self._pages("wiki/cortisol.md"))
        assert bridges == []

    def test_wiki_link_text_not_double_counted(self):
        """Mentions inside [[wiki links]] must be stripped — that's P2.1's
        job. Prose bridges only fire on bare prose."""
        body = (
            "[[cortisol]] regulates [[stress]] through the HPA axis.\n"
            "But also: cortisol influences stress in non-link prose here."
        )
        bridges = find_bridges(
            body, self._pages("wiki/cortisol.md", "wiki/stress.md"),
        )
        # Only the second sentence (bare prose) should produce a bridge.
        assert len(bridges) == 1
        assert "non-link" in bridges[0]["evidence"]

    def test_markdown_link_text_not_double_counted(self):
        body = "[Cortisol info](cortisol.md) explains how it relates to stress."
        bridges = find_bridges(
            body, self._pages("wiki/cortisol.md", "wiki/stress.md"),
        )
        # Cortisol is wrapped in markdown link — stripped → only 'stress'
        # left → not enough for a bridge.
        assert bridges == []

    def test_inline_code_stripped(self):
        body = "Use `cortisol` in code, but stress is unrelated text here."
        bridges = find_bridges(
            body, self._pages("wiki/cortisol.md", "wiki/stress.md"),
        )
        assert bridges == []

    def test_fenced_code_stripped(self):
        body = (
            "Top-level discussion of stress response.\n\n"
            "```\ncortisol = 10\n```\n\n"
            "More text about stress alone."
        )
        bridges = find_bridges(
            body, self._pages("wiki/cortisol.md", "wiki/stress.md"),
        )
        assert bridges == []

    def test_substring_does_not_match(self):
        """'cortisone' should not match the slug 'cortisol' — whole-word only."""
        body = "Cortisone is similar to but distinct from synthetic stress molecules."
        bridges = find_bridges(
            body, self._pages("wiki/cortisol.md", "wiki/stress.md"),
        )
        assert bridges == []

    def test_short_alias_is_skipped(self):
        """Aliases under 3 chars are dropped to avoid noise like 'I' or 'a'."""
        pages = [
            {"filename": "wiki/it.md", "source": "knowledge", "aliases": ["IT"]},
            {"filename": "wiki/cortisol.md", "source": "knowledge", "aliases": []},
        ]
        body = "When IT teams discuss cortisol they usually mean stress."
        bridges = find_bridges(body, pages)
        # 'it' alias <3 chars → skipped → only one mention → no bridge
        assert bridges == []

    def test_dedup_within_chunk(self):
        """Same (subject, object) pair within a chunk → at most one edge."""
        body = (
            "Cortisol influences stress. "
            "Then again, cortisol reshapes stress in many ways."
        )
        bridges = find_bridges(
            body, self._pages("wiki/cortisol.md", "wiki/stress.md"),
        )
        assert len(bridges) == 1
        # First sentence wins as evidence.
        assert "influences" in bridges[0]["evidence"]


class TestFindBridgesEmpty:
    def test_empty_body(self):
        assert find_bridges("", [{"filename": "a.md", "source": "x"}]) == []

    def test_one_known_page(self):
        assert find_bridges(
            "cortisol and stress",
            [{"filename": "wiki/cortisol.md", "source": "knowledge"}],
        ) == []

    def test_no_known_pages(self):
        assert find_bridges("cortisol regulates stress", []) == []


# ---------------------------------------------------------------------------
# Layer 2: end-to-end via KBIndex._build_prose_bridge_edges
# ---------------------------------------------------------------------------

@pytest.fixture
def index_with_prose(tmp_path, monkeypatch):
    """Real KBIndex over a tiny corpus with bare-prose cross-references."""
    import lancedb
    import knowledge.index as kbi
    from knowledge.index import KBIndex
    from tests.conftest import FakeEmbeddingFunction

    kb = tmp_path / "knowledge"
    canon = tmp_path / "canon"
    (kb / "wiki").mkdir(parents=True)
    canon.mkdir()
    monkeypatch.setattr(kbi, "KB_DIR", kb)
    monkeypatch.setattr(kbi, "CANON_DIR", canon)
    monkeypatch.setattr(kbi, "LANCEDB_DIR", tmp_path / "lancedb")

    (kb / "wiki" / "cortisol.md").write_text(
        "# Cortisol\n\nCortisol regulates the stress response in mammals.\n"
    )
    (kb / "wiki" / "stress.md").write_text(
        "# Stress\n\nStress is the body's reaction to perceived threat.\n"
    )
    (kb / "wiki" / "dopamine.md").write_text(
        "# Dopamine\n\nDopamine is a reward neurotransmitter unrelated here.\n"
    )

    idx = KBIndex()
    idx.db = lancedb.connect(str(tmp_path / "lancedb"))
    idx._embedding_fn = FakeEmbeddingFunction()
    idx.build_index(extract_entities=False, llm_summaries=False, force=True)
    return idx


class TestProseBridgeEdgesEndToEnd:

    def test_prose_bridge_creates_edge(self, index_with_prose):
        from knowledge.graph import EdgeType
        graph = index_with_prose.graph
        prose_edges = [
            e for e in graph.edges.values()
            if e.edge_type == EdgeType.REFERENCES
            and e.attributes.get("link_kind") == "prose"
        ]
        assert prose_edges, "expected at least one prose-bridge REFERENCES edge"

    def test_prose_bridge_carries_evidence(self, index_with_prose):
        from knowledge.graph import EdgeType
        graph = index_with_prose.graph
        prose_edges = [
            e for e in graph.edges.values()
            if e.edge_type == EdgeType.REFERENCES
            and e.attributes.get("link_kind") == "prose"
        ]
        assert prose_edges
        for e in prose_edges:
            assert e.evidence, "prose bridge must carry the originating sentence"
            assert "regulates" in e.evidence.lower() or "stress" in e.evidence.lower()

    def test_unrelated_page_not_bridged(self, index_with_prose):
        """dopamine.md only mentions itself — must NOT bridge to cortisol/stress."""
        from knowledge.graph import EdgeType, NodeType
        graph = index_with_prose.graph
        chunks_by_id = {nid: n for nid, n in graph.nodes.items()
                        if n.node_type == NodeType.CHUNK}

        for e in graph.edges.values():
            if e.edge_type != EdgeType.REFERENCES:
                continue
            if e.attributes.get("link_kind") != "prose":
                continue
            src = chunks_by_id.get(e.source_id)
            tgt = chunks_by_id.get(e.target_id)
            if src and tgt:
                assert "dopamine" not in (src.filename + tgt.filename), (
                    "dopamine page should not appear in any prose bridge"
                )

    def test_wiki_link_takes_precedence_over_prose(self, tmp_path, monkeypatch):
        """When both an explicit [[wiki link]] and a prose mention exist for
        the same source→target chunk pair, the wiki-link edge (weight 1.0)
        must win over the prose edge (weight 0.5)."""
        import lancedb
        import knowledge.index as kbi
        from knowledge.index import KBIndex
        from knowledge.graph import EdgeType, NodeType
        from tests.conftest import FakeEmbeddingFunction

        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        (kb / "wiki").mkdir(parents=True)
        canon.mkdir()
        monkeypatch.setattr(kbi, "KB_DIR", kb)
        monkeypatch.setattr(kbi, "CANON_DIR", canon)
        monkeypatch.setattr(kbi, "LANCEDB_DIR", tmp_path / "lancedb")

        # stress.md has BOTH an explicit [[cortisol]] link AND bare prose
        # mentioning cortisol — only one REFERENCES edge should survive,
        # and it must be the wiki-link one.
        (kb / "wiki" / "cortisol.md").write_text("# Cortisol\n\nA hormone.\n")
        (kb / "wiki" / "stress.md").write_text(
            "# Stress\n\nSee [[cortisol]] for the canonical reference.\n\n"
            "Cortisol elevates stress through hypothalamic signaling.\n"
        )

        idx = KBIndex()
        idx.db = lancedb.connect(str(tmp_path / "lancedb"))
        idx._embedding_fn = FakeEmbeddingFunction()
        idx.build_index(extract_entities=False, llm_summaries=False, force=True)

        chunks_by_id = {nid: n for nid, n in idx.graph.nodes.items()
                        if n.node_type == NodeType.CHUNK}
        # Find the stress→cortisol edge(s)
        relevant = []
        for e in idx.graph.edges.values():
            if e.edge_type != EdgeType.REFERENCES:
                continue
            src = chunks_by_id.get(e.source_id)
            tgt = chunks_by_id.get(e.target_id)
            if (src and tgt and src.filename == "wiki/stress.md"
                    and tgt.filename == "wiki/cortisol.md"):
                relevant.append(e)
        assert relevant, "expected at least one stress→cortisol edge"
        # Highest-weight edge wins — must be the wiki-link, not the prose
        winner = max(relevant, key=lambda e: e.weight)
        assert winner.attributes.get("link_kind") == "wiki", (
            f"expected wiki link to win over prose, got {winner.attributes}"
        )

    def test_no_self_edge_from_prose(self, index_with_prose):
        from knowledge.graph import EdgeType
        for e in index_with_prose.graph.edges.values():
            if (e.edge_type == EdgeType.REFERENCES
                    and e.attributes.get("link_kind") == "prose"):
                assert e.source_id != e.target_id
