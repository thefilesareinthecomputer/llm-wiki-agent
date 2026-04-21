"""Tests for P2.4 — graph_stats provenance breakdown.

Validates that `graph_stats` distinguishes:
  - explicit references (wiki + markdown)
  - prose-bridge (heuristic)
  - similarity (embedding cosine)
  - hierarchy (parent_child)
  - entity (relates_to)

so the agent can tell at a glance whether the graph is author-linked,
model-inferred, or purely structural.
"""

import pytest


@pytest.fixture
def tools_with_mixed_edges(tmp_path, monkeypatch):
    """KB with explicit wiki links + prose mentions → mixed edge provenance."""
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
        "# stress\n\nSee [[cortisol]] for the explicit link.\n\n"
        "Cortisol elevates stress through hypothalamic signaling.\n"
    )
    (kb / "wiki" / "dopamine.md").write_text(
        "# dopamine\n\nA different neurotransmitter.\n"
    )

    idx = KBIndex()
    idx.db = lancedb.connect(str(tmp_path / "lancedb"))
    idx._embedding_fn = FakeEmbeddingFunction()
    idx.build_index(extract_entities=False, llm_summaries=False, force=True)
    return KBTools(kb_index=idx)


class TestGraphStatsProvenanceBreakdown:

    def test_references_breakdown_present(self, tools_with_mixed_edges):
        out = tools_with_mixed_edges.graph_stats()
        assert "REFERENCES Breakdown" in out

    def test_provenance_classes_section_present(self, tools_with_mixed_edges):
        out = tools_with_mixed_edges.graph_stats()
        assert "Provenance Classes" in out

    def test_explicit_reference_class_listed(self, tools_with_mixed_edges):
        out = tools_with_mixed_edges.graph_stats()
        assert "explicit-reference" in out

    def test_similarity_class_listed(self, tools_with_mixed_edges):
        out = tools_with_mixed_edges.graph_stats()
        assert "similarity" in out

    def test_hierarchy_class_listed(self, tools_with_mixed_edges):
        out = tools_with_mixed_edges.graph_stats()
        assert "hierarchy" in out

    def test_wiki_links_counted_in_breakdown(self, tools_with_mixed_edges):
        """The corpus has at least one [[cortisol]] wiki link → wiki count
        in the REFERENCES Breakdown section must be ≥ 1."""
        out = tools_with_mixed_edges.graph_stats()
        # Find the line under "REFERENCES Breakdown" mentioning wiki
        lines = out.splitlines()
        in_block = False
        wiki_line = None
        for line in lines:
            if "REFERENCES Breakdown" in line:
                in_block = True
                continue
            if in_block and line.strip().startswith("##"):
                break
            if in_block and "wiki:" in line:
                wiki_line = line
                break
        assert wiki_line, f"expected 'wiki:' in REFERENCES Breakdown, got:\n{out}"
        # Extract the count
        import re
        m = re.search(r"wiki:\s*(\d+)", wiki_line)
        assert m, f"could not parse wiki count from line: {wiki_line}"
        assert int(m.group(1)) >= 1


class TestGraphStatsEmptyGraphSafe:

    def test_no_graph_returns_message(self, tmp_path):
        from agent.tools import KBTools
        tools = KBTools(kb_index=None)
        out = tools.graph_stats()
        assert "No knowledge graph" in out

    def test_no_references_no_breakdown_block(self, tmp_path, monkeypatch):
        """When the corpus has zero REFERENCES edges, the breakdown block
        should be omitted (clean output, no zero-count noise)."""
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

        # Single-page corpus → impossible to have any references
        (kb / "wiki" / "alone.md").write_text("# alone\n\nJust me.\n")

        idx = KBIndex()
        idx.db = lancedb.connect(str(tmp_path / "lancedb"))
        idx._embedding_fn = FakeEmbeddingFunction()
        idx.build_index(extract_entities=False, llm_summaries=False, force=True)
        tools = KBTools(kb_index=idx)

        out = tools.graph_stats()
        assert "REFERENCES Breakdown" not in out


class TestGraphStatsEdgeShareLine:
    """P0-2: a single-line `edge_share:` summary so the model never has to
    recompute percentages from raw counts."""

    def test_edge_share_line_present(self, tools_with_mixed_edges):
        out = tools_with_mixed_edges.graph_stats()
        assert "edge_share:" in out

    def test_edge_share_uses_one_decimal(self, tools_with_mixed_edges):
        out = tools_with_mixed_edges.graph_stats()
        line = next(l for l in out.splitlines() if l.startswith("edge_share:"))
        # Each share must be formatted like ``parent_child=23.0%``
        import re
        m = re.search(r"\b\w+=\d+\.\d%", line)
        assert m, f"expected NN.N% formatting in: {line}"

    def test_edge_share_omits_zero_count_types(self, tmp_path):
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType
        from agent.tools import KBTools
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        a = Node(id="a", node_type=NodeType.CHUNK, name="x.md > A",
                 filename="x.md", heading="A",
                 attributes={"source": "knowledge"})
        b = Node(id="b", node_type=NodeType.CHUNK, name="y.md > B",
                 filename="y.md", heading="B",
                 attributes={"source": "knowledge"})
        graph.add_node(a)
        graph.add_node(b)
        # Only one edge type — `similar`.
        graph.add_edge(Edge(source_id="a", target_id="b",
                            edge_type=EdgeType.SIMILAR, weight=0.8))

        index = KBIndex.__new__(KBIndex)
        index.graph = graph
        index.table = None
        tools = KBTools(kb_index=index)

        out = tools.graph_stats()
        line = next(l for l in out.splitlines() if l.startswith("edge_share:"))
        assert "similar=" in line
        # Edge types with zero count must not appear.
        assert "parent_child=" not in line
        assert "inter_file=" not in line
        assert "cross_domain=" not in line

    def test_edge_share_in_get_stats_dict(self, tools_with_mixed_edges):
        """The same numbers must be available structurally so /kb/stats JSON
        carries them too."""
        stats = tools_with_mixed_edges.kb_index.graph.get_stats()
        assert "edge_share" in stats
        assert isinstance(stats["edge_share"], dict)
        # Every value is a float fraction in [0, 1]
        for k, v in stats["edge_share"].items():
            assert 0.0 <= v <= 1.0


class TestGraphStatsPerHubShare:
    """P0-2: each ``Most Connected`` entry shows the share of non-parent_child
    edges that touch the hub."""

    def test_share_appears_in_most_connected_entry(self, tools_with_mixed_edges):
        out = tools_with_mixed_edges.graph_stats()
        # At least one entry should carry the share annotation.
        assert "% of non-parent_child edges" in out

    def test_share_present_in_get_stats_dict(self, tools_with_mixed_edges):
        stats = tools_with_mixed_edges.kb_index.graph.get_stats()
        for entry in stats.get("most_connected", []):
            assert "share_non_pc" in entry
            assert 0.0 <= entry["share_non_pc"] <= 1.0
