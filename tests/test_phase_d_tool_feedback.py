"""Phase D — tool feedback fixes (post-audit).

Covers the surgical changes made in response to the agent's self-audit:

D1 — describe_node grew a ``min_weight`` filter; graph_traverse grew
     ``offset`` / ``limit`` pagination and ``exclude_edge_types`` for
     dropping noisy edge types in addition to the always-excluded
     PARENT_CHILD.
D2 — folder_tree accepts a ``<source>/<sub/path>`` drill-down argument;
     format_folder_tree threads ``root_path`` end-to-end.
D3 — Intra-file SIMILAR edges are capped at INTRA_FILE_SIMILAR_CAP per
     node, ranked 1..N by weight via ``attributes['intra_rank']`` /
     ``intra_total``, and surfaced by ``_format_edge_provenance`` as
     ``rank N/M in file``.
D5 — lint_knowledge gained a ``Flat-Similarity Clusters`` section that
     flags files whose intra-file SIMILAR weights bunch into a narrow
     band.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _make_graph(tmp_path: Path):
    from knowledge.graph import KnowledgeGraph

    return KnowledgeGraph(persist_path=tmp_path / "graph.json")


def _make_chunk(graph, node_id: str, filename: str, heading: str, source: str = "knowledge"):
    from knowledge.graph import Node, NodeType

    n = Node(
        id=node_id,
        node_type=NodeType.CHUNK,
        name=heading,
        filename=filename,
        heading=heading,
        attributes={"source": source},
    )
    graph.add_node(n)
    return n


def _add_edge(graph, src, tgt, etype, weight, attributes=None, evidence=""):
    from knowledge.graph import Edge

    graph.add_edge(
        Edge(
            source_id=src,
            target_id=tgt,
            edge_type=etype,
            weight=weight,
            evidence=evidence,
            attributes=attributes or {},
        )
    )


def _build_kbtools(graph, tmp_path):
    from agent.tools import KBTools

    class _StubKBIndex:
        def __init__(self, g):
            self.graph = g

    return KBTools(
        kb_index=_StubKBIndex(graph),
        kb_dir=tmp_path / "knowledge",
        canon_dir=tmp_path / "canon",
    )


# ==========================================================================
# D1 — describe_node(min_weight=)
# ==========================================================================


def test_describe_node_min_weight_filters_weak_edges(tmp_path):
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "a", "wiki/a.md", "A")
    _make_chunk(graph, "b1", "wiki/b1.md", "B1")
    _make_chunk(graph, "b2", "wiki/b2.md", "B2")
    _make_chunk(graph, "b3", "wiki/b3.md", "B3")
    # Strong edge (kept), two weak edges (filtered when min_weight=0.7)
    _add_edge(graph, "a", "b1", EdgeType.INTER_FILE, 0.92, evidence="strong")
    _add_edge(graph, "a", "b2", EdgeType.INTER_FILE, 0.60, evidence="weak1")
    _add_edge(graph, "a", "b3", EdgeType.INTER_FILE, 0.55, evidence="weak2")

    tools = _build_kbtools(graph, tmp_path)

    out_unfiltered = tools.describe_node("wiki/a.md", "A")
    assert "B1" in out_unfiltered
    assert "B2" in out_unfiltered
    assert "B3" in out_unfiltered

    out_filtered = tools.describe_node("wiki/a.md", "A", min_weight=0.7)
    assert "B1" in out_filtered  # strong edge kept
    assert "B2" not in out_filtered
    assert "B3" not in out_filtered
    # The summary line tells the agent how many edges were suppressed.
    assert "min_weight=0.70" in out_filtered
    assert "suppressed 2 edges" in out_filtered


def test_describe_node_min_weight_can_zero_out_all_edges(tmp_path):
    """When min_weight filters everything, describe_node says so instead
    of mis-labelling the node as an orphan."""
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "a", "wiki/a.md", "A")
    _make_chunk(graph, "b", "wiki/b.md", "B")
    _add_edge(graph, "a", "b", EdgeType.INTER_FILE, 0.5)

    tools = _build_kbtools(graph, tmp_path)
    out = tools.describe_node("wiki/a.md", "A", min_weight=0.99)
    assert "no edges above min_weight=0.99" in out
    assert "Re-call with a lower min_weight" in out
    # Crucially NOT the orphan branch — that's a real-graph property,
    # not a filter property.
    assert "(orphan)" not in out


def test_describe_node_min_weight_invalid_falls_back_to_zero(tmp_path):
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "a", "wiki/a.md", "A")
    _make_chunk(graph, "b", "wiki/b.md", "B")
    _add_edge(graph, "a", "b", EdgeType.INTER_FILE, 0.5)

    tools = _build_kbtools(graph, tmp_path)
    # A bad value must not blow up; behaves like min_weight=0.
    out = tools.describe_node("wiki/a.md", "A", min_weight="not-a-number")
    assert "B" in out


# ==========================================================================
# D1 — graph_traverse(offset=, limit=, exclude_edge_types=)
# ==========================================================================


def test_graph_traverse_pagination_offset_and_limit(tmp_path):
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "hub", "wiki/hub.md", "hub")
    for i in range(20):
        nid = f"n{i:02d}"
        _make_chunk(graph, nid, f"wiki/n{i}.md", f"n{i}")
        # weight high so min_weight filter doesn't interfere with the
        # paging assertion
        _add_edge(graph, "hub", nid, EdgeType.INTER_FILE, 0.7 + (i / 1000.0))

    tools = _build_kbtools(graph, tmp_path)

    page1 = tools.graph_traverse("wiki/hub.md", "hub", depth=1, limit=5)
    assert "[showing edges 0-5 of 20" in page1
    assert "call again with offset=5" in page1

    page2 = tools.graph_traverse(
        "wiki/hub.md", "hub", depth=1, offset=5, limit=5
    )
    assert "[showing edges 5-10 of 20" in page2
    # Different page = different content from page1
    assert page1 != page2


def test_graph_traverse_offset_past_end_returns_clear_message(tmp_path):
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "hub", "wiki/hub.md", "hub")
    for i in range(3):
        nid = f"n{i}"
        _make_chunk(graph, nid, f"wiki/n{i}.md", f"n{i}")
        _add_edge(graph, "hub", nid, EdgeType.INTER_FILE, 0.7)

    tools = _build_kbtools(graph, tmp_path)
    out = tools.graph_traverse(
        "wiki/hub.md", "hub", depth=1, offset=50, limit=5
    )
    assert "No edges in offset window 50-55 of 3 total" in out


def test_graph_traverse_exclude_edge_types_drops_similar(tmp_path):
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "a", "wiki/a.md", "A")
    _make_chunk(graph, "a2", "wiki/a.md", "A2")  # same file → SIMILAR
    _make_chunk(graph, "b", "wiki/b.md", "B")    # different file → INTER_FILE
    _add_edge(graph, "a", "a2", EdgeType.SIMILAR, 0.85,
              attributes={"intra_rank": 1, "intra_total": 1})
    _add_edge(graph, "a", "b", EdgeType.INTER_FILE, 0.7)

    tools = _build_kbtools(graph, tmp_path)

    out_all = tools.graph_traverse("wiki/a.md", "A", depth=1)
    assert "A2" in out_all
    assert "B" in out_all

    out_no_similar = tools.graph_traverse(
        "wiki/a.md", "A", depth=1, exclude_edge_types="similar"
    )
    assert "[similar]" not in out_no_similar
    assert "A2" not in out_no_similar
    assert "B" in out_no_similar


def test_graph_traverse_exclude_edge_types_invalid_value(tmp_path):
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "a", "wiki/a.md", "A")
    _make_chunk(graph, "b", "wiki/b.md", "B")
    _add_edge(graph, "a", "b", EdgeType.INTER_FILE, 0.7)

    tools = _build_kbtools(graph, tmp_path)
    out = tools.graph_traverse(
        "wiki/a.md", "A", depth=1, exclude_edge_types="bogus,similar"
    )
    assert "Invalid exclude_edge_types" in out
    assert "bogus" in out


def test_graph_traverse_parent_child_in_exclude_list_is_accepted_and_noted(tmp_path):
    """P0-3: passing ``parent_child`` in ``exclude_edge_types`` is accepted
    silently with a one-line note instead of a hard refusal. Traversal
    still runs and produces normal output."""
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "a", "wiki/a.md", "A")
    _make_chunk(graph, "b", "wiki/b.md", "B")
    _add_edge(graph, "a", "b", EdgeType.INTER_FILE, 0.7)

    tools = _build_kbtools(graph, tmp_path)
    out = tools.graph_traverse(
        "wiki/a.md", "A", depth=1, exclude_edge_types="parent_child"
    )
    # Not a hard refusal anymore.
    assert "Invalid exclude_edge_types" not in out
    # Friendly note is present.
    assert "parent_child" in out
    assert "always excluded" in out
    assert "dropped" in out.lower()
    # Traversal body still rendered.
    assert "[inter_file]" in out


def test_graph_traverse_parent_child_with_other_valid_excludes(tmp_path):
    """P0-3: ``parent_child,similar`` accepts and notes parent_child while
    still applying ``similar`` exclusion."""
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "a", "wiki/a.md", "A")
    _make_chunk(graph, "b", "wiki/b.md", "B")
    _make_chunk(graph, "c", "wiki/c.md", "C")
    _add_edge(graph, "a", "b", EdgeType.INTER_FILE, 0.7)
    _add_edge(graph, "a", "c", EdgeType.SIMILAR, 0.85)

    tools = _build_kbtools(graph, tmp_path)
    out = tools.graph_traverse(
        "wiki/a.md", "A", depth=1, exclude_edge_types="parent_child,similar"
    )
    # No refusal.
    assert "Invalid exclude_edge_types" not in out
    # Note present.
    assert "always excluded" in out
    # `similar` exclusion still honored — no [similar] edge in output.
    assert "[similar]" not in out
    # `inter_file` neighbour still present.
    assert "[inter_file]" in out


def test_graph_traverse_parent_child_with_unknown_token_still_errors(tmp_path):
    """P0-3 deliberately keeps the validation strict for genuinely-unknown
    tokens; mixing parent_child with an unknown token yields the existing
    invalid-value error (not the friendly note)."""
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "a", "wiki/a.md", "A")
    _make_chunk(graph, "b", "wiki/b.md", "B")
    _add_edge(graph, "a", "b", EdgeType.INTER_FILE, 0.7)

    tools = _build_kbtools(graph, tmp_path)
    out = tools.graph_traverse(
        "wiki/a.md", "A", depth=1, exclude_edge_types="parent_child,bogus"
    )
    assert "Invalid exclude_edge_types" in out
    assert "bogus" in out


def test_graph_traverse_parent_child_still_always_excluded(tmp_path):
    """PARENT_CHILD must remain hardcoded-excluded even when the agent
    passes a different exclude_edge_types value."""
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "a", "wiki/a.md", "A")
    _make_chunk(graph, "child", "wiki/a.md", "A > child")
    _make_chunk(graph, "b", "wiki/b.md", "B")
    _add_edge(graph, "a", "child", EdgeType.PARENT_CHILD, 1.0)
    # Real semantic neighbour so we don't trip the empty-neighbours
    # vector-search fallback (which would need a stub `search` method).
    _add_edge(graph, "a", "b", EdgeType.INTER_FILE, 0.7)

    tools = _build_kbtools(graph, tmp_path)
    out = tools.graph_traverse(
        "wiki/a.md", "A", depth=1, exclude_edge_types="similar"
    )
    # PARENT_CHILD edge must not show up regardless of caller args.
    assert "[parent_child]" not in out
    assert "A > child" not in out
    # Real semantic neighbour still rendered.
    assert "B" in out
    assert "[inter_file]" in out


# ==========================================================================
# D2 — folder_tree drill-down
# ==========================================================================


def test_folder_tree_drilldown_into_subpath(tmp_path):
    from knowledge.graph import Node, NodeType, Edge, EdgeType

    graph = _make_graph(tmp_path)
    # Build a tiny canon hierarchy: canon/foo/{bar,baz}
    foo = Node(
        id="folder_canon_foo", node_type=NodeType.FOLDER, name="foo",
        attributes={"source": "canon", "tier": "canon"},
    )
    bar = Node(
        id="folder_canon_foo_bar", node_type=NodeType.FOLDER, name="foo/bar",
        attributes={"source": "canon", "tier": "canon"},
    )
    baz = Node(
        id="folder_canon_foo_baz", node_type=NodeType.FOLDER, name="foo/baz",
        attributes={"source": "canon", "tier": "canon"},
    )
    other = Node(
        id="folder_canon_other", node_type=NodeType.FOLDER, name="other",
        attributes={"source": "canon", "tier": "canon"},
    )
    for n in (foo, bar, baz, other):
        graph.add_node(n)
    graph.add_edge(Edge(
        source_id=foo.id, target_id=bar.id,
        edge_type=EdgeType.PARENT_CHILD, weight=1.0,
    ))
    graph.add_edge(Edge(
        source_id=foo.id, target_id=baz.id,
        edge_type=EdgeType.PARENT_CHILD, weight=1.0,
    ))

    class _StubKBIndex:
        def __init__(self, g):
            self.graph = g

        def get_folder_tree(self, source="knowledge", root_path=None):
            from knowledge.graph import format_folder_tree
            return format_folder_tree(self.graph, source=source, root_path=root_path)

    from agent.tools import KBTools

    tools = KBTools(
        kb_index=_StubKBIndex(graph),
        kb_dir=tmp_path / "knowledge",
        canon_dir=tmp_path / "canon",
    )

    # Drill into canon/foo — should see bar and baz, but not the
    # sibling 'other'.
    out = tools.folder_tree("canon/foo")
    assert "bar/" in out
    assert "baz/" in out
    assert "other/" not in out
    assert "canon/foo/" in out  # header line names the subtree


def test_folder_tree_drilldown_unknown_path_returns_clear_error(tmp_path):
    from knowledge.graph import Node, NodeType

    graph = _make_graph(tmp_path)
    real = Node(
        id="folder_canon_foo", node_type=NodeType.FOLDER, name="foo",
        attributes={"source": "canon", "tier": "canon"},
    )
    graph.add_node(real)

    class _StubKBIndex:
        def __init__(self, g):
            self.graph = g

        def get_folder_tree(self, source="knowledge", root_path=None):
            from knowledge.graph import format_folder_tree
            return format_folder_tree(self.graph, source=source, root_path=root_path)

    from agent.tools import KBTools

    tools = KBTools(
        kb_index=_StubKBIndex(graph),
        kb_dir=tmp_path / "knowledge",
        canon_dir=tmp_path / "canon",
    )
    out = tools.folder_tree("canon/does-not-exist")
    assert "No folder 'canon/does-not-exist'" in out


def test_folder_tree_invalid_drilldown_source_rejected(tmp_path):
    """A drill-down path whose head isn't 'canon' or 'knowledge' is a
    user error and must surface explicitly, not silently render
    everything."""
    from knowledge.graph import Node, NodeType

    graph = _make_graph(tmp_path)
    graph.add_node(Node(
        id="folder_canon_foo", node_type=NodeType.FOLDER, name="foo",
        attributes={"source": "canon", "tier": "canon"},
    ))

    class _StubKBIndex:
        def __init__(self, g):
            self.graph = g

        def get_folder_tree(self, source="knowledge", root_path=None):
            from knowledge.graph import format_folder_tree
            return format_folder_tree(self.graph, source=source, root_path=root_path)

    from agent.tools import KBTools

    tools = KBTools(
        kb_index=_StubKBIndex(graph),
        kb_dir=tmp_path / "knowledge",
        canon_dir=tmp_path / "canon",
    )
    out = tools.folder_tree("bogus/sub")
    assert "Invalid folder" in out


def test_format_folder_tree_root_path_unit(tmp_path):
    """Lower-level guard: format_folder_tree with no matching root_path
    must return ''."""
    from knowledge.graph import Node, NodeType, format_folder_tree

    graph = _make_graph(tmp_path)
    graph.add_node(Node(
        id="folder_canon_foo", node_type=NodeType.FOLDER, name="foo",
        attributes={"source": "canon", "tier": "canon"},
    ))
    assert format_folder_tree(graph, source="canon", root_path="missing") == ""
    out = format_folder_tree(graph, source="canon", root_path="foo")
    assert "foo/" in out


# ==========================================================================
# D3 — intra-file SIMILAR cap + rank
# ==========================================================================


def test_format_edge_provenance_renders_intra_rank_for_similar():
    """Unit: when an edge carries intra_rank/intra_total in attributes,
    _format_edge_provenance must emit the human-readable rank string and
    NOT fall back to the freeform evidence text."""
    from agent.tools import _format_edge_provenance
    from knowledge.graph import Edge, EdgeType

    edge = Edge(
        source_id="a", target_id="b",
        edge_type=EdgeType.SIMILAR,
        weight=0.84,
        evidence="semantic similarity (0.84)",
        attributes={"intra_rank": 2, "intra_total": 5},
    )
    out = _format_edge_provenance(edge)
    assert out == " — rank 2/5 in file"


def test_format_edge_provenance_falls_back_when_no_rank():
    """Backward compat: legacy SIMILAR edges with no intra_rank still get
    their evidence string surfaced (unchanged behaviour)."""
    from agent.tools import _format_edge_provenance
    from knowledge.graph import Edge, EdgeType

    edge = Edge(
        source_id="a", target_id="b",
        edge_type=EdgeType.SIMILAR,
        weight=0.84,
        evidence="semantic similarity (0.84)",
    )
    out = _format_edge_provenance(edge)
    assert "semantic similarity" in out
    assert "rank" not in out


def test_format_edge_provenance_references_still_wins_over_rank():
    """References provenance (explicit author intent) must continue to
    take precedence over the intra_rank shortcut."""
    from agent.tools import _format_edge_provenance
    from knowledge.graph import Edge, EdgeType

    edge = Edge(
        source_id="a", target_id="b",
        edge_type=EdgeType.REFERENCES,
        weight=0.9,
        evidence="A → B",
        attributes={
            "link_text": "Foo",
            "link_kind": "wiki",
            "intra_rank": 1,
            "intra_total": 5,  # nonsensical for REFERENCES, but defensive
        },
    )
    out = _format_edge_provenance(edge)
    assert "via 'Foo'" in out
    assert "[wiki]" in out
    assert "rank" not in out


def test_describe_node_renders_intra_rank_string(tmp_path):
    """End-to-end: an intra-file SIMILAR edge with rank/total shows up
    in describe_node's outgoing block as 'rank N/M in file'."""
    from knowledge.graph import EdgeType

    graph = _make_graph(tmp_path)
    _make_chunk(graph, "a", "wiki/quotes.md", "Author A")
    _make_chunk(graph, "b", "wiki/quotes.md", "Author B")
    _add_edge(
        graph, "a", "b", EdgeType.SIMILAR, 0.84,
        attributes={"intra_rank": 1, "intra_total": 5},
    )

    tools = _build_kbtools(graph, tmp_path)
    out = tools.describe_node("wiki/quotes.md", "Author A")
    assert "rank 1/5 in file" in out


def test_build_graph_edges_caps_intra_file_similar_per_node(tmp_path):
    """Integration: hand-feed a vector matrix where node[0] would
    naturally pick up >5 intra-file SIMILAR neighbours, then assert that
    _build_graph_edges caps the output and assigns ranks 1..N."""
    import numpy as np
    import pandas as pd

    from knowledge.graph import KnowledgeGraph, Node, NodeType, EdgeType
    from knowledge.index import KBIndex

    # Build a fake KBIndex shell with a hand-filled graph + a hand-filled
    # table-like df source. _build_graph_edges only needs `self.table`,
    # `self.graph`, and the embeddings inside the table to work.
    kb = KBIndex.__new__(KBIndex)
    kb.graph = KnowledgeGraph(persist_path=tmp_path / "graph.json")

    # 8 chunks, all in the same file (so all candidate edges are SIMILAR).
    n_chunks = 8
    rng = np.random.default_rng(42)
    base = rng.standard_normal(64).astype(np.float32)
    base /= np.linalg.norm(base)
    embs = []
    for i in range(n_chunks):
        # Tight cluster: all very similar to base. Add a tiny per-chunk
        # perturbation so cosine scores are >0.78 but distinguishable.
        v = base + 0.05 * rng.standard_normal(64).astype(np.float32)
        v /= np.linalg.norm(v)
        embs.append(v)
    embs = np.array(embs)

    chunk_ids = []
    for i in range(n_chunks):
        cid = f"c{i}"
        chunk_ids.append(cid)
        node = Node(
            id=cid, node_type=NodeType.CHUNK,
            name=f"section-{i}",
            filename="wiki/quotes.md",
            heading=f"section-{i}",
            attributes={"source": "knowledge"},
        )
        kb.graph.add_node(node)

    df = pd.DataFrame({
        "id": chunk_ids,
        "vector": [e.tolist() for e in embs],
        "source": ["knowledge"] * n_chunks,
        "filename": ["wiki/quotes.md"] * n_chunks,
    })

    class _FakeQuery:
        def __init__(self, df):
            self._df = df

        def select(self, cols):
            return self  # not used by _build_graph_edges

        def to_list(self):
            return self._df.to_dict("records")

    class _FakeTable:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df.copy()

        def search(self):
            return _FakeQuery(self._df)

    kb.table = _FakeTable(df)

    kb._build_graph_edges()

    # Count SIMILAR edges per source node and verify the cap.
    per_node: dict[str, list] = {}
    for e in kb.graph.edges.values():
        if e.edge_type != EdgeType.SIMILAR:
            continue
        per_node.setdefault(e.source_id, []).append(e)

    assert per_node, "expected at least some intra-file SIMILAR edges"
    for src, edges in per_node.items():
        assert len(edges) <= KBIndex.INTRA_FILE_SIMILAR_CAP, (
            f"node {src} has {len(edges)} SIMILAR edges, "
            f"cap is {KBIndex.INTRA_FILE_SIMILAR_CAP}"
        )
        ranks = sorted(e.attributes.get("intra_rank") for e in edges)
        assert ranks == list(range(1, len(edges) + 1)), (
            f"ranks for {src} not contiguous 1..N: {ranks}"
        )
        totals = {e.attributes.get("intra_total") for e in edges}
        assert totals == {len(edges)}, (
            f"intra_total inconsistent for {src}: {totals}"
        )
        # Sanity: ranks line up with weight ordering (rank 1 = strongest).
        by_rank = sorted(edges, key=lambda e: e.attributes["intra_rank"])
        weights = [e.weight for e in by_rank]
        assert weights == sorted(weights, reverse=True), (
            f"rank order does not match weight desc for {src}: {weights}"
        )


def test_intra_file_similar_threshold_bumped_to_078():
    """Sanity guard against accidental regression of the 0.78 floor."""
    from knowledge.index import KBIndex

    assert KBIndex.INTRA_FILE_SIMILAR_THRESHOLD == 0.78
    assert KBIndex.INTRA_FILE_SIMILAR_CAP == 5


# ==========================================================================
# D5 — lint flat-similarity-cluster
# ==========================================================================


def test_lint_flags_flat_similarity_cluster(tmp_path):
    """Build a graph + table where one file has many tightly-clustered
    SIMILAR weights, and assert lint_knowledge surfaces it."""
    import pandas as pd

    from knowledge.graph import EdgeType
    from agent.tools import KBTools

    graph = _make_graph(tmp_path)
    quote_nodes = []
    for i in range(25):
        nid = f"q{i:02d}"
        _make_chunk(graph, nid, "wiki/quotes.md", f"author-{i}")
        quote_nodes.append(nid)

    # 24 intra-file SIMILAR edges all in the band [0.80, 0.84]
    for i, nid in enumerate(quote_nodes[:-1]):
        _add_edge(
            graph, nid, quote_nodes[i + 1], EdgeType.SIMILAR,
            weight=0.80 + (i % 5) * 0.01,  # 0.80..0.84 only
            attributes={"intra_rank": 1, "intra_total": 1},
        )

    # Add one heterogeneous file with wider weight spread (must NOT fire).
    other_nodes = []
    for i in range(25):
        nid = f"o{i:02d}"
        _make_chunk(graph, nid, "wiki/research.md", f"study-{i}")
        other_nodes.append(nid)
    for i, nid in enumerate(other_nodes[:-1]):
        _add_edge(
            graph, nid, other_nodes[i + 1], EdgeType.SIMILAR,
            weight=0.60 + (i * 0.015),  # wide spread 0.60..0.96
        )

    df_rows = []
    for n in graph.nodes.values():
        df_rows.append({
            "id": n.id,
            "filename": n.filename,
            "heading": n.heading,
            "document": "",
            "token_count": 100,
        })
    df = pd.DataFrame(df_rows)

    class _FakeTable:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df.copy()

    class _StubKBIndex:
        def __init__(self, g, df):
            self.graph = g
            self.table = _FakeTable(df)

        def list_indexed_filenames(self):
            return list({r["filename"] for r in df.to_dict("records")})

    tools = KBTools(
        kb_index=_StubKBIndex(graph, df),
        kb_dir=tmp_path / "knowledge",
        canon_dir=tmp_path / "canon",
    )
    out = tools.lint_knowledge()
    assert "Flat-Similarity Clusters" in out
    assert "wiki/quotes.md" in out
    # Heterogeneous file should NOT show up under flat clusters.
    flat_section = out.split("Flat-Similarity Clusters", 1)[1]
    # Trim to just the flat-clusters section — anything under the
    # subsequent fix-it hints block is unrelated.
    flat_section = flat_section.split("Fix-it hints", 1)[0]
    assert "wiki/research.md" not in flat_section
