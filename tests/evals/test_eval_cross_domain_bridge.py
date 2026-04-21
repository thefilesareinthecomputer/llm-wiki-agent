"""Eval: cross_domain_bridge_visible.

Bumping `top_k` in `_build_graph_edges` from 6 to 12 doubles the
candidate set per chunk so cross-file/cross-domain bridges at
0.55-0.65 similarity stop getting crowded out by intra-file matches
at 0.85+.

The fake-embedding test fixture is hash-derived (see conftest.py
`FakeEmbeddingFunction`) — similar texts do not actually cluster.
That means we can't assert on real philosophy↔neuroscience bridges
in unit tests; that requires real Gemini embeddings. What we CAN
guarantee here is the upstream contract:

  1. `_build_graph_edges` runs with top_k=12, not 6.
  2. `KnowledgeGraph.get_neighbors` dedupes (neighbor_id, edge_type),
     so neither A→B nor B→A SIMILAR edges show up twice — a
     prerequisite for the bumped top_k not flooding the agent with
     duplicate noise.
  3. `min_weight` filtering on the three graph tools actually drops
     edges below the threshold.

Together these three guards mean: when real embeddings are in use, a
philosophy chunk that has both an intra-cluster Stoic neighbor at 0.9
AND a neuroscience neighbor at 0.62 will surface BOTH — the bumped
top_k window catches the cross-domain edge, and the dedup keeps the
agent's view clean.
"""

import inspect


def test_build_graph_edges_uses_top_k_12():
    """Source-level guard: regression test for the bumped constant."""
    from knowledge.index import KBIndex

    src = inspect.getsource(KBIndex._build_graph_edges)
    assert "top_k = 12" in src, (
        "Expected top_k = 12 in _build_graph_edges (was 6 before the audit fix). "
        "Lowering this re-introduces the cross-domain crowd-out bug."
    )


def test_get_neighbors_dedupes_pair_and_type(tmp_path):
    """Without dedup, A→B SIMILAR + B→A SIMILAR would surface as two
    rows in graph_neighbors, polluting the agent's view of the graph.
    Verify get_neighbors returns one entry per (neighbor, edge_type)."""
    from knowledge.graph import KnowledgeGraph, Node, Edge, NodeType, EdgeType

    g = KnowledgeGraph(persist_path=tmp_path / "graph.json")
    a = g.add_node(Node(id="a", name="A", node_type=NodeType.CHUNK))
    b = g.add_node(Node(id="b", name="B", node_type=NodeType.CHUNK))

    g.add_edge(Edge(source_id="a", target_id="b",
                    edge_type=EdgeType.SIMILAR, weight=0.7))
    g.add_edge(Edge(source_id="b", target_id="a",
                    edge_type=EdgeType.SIMILAR, weight=0.9))

    neighbors_of_a = g.get_neighbors("a")
    pair_type_keys = [(n.id, e.edge_type) for n, e in neighbors_of_a]
    assert len(pair_type_keys) == 1, (
        f"get_neighbors must dedupe (neighbor, edge_type); "
        f"got {len(pair_type_keys)} entries: {pair_type_keys}"
    )
    assert neighbors_of_a[0][1].weight == 0.9, (
        "Dedup must keep the highest-weight edge."
    )


def test_min_weight_filters_graph_tools(tmp_path):
    """The new min_weight kwarg on graph_neighbors must drop edges
    below the threshold. Builds a tiny graph in memory and exercises
    the filter via the real KBTools API."""
    from knowledge.graph import KnowledgeGraph, Node, Edge, NodeType, EdgeType
    from agent.tools import KBTools
    from knowledge.index import KBIndex

    kb = KBIndex()
    kb.graph = KnowledgeGraph(persist_path=tmp_path / "graph.json")
    kb.graph.add_node(Node(
        id="src", name="src", node_type=NodeType.CHUNK,
        filename="wiki/src.md", heading="src",
    ))
    kb.graph.add_node(Node(
        id="weak", name="weak", node_type=NodeType.CHUNK,
        filename="wiki/weak.md", heading="weak",
    ))
    kb.graph.add_node(Node(
        id="strong", name="strong", node_type=NodeType.CHUNK,
        filename="wiki/strong.md", heading="strong",
    ))
    kb.graph.add_edge(Edge(
        source_id="src", target_id="weak",
        edge_type=EdgeType.SIMILAR, weight=0.30,
    ))
    kb.graph.add_edge(Edge(
        source_id="src", target_id="strong",
        edge_type=EdgeType.SIMILAR, weight=0.85,
    ))

    tools = KBTools(kb_index=kb)

    no_filter = tools.graph_neighbors("wiki/src.md", "src")
    assert "weak" in no_filter and "strong" in no_filter

    filtered = tools.graph_neighbors("wiki/src.md", "src", min_weight=0.5)
    assert "strong" in filtered, "min_weight must keep edges above threshold."
    assert "weak" not in filtered, "min_weight must drop edges below threshold."
