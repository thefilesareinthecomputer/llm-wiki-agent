"""B4 — describe_node + edge provenance visibility for the agent.

Covers:

1. ``describe_node(filename, heading)`` returns a single descriptor with
   tier, summary, outgoing + incoming edges grouped by EdgeType, and
   ``_format_edge_provenance`` annotations on every edge.
2. References edges with ``link_text`` / ``link_kind`` attributes show
   their ``via 'text' [kind]`` provenance.
3. The orphan branch fires when a chunk has zero edges.
4. ``describe_node`` is registered in TOOL_CLASSES + build_tool_registry
   under the explore budget class.
5. ``graph_neighbors``, ``graph_traverse`` and ``graph_search`` already
   render provenance — guarded here against regression.
"""

from __future__ import annotations

import pytest


def _build_graph_with_provenance(persist_path=None):
    """Hand-build a tiny KnowledgeGraph with three CHUNK nodes:

      - A  - has an outgoing REFERENCES edge to B (link_text='Foo' [wiki]),
            an outgoing SIMILAR edge to C, and an incoming SIMILAR from C.
      - B  - REFERENCES target only.
      - C  - SIMILAR neighbour only.
      - D  - orphan (no edges).
    """
    from knowledge.graph import (
        Edge,
        EdgeType,
        KnowledgeGraph,
        Node,
        NodeType,
    )

    if persist_path is None:
        import tempfile
        persist_path = (
            __import__("pathlib").Path(tempfile.mkdtemp()) / "graph.json"
        )
    graph = KnowledgeGraph(persist_path=persist_path)

    a = Node(
        id="a", node_type=NodeType.CHUNK,
        name="A heading",
        filename="wiki/a.md",
        heading="A heading",
        summary="Section A summary.",
        attributes={"source": "knowledge", "token_count": 1234},
    )
    b = Node(
        id="b", node_type=NodeType.CHUNK,
        name="B heading",
        filename="wiki/b.md",
        heading="B heading",
        summary="Section B summary.",
        attributes={"source": "knowledge", "token_count": 800},
    )
    c = Node(
        id="c", node_type=NodeType.CHUNK,
        name="C heading",
        filename="wiki/c.md",
        heading="C heading",
        summary="",
        attributes={"source": "knowledge", "token_count": 500},
    )
    d = Node(
        id="d", node_type=NodeType.CHUNK,
        name="Lonely",
        filename="wiki/lonely.md",
        heading="Lonely",
        summary="",
        attributes={"source": "knowledge", "token_count": 100},
    )
    for n in (a, b, c, d):
        graph.add_node(n)

    graph.add_edge(Edge(
        source_id="a", target_id="b",
        edge_type=EdgeType.REFERENCES,
        weight=0.9,
        evidence="A links to B via [[Foo]]",
        attributes={
            "link_text": "Foo",
            "link_kind": "wiki",
            "target_anchor": "intro",
        },
    ))
    graph.add_edge(Edge(
        source_id="a", target_id="c",
        edge_type=EdgeType.SIMILAR,
        weight=0.6,
        evidence="cosine 0.6",
    ))
    graph.add_edge(Edge(
        source_id="c", target_id="a",
        edge_type=EdgeType.SIMILAR,
        weight=0.55,
        evidence="cosine 0.55",
    ))
    return graph


def _build_kbtools_with_graph(graph, tmp_path):
    from agent.tools import KBTools

    class _StubKBIndex:
        def __init__(self, g):
            self.graph = g

    return KBTools(
        kb_index=_StubKBIndex(graph),
        kb_dir=tmp_path / "knowledge",
        canon_dir=tmp_path / "canon",
    )


# ---------- describe_node behaviour ----------------------------------------

def test_describe_node_renders_tier_summary_and_degree(tmp_path):
    graph = _build_graph_with_provenance(tmp_path / "graph.json")
    tools = _build_kbtools_with_graph(graph, tmp_path)

    out = tools.describe_node("wiki/a.md", "A heading")
    assert "# Node: knowledge:wiki/a.md" in out
    assert "tier: wiki" in out
    assert "Section A summary." in out
    assert "tokens: ~1,234" in out
    # A has 2 out + 1 in
    assert "out=2" in out
    assert "in=1" in out


def test_describe_node_renders_outgoing_with_references_provenance(tmp_path):
    graph = _build_graph_with_provenance(tmp_path / "graph.json")
    tools = _build_kbtools_with_graph(graph, tmp_path)

    out = tools.describe_node("wiki/a.md", "A heading")
    assert "## Outgoing" in out
    assert "[references]" in out
    assert "B heading" in out
    # _format_edge_provenance must have surfaced the wiki-link metadata.
    assert "via 'Foo'" in out
    assert "[wiki]" in out
    assert "#intro" in out
    # SIMILAR edge present, with its weight.
    assert "[similar]" in out
    assert "weight: 0.60" in out or "weight: 0.55" in out


def test_describe_node_renders_incoming_block(tmp_path):
    graph = _build_graph_with_provenance(tmp_path / "graph.json")
    tools = _build_kbtools_with_graph(graph, tmp_path)

    out = tools.describe_node("wiki/a.md", "A heading")
    assert "## Incoming" in out
    # The incoming edge from C must surface
    assert "C heading" in out


def test_describe_node_orphan_branch(tmp_path):
    graph = _build_graph_with_provenance(tmp_path / "graph.json")
    tools = _build_kbtools_with_graph(graph, tmp_path)

    out = tools.describe_node("wiki/lonely.md", "Lonely")
    assert "## (orphan)" in out
    assert "No edges" in out


def test_describe_node_unknown_returns_clear_error(tmp_path):
    graph = _build_graph_with_provenance(tmp_path / "graph.json")
    tools = _build_kbtools_with_graph(graph, tmp_path)

    out = tools.describe_node("wiki/does-not-exist.md", "missing")
    assert "No sections found" in out


def test_describe_node_no_graph(tmp_path):
    from agent.tools import KBTools

    class _StubKBIndex:
        graph = None

    tools = KBTools(
        kb_index=_StubKBIndex(),
        kb_dir=tmp_path / "knowledge",
        canon_dir=tmp_path / "canon",
    )
    out = tools.describe_node("wiki/a.md", "A")
    assert out == "No knowledge graph available."


def test_describe_node_caps_edges_per_type(tmp_path):
    """Hammer one edge type past the per-type cap and verify truncation."""
    from knowledge.graph import (
        Edge,
        EdgeType,
        KnowledgeGraph,
        Node,
        NodeType,
    )
    from agent.tools import KBTools

    graph = KnowledgeGraph(persist_path=tmp_path / "graph.json")
    hub = Node(
        id="hub", node_type=NodeType.CHUNK,
        name="hub", filename="wiki/hub.md", heading="hub",
        attributes={"source": "knowledge"},
    )
    graph.add_node(hub)
    cap = KBTools.DESCRIBE_NODE_MAX_EDGES_PER_TYPE
    for i in range(cap + 5):
        nid = f"nbr{i}"
        graph.add_node(Node(
            id=nid, node_type=NodeType.CHUNK,
            name=f"nbr{i}", filename=f"wiki/n{i}.md", heading=f"nbr{i}",
            attributes={"source": "knowledge"},
        ))
        graph.add_edge(Edge(
            source_id="hub", target_id=nid,
            edge_type=EdgeType.SIMILAR,
            weight=0.5 + (i / 100.0),
        ))

    tools = _build_kbtools_with_graph(graph, tmp_path)
    out = tools.describe_node("wiki/hub.md", "hub")
    assert f"and 5 more" in out
    assert "graph_neighbors" in out  # must hint how to page through


# ---------- registry + budget wiring ---------------------------------------

def test_describe_node_in_registry_and_explore_class(tmp_path):
    from agent.tools import (
        KBTools,
        TOOL_CLASSES,
        build_tool_registry,
        class_for_tool,
    )

    class _StubKBIndex:
        graph = None

    tools = KBTools(_StubKBIndex(), tmp_path / "k", tmp_path / "c")
    registry = build_tool_registry(tools)

    assert "describe_node" in registry
    assert callable(registry["describe_node"])
    assert TOOL_CLASSES["describe_node"] == "explore"
    assert class_for_tool("describe_node") == "explore"


# ---------- regression: provenance still rendered by other graph tools -----

def test_graph_neighbors_renders_provenance(tmp_path):
    graph = _build_graph_with_provenance(tmp_path / "graph.json")
    tools = _build_kbtools_with_graph(graph, tmp_path)
    out = tools.graph_neighbors("wiki/a.md", "A heading")
    assert "via 'Foo'" in out
    assert "[wiki]" in out


def test_graph_traverse_renders_provenance(tmp_path):
    graph = _build_graph_with_provenance(tmp_path / "graph.json")
    tools = _build_kbtools_with_graph(graph, tmp_path)
    out = tools.graph_traverse("wiki/a.md", "A heading", depth=1)
    # Either via the references edge to B or the similar edge to C.
    assert "via 'Foo'" in out or "[similar]" in out
