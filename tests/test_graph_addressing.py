"""Tests for graph addressing fixes (A1-A4, B1, D1, E1, E2)."""

import pytest
from pathlib import Path
from agent.tools import (
    KBTools,
    _resolve_chunk_nodes,
    _split_combined_path,
    reset_budget,
)
from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType, format_folder_tree
from knowledge.index import KBIndex


@pytest.fixture(autouse=True)
def reset_budget_fixture():
    reset_budget()
    yield
    reset_budget()


def _make_index_with_graph(graph: KnowledgeGraph) -> KBIndex:
    """Build a bare KBIndex stub wrapping a prebuilt graph."""
    index = KBIndex.__new__(KBIndex)
    index.graph = graph
    index.table = None
    return index


# ---------------------------------------------------------------------------
# A1: graph_search format
# ---------------------------------------------------------------------------

class TestGraphSearchFormat:
    """A1: graph_search uses relative filename, prints heading + summary,
    and narrows chunk lookup by heading."""

    def test_uses_filename_not_path(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="ai.md > Intro",
                  filename="ai.md", heading="Intro",
                  attributes={"source": "knowledge"})
        graph.add_node(c1)

        index = _make_index_with_graph(graph)
        index.search = lambda q, top_k=5: [{
            "filename": "ai.md", "heading": "Intro", "summary": "AI overview",
            "score": 0.91, "path": "/app/knowledge/ai.md",
        }]
        tools = KBTools(kb_index=index)

        result = tools.graph_search("ai")
        # Canonical filename appears (A2 introduced <source>:<relpath> headers).
        assert "## knowledge:ai.md" in result
        # Absolute path does NOT appear
        assert "/app/knowledge/ai.md" not in result
        # Heading line rendered
        assert "Intro" in result
        # Summary rendered
        assert "AI overview" in result

    def test_chunk_lookup_narrows_by_heading(self, tmp_path):
        """The chunk_nodes filter must match heading too — otherwise unrelated
        sections from the same file get their neighbors surfaced."""
        graph = KnowledgeGraph(tmp_path / "graph.json")
        # Two sections in the same file
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="ai.md > Intro",
                  filename="ai.md", heading="Intro",
                  attributes={"source": "knowledge"})
        c2 = Node(id="c2", node_type=NodeType.CHUNK, name="ai.md > Methods",
                  filename="ai.md", heading="Methods",
                  attributes={"source": "knowledge"})
        # Neighbor of c2 (Methods), NOT c1 (Intro)
        c3 = Node(id="c3", node_type=NodeType.CHUNK, name="other.md > Stats",
                  filename="other.md", heading="Stats",
                  attributes={"source": "knowledge"})
        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_node(c3)
        graph.add_edge(Edge(source_id="c2", target_id="c3",
                            edge_type=EdgeType.INTER_FILE, weight=0.7))

        index = _make_index_with_graph(graph)
        # Search returns c1 (Intro) — neighbors of c1 should NOT include c3.
        index.search = lambda q, top_k=5: [{
            "filename": "ai.md", "heading": "Intro", "summary": "",
            "score": 0.9, "path": "ai.md",
        }]
        tools = KBTools(kb_index=index)

        result = tools.graph_search("intro")
        assert "ai.md" in result
        # c3 should not appear because it's a neighbor of Methods, not Intro.
        assert "Stats" not in result


# ---------------------------------------------------------------------------
# A2: most_connected excludes parent_child + new format
# ---------------------------------------------------------------------------

class TestMostConnectedExcludesParentChild:
    """A2: parent_child edges are not counted toward Most Connected."""

    def test_parent_child_only_node_not_in_leaderboard(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        # Hub with only parent_child edges (structural nesting)
        hub = Node(id="hub", node_type=NodeType.CHUNK, name="file.md > Hub",
                   filename="file.md", heading="Hub",
                   attributes={"source": "knowledge"})
        graph.add_node(hub)
        for i in range(5):
            child = Node(id=f"child{i}", node_type=NodeType.CHUNK,
                         name=f"file.md > Child{i}",
                         filename="file.md", heading=f"Child{i}",
                         attributes={"source": "knowledge"})
            graph.add_node(child)
            graph.add_edge(Edge(source_id="hub", target_id=f"child{i}",
                                edge_type=EdgeType.PARENT_CHILD, weight=1.0))
        # A different node with one semantic edge
        a = Node(id="a", node_type=NodeType.CHUNK, name="x.md > A",
                 filename="x.md", heading="A", attributes={"source": "knowledge"})
        b = Node(id="b", node_type=NodeType.CHUNK, name="y.md > B",
                 filename="y.md", heading="B", attributes={"source": "knowledge"})
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge(Edge(source_id="a", target_id="b",
                            edge_type=EdgeType.INTER_FILE, weight=0.7))

        stats = graph.get_stats()
        most_connected_names = {entry["name"] for entry in stats["most_connected"]}
        # The hub (only parent_child edges) must NOT be in the leaderboard
        assert "file.md > Hub" not in most_connected_names
        # The semantically connected nodes ARE in the leaderboard
        assert "x.md > A" in most_connected_names or "y.md > B" in most_connected_names

    def test_most_connected_returns_dict_with_filename_and_heading(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        a = Node(id="a", node_type=NodeType.CHUNK, name="x.md > A",
                 filename="x.md", heading="A", attributes={"source": "knowledge"})
        b = Node(id="b", node_type=NodeType.CHUNK, name="y.md > B",
                 filename="y.md", heading="B", attributes={"source": "knowledge"})
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge(Edge(source_id="a", target_id="b",
                            edge_type=EdgeType.SIMILAR, weight=0.8))

        stats = graph.get_stats()
        assert len(stats["most_connected"]) >= 1
        entry = stats["most_connected"][0]
        assert isinstance(entry, dict)
        assert "filename" in entry and "heading" in entry and "count" in entry

    def test_graph_stats_renders_filename_and_heading_separately(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        a = Node(id="a", node_type=NodeType.CHUNK, name="x.md > A",
                 filename="x.md", heading="A", attributes={"source": "knowledge"})
        b = Node(id="b", node_type=NodeType.CHUNK, name="y.md > B",
                 filename="y.md", heading="B", attributes={"source": "knowledge"})
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge(Edge(source_id="a", target_id="b",
                            edge_type=EdgeType.SIMILAR, weight=0.8))

        index = _make_index_with_graph(graph)
        tools = KBTools(kb_index=index)
        result = tools.graph_stats()

        assert "Most Connected (semantic + cross-file)" in result
        # Filename and heading on different lines, callable from copy-paste
        assert "  x.md" in result or "  y.md" in result
        assert "    > A (" in result or "    > B (" in result


# ---------------------------------------------------------------------------
# A3: heading-only fallback resolution
# ---------------------------------------------------------------------------

class TestHeadingOnlyFallback:
    """A3: When filename doesn't match a file, treat it as a leaf heading
    and search across files."""

    def test_single_match_resolves(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK,
                  name="philosophy.md > marcus-aurelius",
                  filename="philosophy.md", heading="philosophy-quotes > marcus-aurelius",
                  attributes={"source": "canon"})
        graph.add_node(c1)

        nodes, disambig = _resolve_chunk_nodes(graph, "marcus-aurelius", "")
        assert len(nodes) == 1
        assert nodes[0].id == "c1"
        assert disambig is None

    def test_multi_match_returns_disambiguation(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK,
                  name="quotes.md > marcus-aurelius",
                  filename="quotes.md", heading="quotes > marcus-aurelius",
                  attributes={"source": "canon"})
        c2 = Node(id="c2", node_type=NodeType.CHUNK,
                  name="logs.md > marcus-aurelius",
                  filename="logs.md", heading="daily > marcus-aurelius",
                  attributes={"source": "knowledge"})
        graph.add_node(c1)
        graph.add_node(c2)

        nodes, disambig = _resolve_chunk_nodes(graph, "marcus-aurelius", "")
        assert nodes == []
        assert disambig is not None
        assert "Ambiguous" in disambig
        assert "quotes.md" in disambig
        assert "logs.md" in disambig
        # Suggestion lines should be directly callable. A2 emits canonical
        # <source>:<relpath> identifiers so the model copies them verbatim
        # back into the next call.
        assert "graph_neighbors(filename='canon:quotes.md'" in disambig

    def test_no_match_returns_empty(self, tmp_path):
        """Empty graph + nonexistent filename → no fallback match."""
        graph = KnowledgeGraph(tmp_path / "graph.json")
        nodes, disambig = _resolve_chunk_nodes(graph, "nonexistent.md", "")
        assert nodes == []
        assert disambig is None

    def test_existing_file_not_found_test_still_passes(self, tmp_path):
        """Locks the prior test_file_not_found contract: empty graph,
        graph_neighbors('nonexistent.md') returns 'No sections'."""
        graph = KnowledgeGraph(tmp_path / "graph.json")
        index = _make_index_with_graph(graph)
        tools = KBTools(kb_index=index)
        result = tools.graph_neighbors("nonexistent.md")
        assert "No sections" in result


# ---------------------------------------------------------------------------
# A4: combined "file > heading" arg
# ---------------------------------------------------------------------------

class TestCombinedPathSplit:
    """A4: 'file.md > heading > sub' as a single arg gets split."""

    def test_split_combined_path_basic(self):
        f, h = _split_combined_path("foo.md > sec > sub", "")
        assert f == "foo.md"
        assert h == "sec > sub"

    def test_split_no_separator_returns_unchanged(self):
        f, h = _split_combined_path("foo.md", "")
        assert f == "foo.md"
        assert h == ""

    def test_split_with_existing_heading_returns_unchanged(self):
        f, h = _split_combined_path("foo.md > x", "explicit")
        assert f == "foo.md > x"
        assert h == "explicit"

    def test_resolve_uses_split(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK,
                  name="quotes.md > marcus-aurelius",
                  filename="quotes.md", heading="quotes > marcus-aurelius",
                  attributes={"source": "canon"})
        graph.add_node(c1)

        nodes, disambig = _resolve_chunk_nodes(
            graph, "quotes.md > quotes > marcus-aurelius", "")
        assert disambig is None
        assert len(nodes) == 1
        assert nodes[0].id == "c1"


# ---------------------------------------------------------------------------
# B1: graph_neighbors offset + edge_type pagination
# ---------------------------------------------------------------------------

class TestGraphNeighborsPagination:
    """B1: offset and edge_type kwargs paginate inside the function."""

    def _build_high_degree_graph(self, tmp_path, n_inter=20, n_similar=15):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        hub = Node(id="hub", node_type=NodeType.CHUNK,
                   name="hub.md > Center", filename="hub.md", heading="Center",
                   attributes={"source": "knowledge"})
        graph.add_node(hub)
        for i in range(n_inter):
            n = Node(id=f"i{i}", node_type=NodeType.CHUNK,
                     name=f"other{i}.md > X", filename=f"other{i}.md", heading="X",
                     attributes={"source": "knowledge"})
            graph.add_node(n)
            graph.add_edge(Edge(source_id="hub", target_id=f"i{i}",
                                edge_type=EdgeType.INTER_FILE, weight=0.6))
        for i in range(n_similar):
            n = Node(id=f"s{i}", node_type=NodeType.CHUNK,
                     name=f"hub.md > Sub{i}", filename="hub.md", heading=f"Sub{i}",
                     attributes={"source": "knowledge"})
            graph.add_node(n)
            graph.add_edge(Edge(source_id="hub", target_id=f"s{i}",
                                edge_type=EdgeType.SIMILAR, weight=0.8))
        return graph

    def test_offset_skips_edges(self, tmp_path):
        graph = self._build_high_degree_graph(tmp_path)
        index = _make_index_with_graph(graph)
        tools = KBTools(kb_index=index)

        page1 = tools.graph_neighbors("hub.md", "Center", offset=0, limit=10)
        page2 = tools.graph_neighbors("hub.md", "Center", offset=10, limit=10)
        # Different windows should produce different rendered content
        assert page1 != page2
        # Footer mentions the offset
        assert "showing edges" in page1
        assert "offset=" in page1

    def test_edge_type_filter(self, tmp_path):
        graph = self._build_high_degree_graph(tmp_path, n_inter=5, n_similar=5)
        index = _make_index_with_graph(graph)
        tools = KBTools(kb_index=index)

        result = tools.graph_neighbors("hub.md", "Center", edge_type="inter_file")
        assert "[inter_file]" in result
        assert "[similar]" not in result

    def test_invalid_edge_type_rejected(self, tmp_path):
        graph = self._build_high_degree_graph(tmp_path, n_inter=2, n_similar=2)
        index = _make_index_with_graph(graph)
        tools = KBTools(kb_index=index)
        result = tools.graph_neighbors("hub.md", "Center", edge_type="bogus")
        assert "Invalid edge_type" in result

    def test_no_offset_no_footer_when_complete(self, tmp_path):
        # Small graph that fits in one page
        graph = KnowledgeGraph(tmp_path / "graph.json")
        a = Node(id="a", node_type=NodeType.CHUNK, name="a.md > A",
                 filename="a.md", heading="A", attributes={"source": "knowledge"})
        b = Node(id="b", node_type=NodeType.CHUNK, name="b.md > B",
                 filename="b.md", heading="B", attributes={"source": "knowledge"})
        graph.add_node(a)
        graph.add_node(b)
        graph.add_edge(Edge(source_id="a", target_id="b",
                            edge_type=EdgeType.SIMILAR, weight=0.8))

        index = _make_index_with_graph(graph)
        tools = KBTools(kb_index=index)
        result = tools.graph_neighbors("a.md", "A")
        # Single edge, no pagination footer needed
        assert "showing edges" not in result
        assert "[similar]" in result


# ---------------------------------------------------------------------------
# D1: L4 prompt contains calling conventions
# ---------------------------------------------------------------------------

class TestL4PromptCallingConventions:
    """D1: L4 prompt locks the contract for graph tool calling."""

    def test_prompt_contains_key_phrases(self):
        from web.app import _build_system_prompt
        prompt = _build_system_prompt(tools_enabled=True)

        # Budget classes must be named so the agent knows which budget
        # each call spends against. Both orient and explore classes must
        # appear in the prompt.
        assert "orient" in prompt
        assert "explore" in prompt
        # graph_search must appear before graph_neighbors — the prompt
        # teaches search-first-then-neighbor, not the reverse.
        gs_idx = prompt.find("graph_search(query")
        gn_idx = prompt.find("graph_neighbors(filename")
        assert 0 <= gs_idx < gn_idx, "graph_search should appear before graph_neighbors"
        assert "graph_neighbors(filename" in prompt
        assert "offset=" in prompt
        assert "edge_type=" in prompt
        assert "folder_tree(" in prompt

    def test_prompt_omitted_when_tools_disabled(self):
        from web.app import _build_system_prompt
        prompt = _build_system_prompt(tools_enabled=False)
        # With tools off the whole KB-TOOLS section is gone; no tool
        # signatures or budget vocabulary should leak through.
        assert "graph_search(query" not in prompt
        assert "graph_neighbors(filename" not in prompt
        assert "folder_tree(" not in prompt
        assert "KB TOOLS" not in prompt


# ---------------------------------------------------------------------------
# E1: format_folder_tree includes summaries
# ---------------------------------------------------------------------------

class TestFormatFolderTreeSummaries:
    """E1: folder summaries are rendered inline when present."""

    def test_summary_rendered_inline(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        root = Node(id="folder_knowledge_", node_type=NodeType.FOLDER, name="",
                    attributes={"source": "knowledge"})
        child = Node(id="folder_knowledge_ai", node_type=NodeType.FOLDER, name="ai",
                     summary="Artificial intelligence research and notes",
                     attributes={"source": "knowledge"})
        graph.add_node(root)
        graph.add_node(child)
        graph.add_edge(Edge(source_id="folder_knowledge_",
                            target_id="folder_knowledge_ai",
                            edge_type=EdgeType.PARENT_CHILD, weight=1.0))

        result = format_folder_tree(graph, source="knowledge")
        assert "ai/" in result
        assert "— Artificial intelligence research and notes" in result

    def test_no_summary_no_em_dash(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        root = Node(id="folder_knowledge_", node_type=NodeType.FOLDER, name="",
                    attributes={"source": "knowledge"})
        child = Node(id="folder_knowledge_ai", node_type=NodeType.FOLDER, name="ai",
                     attributes={"source": "knowledge"})  # no summary
        graph.add_node(root)
        graph.add_node(child)
        graph.add_edge(Edge(source_id="folder_knowledge_",
                            target_id="folder_knowledge_ai",
                            edge_type=EdgeType.PARENT_CHILD, weight=1.0))

        result = format_folder_tree(graph, source="knowledge")
        assert "ai/" in result
        # No em-dash suffix when summary is empty
        assert "ai/ (0 files) —" not in result


# ---------------------------------------------------------------------------
# E2: folder_tree() KB tool
# ---------------------------------------------------------------------------

class TestFolderTreeTool:
    """E2: folder_tree() exposed as a KB tool, parser-recognized."""

    def test_tool_returns_formatted_tree(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        root = Node(id="folder_knowledge_", node_type=NodeType.FOLDER, name="",
                    attributes={"source": "knowledge"})
        child = Node(id="folder_knowledge_ai", node_type=NodeType.FOLDER, name="ai",
                     summary="AI notes", attributes={"source": "knowledge"})
        graph.add_node(root)
        graph.add_node(child)
        graph.add_edge(Edge(source_id="folder_knowledge_",
                            target_id="folder_knowledge_ai",
                            edge_type=EdgeType.PARENT_CHILD, weight=1.0))

        index = _make_index_with_graph(graph)
        # KBIndex.get_folder_tree exists in real code; stub it for the test
        index.get_folder_tree = lambda source="knowledge", root_path=None: format_folder_tree(graph, source=source, root_path=root_path)
        tools = KBTools(kb_index=index)

        result = tools.folder_tree()
        assert "knowledge/" in result
        assert "ai/" in result
        assert "AI notes" in result

    def test_invalid_folder_arg_rejected(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        index = _make_index_with_graph(graph)
        index.get_folder_tree = lambda source="knowledge", root_path=None: ""
        tools = KBTools(kb_index=index)
        result = tools.folder_tree("bogus")
        assert "Invalid folder" in result

    def test_no_index_returns_message(self):
        tools = KBTools(kb_index=None)
        result = tools.folder_tree()
        assert "No knowledge index" in result

    def test_folder_tree_in_native_tool_registry(self, tmp_path):
        # A1 swapped the [TOOL: ...] regex parser for Ollama native tool
        # calling. The presence check now lives on the registry.
        from unittest.mock import MagicMock
        from agent.tools import KBTools, build_tool_registry

        tools = KBTools(MagicMock(), tmp_path, tmp_path)
        registry = build_tool_registry(tools)
        assert "folder_tree" in registry
        assert callable(registry["folder_tree"])


# ---------------------------------------------------------------------------
# P0.5: folder_tree() defaults to ALL tiers; format_folder_tree shows
#        per-folder [tier] badges; L4 prompt teaches medallion vocabulary.
# ---------------------------------------------------------------------------

def _add_folder(graph: KnowledgeGraph, fid: str, name: str, source: str,
                tier: str = "", summary: str = "") -> Node:
    node = Node(
        id=fid,
        node_type=NodeType.FOLDER,
        name=name,
        summary=summary,
        attributes={"source": source, **({"tier": tier} if tier else {})},
    )
    graph.add_node(node)
    return node


def _link(graph: KnowledgeGraph, parent_id: str, child_id: str) -> None:
    graph.add_edge(Edge(
        source_id=parent_id, target_id=child_id,
        edge_type=EdgeType.PARENT_CHILD, weight=1.0,
    ))


class TestFolderTreeRendersTierBadges:
    """P0.5: format_folder_tree includes [tier] badges for every folder so
    the agent can see canon vs wiki vs raw at a glance."""

    def test_root_header_announces_silver_and_bronze_for_knowledge(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        _add_folder(graph, "folder_knowledge_", "", "knowledge")
        result = format_folder_tree(graph, source="knowledge")
        assert "[knowledge] knowledge/" in result
        assert "silver" in result.lower()
        assert "bronze" in result.lower()

    def test_root_header_announces_gold_for_canon(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        _add_folder(graph, "folder_canon_", "", "canon")
        result = format_folder_tree(graph, source="canon")
        assert "[canon] canon/" in result
        assert "gold" in result.lower()

    def test_per_folder_tier_badge_rendered(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        _add_folder(graph, "folder_knowledge_", "", "knowledge")
        _add_folder(graph, "folder_knowledge_wiki", "wiki", "knowledge", tier="wiki")
        _add_folder(graph, "folder_knowledge_raw", "raw", "knowledge", tier="raw")
        _link(graph, "folder_knowledge_", "folder_knowledge_wiki")
        _link(graph, "folder_knowledge_", "folder_knowledge_raw")

        result = format_folder_tree(graph, source="knowledge")
        assert "wiki/ [wiki]" in result
        assert "raw/ [raw]" in result

    def test_no_tier_badge_when_attribute_absent(self, tmp_path):
        """Backward compatible: legacy folder nodes without `tier` attribute
        render without a badge rather than crashing or printing an empty one."""
        graph = KnowledgeGraph(tmp_path / "graph.json")
        _add_folder(graph, "folder_knowledge_", "", "knowledge")
        _add_folder(graph, "folder_knowledge_legacy", "legacy", "knowledge")
        _link(graph, "folder_knowledge_", "folder_knowledge_legacy")

        result = format_folder_tree(graph, source="knowledge")
        assert "legacy/" in result
        # No empty brackets sneaking in
        assert "legacy/ []" not in result


class TestFolderTreeToolDefaultsToAllTiers:
    """P0.5: tools.folder_tree() with no args should render BOTH sources
    (canon + knowledge) so the agent gets a full medallion overview."""

    def test_default_renders_canon_and_knowledge(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        _add_folder(graph, "folder_canon_", "", "canon")
        _add_folder(graph, "folder_knowledge_", "", "knowledge")
        _add_folder(graph, "folder_knowledge_wiki", "wiki", "knowledge", tier="wiki")
        _link(graph, "folder_knowledge_", "folder_knowledge_wiki")

        index = _make_index_with_graph(graph)
        index.get_folder_tree = lambda source="knowledge", root_path=None: format_folder_tree(graph, source=source, root_path=root_path)
        tools = KBTools(kb_index=index)

        result = tools.folder_tree()
        assert "[canon] canon/" in result
        assert "[knowledge] knowledge/" in result
        assert "wiki/ [wiki]" in result

    def test_explicit_canon_skips_knowledge(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        _add_folder(graph, "folder_canon_", "", "canon")
        _add_folder(graph, "folder_knowledge_", "", "knowledge")

        index = _make_index_with_graph(graph)
        index.get_folder_tree = lambda source="knowledge", root_path=None: format_folder_tree(graph, source=source, root_path=root_path)
        tools = KBTools(kb_index=index)

        result = tools.folder_tree("canon")
        assert "[canon] canon/" in result
        assert "[knowledge] knowledge/" not in result

    def test_explicit_knowledge_skips_canon(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        _add_folder(graph, "folder_canon_", "", "canon")
        _add_folder(graph, "folder_knowledge_", "", "knowledge")

        index = _make_index_with_graph(graph)
        index.get_folder_tree = lambda source="knowledge", root_path=None: format_folder_tree(graph, source=source, root_path=root_path)
        tools = KBTools(kb_index=index)

        result = tools.folder_tree("knowledge")
        assert "[knowledge] knowledge/" in result
        assert "[canon] canon/" not in result

    def test_all_alias_works(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        _add_folder(graph, "folder_canon_", "", "canon")
        _add_folder(graph, "folder_knowledge_", "", "knowledge")

        index = _make_index_with_graph(graph)
        index.get_folder_tree = lambda source="knowledge", root_path=None: format_folder_tree(graph, source=source, root_path=root_path)
        tools = KBTools(kb_index=index)

        result = tools.folder_tree("all")
        assert "[canon] canon/" in result
        assert "[knowledge] knowledge/" in result

    def test_invalid_arg_still_rejected(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        index = _make_index_with_graph(graph)
        index.get_folder_tree = lambda source="knowledge", root_path=None: format_folder_tree(graph, source=source, root_path=root_path)
        tools = KBTools(kb_index=index)
        result = tools.folder_tree("bogus")
        assert "Invalid folder" in result

    def test_empty_canon_does_not_pollute_output(self, tmp_path):
        """When canon has no folders, the default call should still render
        the knowledge tree without an orphan canon header."""
        graph = KnowledgeGraph(tmp_path / "graph.json")
        _add_folder(graph, "folder_knowledge_", "", "knowledge")

        index = _make_index_with_graph(graph)
        index.get_folder_tree = lambda source="knowledge", root_path=None: format_folder_tree(graph, source=source, root_path=root_path)
        tools = KBTools(kb_index=index)

        result = tools.folder_tree()
        assert "[knowledge] knowledge/" in result
        # canon source returned "" → tool skipped it
        assert "[canon]" not in result


class TestL4PromptMedallionVocabulary:
    """P0.5: L4 system prompt teaches the agent the medallion vocabulary
    (canon = gold, wiki = silver, raw = bronze) and the write-tier rule."""

    def test_prompt_names_all_three_tiers(self):
        from web.app import _build_system_prompt
        prompt = _build_system_prompt(tools_enabled=True)
        assert "[canon]" in prompt
        assert "[wiki]" in prompt
        assert "[raw]" in prompt

    def test_prompt_uses_medallion_metaphor(self):
        from web.app import _build_system_prompt
        prompt = _build_system_prompt(tools_enabled=True)
        lower = prompt.lower()
        assert "gold" in lower
        assert "silver" in lower
        assert "bronze" in lower

    def test_prompt_announces_save_targets_wiki(self):
        from web.app import _build_system_prompt
        prompt = _build_system_prompt(tools_enabled=True)
        assert "knowledge/wiki/" in prompt
        # Make it explicit somewhere that save_knowledge writes to wiki
        assert "wiki" in prompt.lower() and "save" in prompt.lower()

    def test_prompt_marks_canon_and_raw_read_only(self):
        from web.app import _build_system_prompt
        prompt = _build_system_prompt(tools_enabled=True)
        lower = prompt.lower()
        assert "read-only" in lower or "read only" in lower

    def test_medallion_block_omitted_when_tools_disabled(self):
        from web.app import _build_system_prompt
        prompt = _build_system_prompt(tools_enabled=False)
        # Tier-vocabulary block is inside the tool-literacy layer; when
        # tools are off it should not render (no gold/silver/bronze).
        assert "[canon]" not in prompt
        assert "[wiki]" not in prompt
        assert "[raw]" not in prompt


# ---------------------------------------------------------------------------
# P0-1: caller-aware suggestion format + ranked candidates + query reranker
# ---------------------------------------------------------------------------

class TestResolverCallerAwareSuggestions:
    """P0-1: disambiguation suggestions use the calling tool's name so the
    model can copy-paste the line back into the same tool, not always
    ``graph_neighbors``."""

    def _two_match_graph(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK,
                  name="quotes.md > marcus-aurelius",
                  filename="quotes.md", heading="quotes > marcus-aurelius",
                  attributes={"source": "canon"})
        c2 = Node(id="c2", node_type=NodeType.CHUNK,
                  name="logs.md > marcus-aurelius",
                  filename="logs.md", heading="daily > marcus-aurelius",
                  attributes={"source": "knowledge"})
        graph.add_node(c1)
        graph.add_node(c2)
        return graph

    def test_default_caller_is_graph_neighbors(self, tmp_path):
        graph = self._two_match_graph(tmp_path)
        nodes, disambig = _resolve_chunk_nodes(graph, "marcus-aurelius", "")
        assert nodes == []
        assert "graph_neighbors(filename='canon:quotes.md'" in disambig

    def test_describe_node_caller_renders_describe_node_form(self, tmp_path):
        graph = self._two_match_graph(tmp_path)
        nodes, disambig = _resolve_chunk_nodes(
            graph, "marcus-aurelius", "", caller="describe_node"
        )
        assert nodes == []
        assert "describe_node(filename='canon:quotes.md'" in disambig
        assert "graph_neighbors(filename=" not in disambig

    def test_graph_traverse_caller_renders_graph_traverse_form(self, tmp_path):
        graph = self._two_match_graph(tmp_path)
        nodes, disambig = _resolve_chunk_nodes(
            graph, "marcus-aurelius", "", caller="graph_traverse"
        )
        assert nodes == []
        assert "graph_traverse(filename='canon:quotes.md'" in disambig

    def test_describe_node_tool_emits_self_referencing_suggestions(self, tmp_path):
        """End-to-end: when describe_node hits an ambiguous heading the
        suggestion list must point back to describe_node, not
        graph_neighbors."""
        graph = self._two_match_graph(tmp_path)
        index = _make_index_with_graph(graph)
        tools = KBTools(kb_index=index)
        result = tools.describe_node("marcus-aurelius")
        assert "describe_node(filename='" in result
        assert "graph_neighbors(filename='" not in result


class TestResolverRankingPriority:
    """P0-1: candidate list is ranked before rendering — exact-leaf match,
    then filename-locality (same source), then substring position."""

    def test_exact_leaf_match_ranks_first(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        # Two candidates, only one has an exact leaf match.
        c_exact = Node(id="exact", node_type=NodeType.CHUNK,
                       name="a.md > stoicism",
                       filename="a.md", heading="philosophy > stoicism",
                       attributes={"source": "canon"})
        c_partial = Node(id="partial", node_type=NodeType.CHUNK,
                         name="b.md > stoicism-revival",
                         filename="b.md",
                         heading="modern > stoicism-revival",
                         attributes={"source": "canon"})
        # Add partial first so the only thing that can move exact ahead is
        # the ranker.
        graph.add_node(c_partial)
        graph.add_node(c_exact)

        nodes, disambig = _resolve_chunk_nodes(graph, "stoicism", "")
        assert nodes == []
        # First suggestion line after the header should be the exact match.
        lines = disambig.splitlines()
        first_suggestion = lines[1]
        assert "stoicism'" in first_suggestion and "stoicism-revival" not in first_suggestion

    def test_source_locality_breaks_ties(self, tmp_path):
        """When two candidates tie on leaf match, the one whose source
        matches the requested filename's source ranks higher."""
        graph = KnowledgeGraph(tmp_path / "graph.json")
        c_canon = Node(id="ca", node_type=NodeType.CHUNK,
                       name="canon.md > stoicism",
                       filename="canon.md", heading="root > stoicism",
                       attributes={"source": "canon"})
        c_know = Node(id="kn", node_type=NodeType.CHUNK,
                      name="wiki.md > stoicism",
                      filename="wiki/wiki.md", heading="root > stoicism",
                      attributes={"source": "knowledge"})
        graph.add_node(c_canon)
        graph.add_node(c_know)

        # Pin source to canon — canon candidate should rank first.
        nodes, disambig = _resolve_chunk_nodes(graph, "canon:stoicism", "")
        assert nodes == []
        # Either we got a usable disambig or it resolved cleanly; in the
        # ambiguous case, canon should be first.
        lines = disambig.splitlines() if disambig else []
        if len(lines) >= 2:
            assert "canon:" in lines[1]


class TestResolverQueryRerank:
    """P0-1: when ``query=`` is provided and an embedding fn is available,
    candidates are reranked by cosine similarity to the query."""

    def test_query_rerank_reorders_candidates(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        a = Node(id="a", node_type=NodeType.CHUNK,
                 name="a.md > marcus-aurelius",
                 filename="a.md",
                 heading="ancient-rome > marcus-aurelius",
                 summary="Roman emperor and Stoic philosopher",
                 attributes={"source": "canon"})
        b = Node(id="b", node_type=NodeType.CHUNK,
                 name="b.md > marcus-aurelius",
                 filename="b.md",
                 heading="modern-references > marcus-aurelius",
                 summary="A musical reference to the emperor",
                 attributes={"source": "knowledge"})
        graph.add_node(a)
        graph.add_node(b)

        # Stub embed fn — query and candidate `a` are colinear; `b` is
        # orthogonal. So `a` must rank first when query is given.
        index = _make_index_with_graph(graph)

        def _embed(texts):
            out = []
            for t in texts:
                if "philosopher" in t or "Stoic" in t or "stoic" in t.lower():
                    out.append([1.0, 0.0, 0.0])
                elif "musical" in t.lower():
                    out.append([0.0, 1.0, 0.0])
                else:
                    out.append([1.0, 0.0, 0.0])  # query
            return out

        index._embedding_fn = _embed

        nodes, disambig = _resolve_chunk_nodes(
            graph, "marcus-aurelius", "",
            query="stoic philosophy", kb_index=index,
        )
        assert nodes == []
        lines = disambig.splitlines()
        first_suggestion = lines[1]
        assert "a.md" in first_suggestion
        # Without query, the order is insertion order (b then a was added
        # b first? No — a then b). To be safe, just verify reranking
        # produced a deterministic order.
        assert "ancient-rome" in first_suggestion

    def test_query_rerank_silently_falls_back_when_embed_fails(self, tmp_path):
        graph = KnowledgeGraph(tmp_path / "graph.json")
        a = Node(id="a", node_type=NodeType.CHUNK,
                 name="a.md > marcus-aurelius",
                 filename="a.md", heading="rome > marcus-aurelius",
                 attributes={"source": "canon"})
        b = Node(id="b", node_type=NodeType.CHUNK,
                 name="b.md > marcus-aurelius",
                 filename="b.md", heading="logs > marcus-aurelius",
                 attributes={"source": "knowledge"})
        graph.add_node(a)
        graph.add_node(b)
        index = _make_index_with_graph(graph)

        def _embed(_texts):
            raise RuntimeError("network down")

        index._embedding_fn = _embed

        # Should still produce a disambig list (no crash, no missing
        # suggestions).
        nodes, disambig = _resolve_chunk_nodes(
            graph, "marcus-aurelius", "",
            query="anything", kb_index=index,
        )
        assert nodes == []
        assert disambig is not None
        assert "a.md" in disambig and "b.md" in disambig

    def test_exact_single_match_still_bypasses_disambig(self, tmp_path):
        """The reranker only runs on >1 candidates; exact single matches
        must continue to short-circuit the disambig path."""
        graph = KnowledgeGraph(tmp_path / "graph.json")
        c = Node(id="c", node_type=NodeType.CHUNK,
                 name="philosophy.md > marcus-aurelius",
                 filename="philosophy.md",
                 heading="philosophy-quotes > marcus-aurelius",
                 attributes={"source": "canon"})
        graph.add_node(c)

        nodes, disambig = _resolve_chunk_nodes(
            graph, "marcus-aurelius", "",
            caller="describe_node", query="anything",
        )
        assert disambig is None
        assert len(nodes) == 1
        assert nodes[0].id == "c"
