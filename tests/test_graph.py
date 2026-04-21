"""Tests for the knowledge graph data structure."""

import json
import tempfile
from pathlib import Path

import pytest

from knowledge.graph import (
    KnowledgeGraph,
    Node,
    Edge,
    EdgeType,
    NodeType,
    build_folder_tree,
    format_folder_tree,
)


@pytest.fixture
def graph_path(tmp_path):
    return tmp_path / "test_graph.json"


@pytest.fixture
def graph(graph_path):
    return KnowledgeGraph(graph_path)


@pytest.fixture
def populated_graph(graph):
    # Add chunk nodes
    chunk1 = Node(id="c1", node_type=NodeType.CHUNK, name="file1.md > Intro",
                  filename="file1.md", heading="Intro", summary="An introduction")
    chunk2 = Node(id="c2", node_type=NodeType.CHUNK, name="file1.md > Methods",
                  filename="file1.md", heading="Methods", summary="Methods section")
    chunk3 = Node(id="c3", node_type=NodeType.CHUNK, name="file2.md > Overview",
                  filename="file2.md", heading="Overview", summary="Overview of topic")
    graph.add_node(chunk1)
    graph.add_node(chunk2)
    graph.add_node(chunk3)

    # Add entity nodes
    entity1 = Node(id="e1", node_type=NodeType.CONCEPT, name="quantum computing",
                   source_chunk_id="c1", tags=["knowledge"])
    entity2 = Node(id="e2", node_type=NodeType.ENTITY, name="Ollama",
                   source_chunk_id="c2", tags=["knowledge"])
    graph.add_node(entity1)
    graph.add_node(entity2)

    # Add edges
    graph.add_edge(Edge(source_id="c1", target_id="c2", edge_type=EdgeType.SIMILAR,
                        weight=0.85, evidence="high similarity"))
    graph.add_edge(Edge(source_id="c1", target_id="c3", edge_type=EdgeType.CROSS_DOMAIN,
                        weight=0.65, evidence="cross-folder"))
    graph.add_edge(Edge(source_id="e1", target_id="c1", edge_type=EdgeType.REFERENCES,
                        weight=0.9, evidence="entity in chunk"))
    graph.add_edge(Edge(source_id="c1", target_id="c2", edge_type=EdgeType.PARENT_CHILD,
                        weight=1.0, evidence="heading hierarchy"))

    return graph


class TestNode:
    def test_node_creation(self):
        node = Node(id="n1", node_type=NodeType.CHUNK, name="test")
        assert node.id == "n1"
        assert node.node_type == NodeType.CHUNK
        assert node.tags == []

    def test_node_serialization(self):
        node = Node(id="n1", node_type=NodeType.ENTITY, name="test",
                    attributes={"key": "val"}, tags=["tag1"])
        d = node.to_dict()
        assert d["node_type"] == "entity"
        assert d["tags"] == ["tag1"]

        restored = Node.from_dict(d)
        assert restored.node_type == NodeType.ENTITY
        assert restored.name == "test"
        assert restored.tags == ["tag1"]


class TestEdge:
    def test_edge_creation(self):
        edge = Edge(source_id="a", target_id="b", edge_type=EdgeType.SIMILAR, weight=0.8)
        assert edge.key == "a:b:similar"

    def test_edge_serialization(self):
        edge = Edge(source_id="a", target_id="b", edge_type=EdgeType.CROSS_DOMAIN,
                    weight=0.6, evidence="test")
        d = edge.to_dict()
        assert d["edge_type"] == "cross_domain"

        restored = Edge.from_dict(d)
        assert restored.edge_type == EdgeType.CROSS_DOMAIN
        assert restored.weight == 0.6


class TestKnowledgeGraph:
    def test_add_node(self, graph):
        node = Node(id="n1", node_type=NodeType.CHUNK, name="test")
        graph.add_node(node)
        assert graph.get_node("n1") is not None
        assert graph.get_node("n1").name == "test"

    def test_add_edge(self, graph):
        graph.add_node(Node(id="a", node_type=NodeType.CHUNK, name="a"))
        graph.add_node(Node(id="b", node_type=NodeType.CHUNK, name="b"))
        graph.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.SIMILAR, weight=0.8))
        neighbors = graph.get_neighbors("a")
        assert len(neighbors) == 1
        assert neighbors[0][0].id == "b"

    def test_entity_dedup(self, graph):
        """Same name + type merges into one node."""
        e1 = Node(id="e1", node_type=NodeType.CONCEPT, name="Quantum Computing",
                  source_chunk_id="c1")
        e2 = Node(id="e2", node_type=NodeType.CONCEPT, name="quantum computing",
                  source_chunk_id="c2")
        graph.add_node(e1)
        merged = graph.add_node(e2)
        # Should return the existing node, not create a new one
        assert merged.id == "e1"
        assert len(graph.nodes) == 1

    def test_different_types_not_deduped(self, graph):
        """Different types with same name are separate nodes."""
        e1 = Node(id="e1", node_type=NodeType.CONCEPT, name="Python")
        e2 = Node(id="e2", node_type=NodeType.ENTITY, name="Python")
        graph.add_node(e1)
        graph.add_node(e2)
        assert len(graph.nodes) == 2

    def test_get_neighbors_filtered(self, populated_graph):
        neighbors = populated_graph.get_neighbors("c1", edge_type=EdgeType.SIMILAR)
        assert len(neighbors) >= 1
        # CROSS_DOMAIN should be filtered out
        for _, edge in neighbors:
            assert edge.edge_type == EdgeType.SIMILAR

    def test_traverse(self, populated_graph):
        results = populated_graph.traverse("c1", max_depth=2)
        # Should find c2, c3, e1 via edges from c1
        found_ids = {r[0].id for r in results}
        assert "c2" in found_ids
        assert "c3" in found_ids

    def test_traverse_depth_limit(self, populated_graph):
        results = populated_graph.traverse("c1", max_depth=1)
        for node, edge, depth in results:
            assert depth <= 1

    def test_search_entities(self, populated_graph):
        results = populated_graph.search_entities("quantum")
        assert len(results) >= 1
        assert results[0].name == "quantum computing"

    def test_find_chunk_node(self, populated_graph):
        node = populated_graph.find_chunk_node("file1.md", "Intro")
        assert node is not None
        assert node.heading == "Intro"

    def test_find_chunk_node_case_insensitive(self, populated_graph):
        node = populated_graph.find_chunk_node("file1.md", "intro")
        assert node is not None

    def test_get_stats(self, populated_graph):
        stats = populated_graph.get_stats()
        assert stats["nodes"] == 5  # 3 chunks + 2 entities
        assert stats["edges"] >= 2
        assert "chunk" in stats["node_types"]
        assert "concept" in stats["node_types"]

    def test_save_load_roundtrip(self, populated_graph, graph_path):
        populated_graph.save()
        assert graph_path.exists()

        # Load into new graph
        graph2 = KnowledgeGraph(graph_path)
        assert len(graph2.nodes) == len(populated_graph.nodes)
        assert len(graph2.edges) == len(populated_graph.edges)

        # Check a specific node survived
        node = graph2.get_node("e1")
        assert node is not None
        assert node.name == "quantum computing"

    def test_clear(self, populated_graph):
        populated_graph.clear()
        assert len(populated_graph.nodes) == 0
        assert len(populated_graph.edges) == 0

    def test_edge_weight_merge(self, graph):
        graph.add_node(Node(id="a", node_type=NodeType.CHUNK, name="a"))
        graph.add_node(Node(id="b", node_type=NodeType.CHUNK, name="b"))
        graph.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.SIMILAR, weight=0.5))
        graph.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.SIMILAR, weight=0.9))
        # Should keep the higher weight
        key = "a:b:similar"
        assert graph.edges[key].weight == 0.9

    def test_load_nonexistent(self, graph_path):
        graph = KnowledgeGraph(graph_path)
        assert len(graph.nodes) == 0

    def test_neighbors_bidirectional(self, graph):
        graph.add_node(Node(id="a", node_type=NodeType.CHUNK, name="a"))
        graph.add_node(Node(id="b", node_type=NodeType.CHUNK, name="b"))
        graph.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.SIMILAR, weight=0.8))
        # From a -> b
        neighbors_a = graph.get_neighbors("a")
        assert any(n.id == "b" for n, _ in neighbors_a)
        # From b -> a (bidirectional)
        neighbors_b = graph.get_neighbors("b")
        assert any(n.id == "a" for n, _ in neighbors_b)


class TestFolderNode:
    """Tests for FOLDER node type and folder hierarchy."""

    def test_folder_node_creation(self):
        node = Node(
            id="folder_knowledge_ai",
            node_type=NodeType.FOLDER,
            name="ai",
            summary="AI research notes",
            attributes={"source": "knowledge", "path": "/app/knowledge/ai", "file_count": 3, "files": ["intro.md"]},
            tags=["knowledge"],
        )
        assert node.node_type == NodeType.FOLDER
        assert node.name == "ai"
        assert node.summary == "AI research notes"
        assert node.attributes["file_count"] == 3

    def test_folder_node_serialization(self):
        node = Node(
            id="folder_knowledge_ml",
            node_type=NodeType.FOLDER,
            name="ml",
            attributes={"source": "knowledge"},
            tags=["knowledge", "ml"],
        )
        d = node.to_dict()
        assert d["node_type"] == "folder"

        restored = Node.from_dict(d)
        assert restored.node_type == NodeType.FOLDER
        assert restored.tags == ["knowledge", "ml"]

    def test_folder_parent_child_edge(self):
        graph = KnowledgeGraph(Path("/tmp/test_graph_folder.json"))
        parent = Node(id="folder_knowledge_root", node_type=NodeType.FOLDER, name="root",
                      attributes={"source": "knowledge"})
        child = Node(id="folder_knowledge_ai", node_type=NodeType.FOLDER, name="root/ai",
                     attributes={"source": "knowledge"})
        graph.add_node(parent)
        graph.add_node(child)
        graph.add_edge(Edge(source_id="folder_knowledge_root", target_id="folder_knowledge_ai",
                           edge_type=EdgeType.PARENT_CHILD, weight=1.0, evidence="folder hierarchy"))
        neighbors = graph.get_neighbors("folder_knowledge_root", edge_type=EdgeType.PARENT_CHILD)
        assert len(neighbors) == 1
        assert neighbors[0][0].id == "folder_knowledge_ai"


class TestBuildFolderTree:
    """Tests for build_folder_tree function."""

    @pytest.fixture
    def kb_structure(self, tmp_path):
        """Create a realistic KB directory structure with subfolders."""
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        # Root file
        (kb_dir / "root.md").write_text("# Root\nRoot content")
        # AI subfolder
        ai_dir = kb_dir / "ai"
        ai_dir.mkdir()
        (ai_dir / "intro.md").write_text("# AI Intro\nArtificial intelligence basics")
        (ai_dir / "ml.md").write_text("# ML\nMachine learning")
        # Nested subfolder
        deep_dir = ai_dir / "deep"
        deep_dir.mkdir()
        (deep_dir / "neural.md").write_text("# Neural Nets\nDeep learning")
        # Folder with README
        readme_dir = kb_dir / "guides"
        readme_dir.mkdir()
        (readme_dir / "README.md").write_text("# Guides\nUser guides and tutorials\nThis is the guide folder.")
        (readme_dir / "getting-started.md").write_text("# Getting Started\nFirst steps")
        return kb_dir

    def test_build_folder_tree_creates_folder_nodes(self, kb_structure):
        canon_dir = kb_structure.parent / "canon"
        results = build_folder_tree(kb_structure, canon_dir)
        folder_nodes = [r[0] for r in results if r[0] is not None]
        assert len(folder_nodes) >= 3  # ai, ai/deep, guides at minimum
        node_ids = {n.id for n in folder_nodes}
        assert "folder_knowledge_ai" in node_ids
        assert "folder_knowledge_guides" in node_ids

    def test_build_folder_tree_creates_parent_child_edges(self, kb_structure):
        canon_dir = kb_structure.parent / "canon"
        results = build_folder_tree(kb_structure, canon_dir)
        edges = [r[1] for r in results if r[1] is not None]
        parent_child_edges = [e for e in edges if e.edge_type == EdgeType.PARENT_CHILD]
        # ai/deep should have parent ai
        assert len(parent_child_edges) >= 1
        deep_edge = [e for e in parent_child_edges if "deep" in e.target_id]
        assert len(deep_edge) >= 1

    def test_build_folder_tree_reads_readme_summary(self, kb_structure):
        canon_dir = kb_structure.parent / "canon"
        results = build_folder_tree(kb_structure, canon_dir)
        folder_nodes = [r[0] for r in results if r[0] is not None]
        guides_node = next(n for n in folder_nodes if n.id == "folder_knowledge_guides")
        assert "guide" in guides_node.summary.lower()

    def test_build_folder_tree_file_count(self, kb_structure):
        canon_dir = kb_structure.parent / "canon"
        results = build_folder_tree(kb_structure, canon_dir)
        folder_nodes = [r[0] for r in results if r[0] is not None]
        ai_node = next(n for n in folder_nodes if n.id == "folder_knowledge_ai")
        assert ai_node.attributes["file_count"] == 2  # intro.md, ml.md


class TestFormatFolderTree:
    """Tests for format_folder_tree function."""

    def test_format_empty_graph(self):
        graph_path = Path("/tmp/test_format_tree.json")
        graph = KnowledgeGraph(graph_path)
        result = format_folder_tree(graph, source="knowledge")
        assert result == ""

    def test_format_folder_tree_rendes_hierarchy(self):
        graph_path = Path("/tmp/test_format_tree2.json")
        graph = KnowledgeGraph(graph_path)
        root = Node(id="folder_knowledge_", node_type=NodeType.FOLDER, name="",
                    attributes={"source": "knowledge"})
        child = Node(id="folder_knowledge_ai", node_type=NodeType.FOLDER, name="ai",
                     attributes={"source": "knowledge"})
        graph.add_node(root)
        graph.add_node(child)
        graph.add_edge(Edge(source_id="folder_knowledge_", target_id="folder_knowledge_ai",
                           edge_type=EdgeType.PARENT_CHILD, weight=1.0))
        # Add a chunk in the ai folder
        chunk = Node(id="chunk1", node_type=NodeType.CHUNK, name="ai/intro.md > Intro",
                    attributes={"folder": "ai"})
        graph.add_node(chunk)
        result = format_folder_tree(graph, source="knowledge")
        assert "knowledge/" in result
        assert "ai" in result


class TestExtractEntities:
    """Tests for extract_entities function with relationship support."""

    def test_extract_entities_returns_entities(self, monkeypatch):
        """extract_entities returns entity dicts from langextract."""
        from knowledge.graph import extract_entities

        # Mock langextract
        mock_extraction = type("Ext", (), {
            "extraction_class": "concept",
            "extraction_text": "cortisol",
            "char_interval": (0, 8),
            "attributes": {},
        })()
        mock_result = type("Res", (), {"extractions": [mock_extraction]})()

        import langextract as lx
        monkeypatch.setattr(lx, "extract", lambda **kwargs: mock_result)

        result = extract_entities("Cortisol affects stress.", "Stress")
        assert len(result) == 1
        assert result[0]["class"] == "concept"
        assert result[0]["text"] == "cortisol"

    def test_extract_entities_includes_relationships(self, monkeypatch):
        """extract_entities returns relationship dicts with subject/object attributes."""
        from knowledge.graph import extract_entities

        ent_ext = type("Ext", (), {
            "extraction_class": "concept",
            "extraction_text": "hippocampus",
            "char_interval": (0, 10),
            "attributes": {},
        })()
        rel_ext = type("Ext", (), {
            "extraction_class": "relationship",
            "extraction_text": "regulates",
            "char_interval": None,  # relationships may not have char_interval
            "attributes": {"subject": "hippocampus", "object": "cortisol", "context": "stress response"},
        })()
        mock_result = type("Res", (), {"extractions": [ent_ext, rel_ext]})()

        import langextract as lx
        monkeypatch.setattr(lx, "extract", lambda **kwargs: mock_result)

        result = extract_entities("The hippocampus regulates cortisol.", "Brain")
        assert len(result) == 2
        # First is entity
        assert result[0]["class"] == "concept"
        # Second is relationship with attributes
        assert result[1]["class"] == "relationship"
        assert result[1]["attributes"]["subject"] == "hippocampus"
        assert result[1]["attributes"]["object"] == "cortisol"

    def test_extract_entities_handles_failure(self, monkeypatch):
        """extract_entities returns empty list on failure."""
        from knowledge.graph import extract_entities

        import langextract as lx
        monkeypatch.setattr(lx, "extract", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))

        result = extract_entities("Some text", "Test")
        assert result == []

    def test_relationship_without_char_interval_included(self, monkeypatch):
        """Relationships without char_interval are included (unlike regular entities)."""
        from knowledge.graph import extract_entities

        # Regular entity without char_interval should be skipped
        ent_no_interval = type("Ext", (), {
            "extraction_class": "concept",
            "extraction_text": "skipped",
            "char_interval": None,
            "attributes": {},
        })()
        # Relationship without char_interval should be included
        rel_no_interval = type("Ext", (), {
            "extraction_class": "relationship",
            "extraction_text": "causes",
            "char_interval": None,
            "attributes": {"subject": "a", "object": "b"},
        })()
        mock_result = type("Res", (), {"extractions": [ent_no_interval, rel_no_interval]})()

        import langextract as lx
        monkeypatch.setattr(lx, "extract", lambda **kwargs: mock_result)

        result = extract_entities("Text", "Test")
        assert len(result) == 1
        assert result[0]["class"] == "relationship"


class TestTraverseExcludeEdgeTypes:
    """Test traverse() with exclude_edge_types parameter."""

    def test_traverse_excludes_parent_child(self, graph):
        """traverse() skips PARENT_CHILD edges when excluded."""
        # Build a small graph with both semantic and structural edges
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="file.md > Root",
                   filename="file.md", heading="Root")
        c2 = Node(id="c2", node_type=NodeType.CHUNK, name="file.md > Child",
                   filename="file.md", heading="Child")
        c3 = Node(id="c3", node_type=NodeType.CHUNK, name="other.md > Related",
                   filename="other.md", heading="Related")
        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_node(c3)
        graph.add_edge(Edge(source_id="c1", target_id="c2",
                           edge_type=EdgeType.PARENT_CHILD, weight=1.0))
        graph.add_edge(Edge(source_id="c1", target_id="c3",
                           edge_type=EdgeType.INTER_FILE, weight=0.65))

        # Without exclusion: both edges followed
        result_all = graph.traverse("c1", max_depth=2)
        visited_names = [n.name for n, _, _ in result_all]
        assert "file.md > Child" in visited_names
        assert "other.md > Related" in visited_names

        # With PARENT_CHILD excluded: only INTER_FILE followed
        result_excl = graph.traverse("c1", max_depth=2,
                                     exclude_edge_types={EdgeType.PARENT_CHILD})
        visited_names_excl = [n.name for n, _, _ in result_excl]
        assert "file.md > Child" not in visited_names_excl
        assert "other.md > Related" in visited_names_excl