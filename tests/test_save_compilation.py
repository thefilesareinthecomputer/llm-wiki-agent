"""Tests for P3.1 — save_knowledge appends a Related block of similarity-
suggested wiki links so newly-compiled pages join the graph as a connected
node, not an island.

Test layers:
  1. _inject_related_block  — pure string surgery (no graph)
  2. _compute_related_block — graph traversal logic, fed a hand-built graph
  3. save_knowledge wiring  — full flow with _compute_related_block stubbed
     to return a known block (avoids depending on real embedding similarity
     thresholds, which fake hash-based embeddings cannot satisfy)
  4. REFERENCES round-trip  — proves the auto-injected wiki links are
     re-indexed and surface as REFERENCES edges in the graph
"""

import pytest


# ---------------------------------------------------------------------------
# Layer 1: pure string helpers (no graph required)
# ---------------------------------------------------------------------------

class TestInjectRelatedBlock:

    def _tools(self):
        from agent.tools import KBTools
        return KBTools(kb_index=None)

    def test_appends_when_no_block_present(self):
        tools = self._tools()
        body = "# title\n\nbody.\n"
        block = (
            tools._RELATED_BLOCK_START + "\n\n## Related\n\n- [[x]]\n\n"
            + tools._RELATED_BLOCK_END
        )
        out = tools._inject_related_block(body, block)
        assert tools._RELATED_BLOCK_START in out
        assert tools._RELATED_BLOCK_END in out
        assert "[[x]]" in out
        assert out.startswith(body)

    def test_replaces_existing_block(self):
        tools = self._tools()
        body = (
            "# title\n\nbody.\n\n"
            + tools._RELATED_BLOCK_START
            + "\n\n## Related\n\n- [[old]]\n\n"
            + tools._RELATED_BLOCK_END
            + "\n"
        )
        new_block = (
            tools._RELATED_BLOCK_START + "\n\n## Related\n\n- [[new]]\n\n"
            + tools._RELATED_BLOCK_END
        )
        out = tools._inject_related_block(body, new_block)
        assert "[[new]]" in out
        assert "[[old]]" not in out
        assert out.count(tools._RELATED_BLOCK_START) == 1
        assert out.count(tools._RELATED_BLOCK_END) == 1

    def test_empty_block_returns_body_unchanged(self):
        tools = self._tools()
        body = "# title\n\nbody.\n"
        assert tools._inject_related_block(body, "") == body

    def test_idempotent_when_block_already_current(self):
        tools = self._tools()
        block = (
            tools._RELATED_BLOCK_START + "\n\n## Related\n\n- [[x]]\n\n"
            + tools._RELATED_BLOCK_END
        )
        body = "# title\n\nbody.\n\n" + block + "\n"
        once = tools._inject_related_block(body, block)
        twice = tools._inject_related_block(once, block)
        assert once == twice


# ---------------------------------------------------------------------------
# Layer 2: graph traversal — _compute_related_block against a hand-built graph
# ---------------------------------------------------------------------------

def _build_graph_with_edges(tmp_path):
    """Build a minimal in-memory graph with one 'self' chunk and three
    cross-file chunks connected via INTER_FILE / SIMILAR / CROSS_DOMAIN
    edges. Returns (graph, kb_index_stub, self_filename)."""
    from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType

    g = KnowledgeGraph(tmp_path / "graph.json")

    self_chunk = Node(
        id="self-1",
        node_type=NodeType.CHUNK,
        name="wiki/hpa-axis.md > overview",
        filename="wiki/hpa-axis.md",
        heading="overview",
    )
    cortisol = Node(
        id="cort-1", node_type=NodeType.CHUNK,
        name="wiki/cortisol.md > cortisol",
        filename="wiki/cortisol.md", heading="cortisol",
    )
    stress = Node(
        id="str-1", node_type=NodeType.CHUNK,
        name="wiki/stress.md > stress",
        filename="wiki/stress.md", heading="stress",
    )
    canon_endo = Node(
        id="can-1", node_type=NodeType.CHUNK,
        name="endocrinology.md > overview",
        filename="endocrinology.md", heading="overview",
        attributes={"source": "canon"},
    )
    for n in (self_chunk, cortisol, stress, canon_endo):
        g.add_node(n)

    # Cross-file edges: heaviest weight wins ranking
    g.add_edge(Edge(
        source_id="self-1", target_id="cort-1",
        edge_type=EdgeType.INTER_FILE, weight=0.85,
        evidence="strong cross-file",
    ))
    g.add_edge(Edge(
        source_id="self-1", target_id="str-1",
        edge_type=EdgeType.INTER_FILE, weight=0.62,
        evidence="medium cross-file",
    ))
    g.add_edge(Edge(
        source_id="self-1", target_id="can-1",
        edge_type=EdgeType.CROSS_DOMAIN, weight=0.71,
        evidence="cross-domain",
    ))
    # Noise edges that should NOT show up: PARENT_CHILD + REFERENCES + self
    g.add_edge(Edge(
        source_id="self-1", target_id="folder_x",
        edge_type=EdgeType.PARENT_CHILD, weight=1.0,
    ))
    g.add_edge(Edge(
        source_id="self-1", target_id="cort-1",
        edge_type=EdgeType.REFERENCES, weight=1.0,
    ))

    class _KBIndexStub:
        pass
    stub = _KBIndexStub()
    stub.graph = g
    return g, stub, "wiki/hpa-axis.md"


class TestComputeRelatedBlock:

    def test_returns_empty_when_no_index(self):
        from agent.tools import KBTools
        tools = KBTools(kb_index=None)
        assert tools._compute_related_block("foo.md") == ""

    def test_returns_empty_when_no_chunks_for_filename(self, tmp_path):
        from agent.tools import KBTools
        _, stub, _ = _build_graph_with_edges(tmp_path)
        tools = KBTools(kb_index=stub)
        assert tools._compute_related_block("nonexistent.md") == ""

    def test_emits_wiki_links_for_cross_file_neighbors(self, tmp_path):
        from agent.tools import KBTools
        _, stub, fname = _build_graph_with_edges(tmp_path)
        tools = KBTools(kb_index=stub)
        block = tools._compute_related_block(fname, top_n=5)
        assert block, "expected a non-empty block when neighbors exist"
        assert tools._RELATED_BLOCK_START in block
        assert tools._RELATED_BLOCK_END in block
        assert "## Related" in block
        # All three cross-file neighbors should appear as wiki links
        assert "[[cortisol]]" in block
        assert "[[stress]]" in block
        assert "[[endocrinology]]" in block

    def test_excludes_self_filename(self, tmp_path):
        from agent.tools import KBTools
        _, stub, fname = _build_graph_with_edges(tmp_path)
        tools = KBTools(kb_index=stub)
        block = tools._compute_related_block(fname)
        assert "[[hpa-axis]]" not in block

    def test_skips_parent_child_and_references_edges(self, tmp_path):
        """PARENT_CHILD (folders) and REFERENCES (already-explicit) edges
        must not contribute to the suggestion list."""
        from agent.tools import KBTools
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType

        g = KnowledgeGraph(tmp_path / "graph.json")
        self_chunk = Node(
            id="self-1", node_type=NodeType.CHUNK,
            name="a.md > root", filename="a.md", heading="root",
        )
        other = Node(
            id="other-1", node_type=NodeType.CHUNK,
            name="b.md > root", filename="b.md", heading="root",
        )
        g.add_node(self_chunk); g.add_node(other)
        g.add_edge(Edge(
            source_id="self-1", target_id="other-1",
            edge_type=EdgeType.REFERENCES, weight=1.0,
        ))
        g.add_edge(Edge(
            source_id="self-1", target_id="other-1",
            edge_type=EdgeType.PARENT_CHILD, weight=1.0,
        ))

        class _Stub: pass
        stub = _Stub(); stub.graph = g
        tools = KBTools(kb_index=stub)
        # Only PARENT_CHILD + REFERENCES exist → block is empty
        assert tools._compute_related_block("a.md") == ""

    def test_top_n_caps_suggestions(self, tmp_path):
        from agent.tools import KBTools
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType

        g = KnowledgeGraph(tmp_path / "graph.json")
        self_chunk = Node(
            id="self-1", node_type=NodeType.CHUNK,
            name="me.md > root", filename="me.md", heading="root",
        )
        g.add_node(self_chunk)
        for i in range(8):
            tgt = Node(
                id=f"t-{i}", node_type=NodeType.CHUNK,
                name=f"page{i}.md > h", filename=f"page{i}.md", heading="h",
            )
            g.add_node(tgt)
            g.add_edge(Edge(
                source_id="self-1", target_id=f"t-{i}",
                edge_type=EdgeType.INTER_FILE, weight=0.6 + 0.01 * i,
            ))

        class _Stub: pass
        stub = _Stub(); stub.graph = g
        tools = KBTools(kb_index=stub)
        block = tools._compute_related_block("me.md", top_n=3)
        import re
        links = re.findall(r"\[\[([^\]]+)\]\]", block)
        assert len(links) == 3
        # Highest-weighted (last added with biggest weight) win
        assert "page7" in links


# ---------------------------------------------------------------------------
# Layer 3: end-to-end save_knowledge wiring (with _compute_related_block stubbed)
# ---------------------------------------------------------------------------

@pytest.fixture
def tools_with_kb(tmp_path, monkeypatch):
    """Real KB on disk + real KBIndex (fake embeddings) so save_knowledge
    actually writes, reindexes, and rebuilds the graph. Tests that hook
    _compute_related_block control whether a block is injected, isolating
    the wiring from real-similarity nondeterminism."""
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

    import agent.tools as agent_tools
    monkeypatch.setattr(agent_tools, "_KNOWLEDGE_DIR", kb, raising=False)
    monkeypatch.setattr(agent_tools, "_CANON_DIR", canon, raising=False)

    (kb / "wiki" / "cortisol.md").write_text(
        "# cortisol\n\nA stress-related steroid hormone.\n"
    )
    (kb / "wiki" / "stress.md").write_text(
        "# stress\n\nThe body's response to perceived threat.\n"
    )

    idx = KBIndex()
    idx.db = lancedb.connect(str(tmp_path / "lancedb"))
    idx._embedding_fn = FakeEmbeddingFunction()
    idx.build_index(extract_entities=False, llm_summaries=False, force=True)

    tools = KBTools(kb_index=idx, kb_dir=kb, canon_dir=canon)
    return tools, kb


class TestSaveKnowledgeAppendsRelatedBlock:

    def test_save_injects_block_when_compute_returns_one(
        self, tools_with_kb, monkeypatch,
    ):
        """When _compute_related_block produces a block, save_knowledge must
        write it into the file on disk."""
        tools, kb = tools_with_kb
        canned = (
            tools._RELATED_BLOCK_START
            + "\n\n## Related\n\n- [[cortisol]]\n- [[stress]]\n\n"
            + tools._RELATED_BLOCK_END
        )
        monkeypatch.setattr(
            tools, "_compute_related_block", lambda *a, **k: canned,
        )
        result = tools.save_knowledge(
            "hpa-axis.md",
            "## Overview\n\nThe HPA axis governs stress hormone release.\n",
        )
        assert "Saved:" in result
        saved = (kb / "wiki" / "hpa-axis.md").read_text()
        assert tools._RELATED_BLOCK_START in saved
        assert tools._RELATED_BLOCK_END in saved
        assert "[[cortisol]]" in saved
        assert "[[stress]]" in saved

    def test_save_does_not_inject_when_compute_returns_empty(
        self, tools_with_kb, monkeypatch,
    ):
        """No neighbors → no block → file stays clean (no empty markers)."""
        tools, kb = tools_with_kb
        monkeypatch.setattr(
            tools, "_compute_related_block", lambda *a, **k: "",
        )
        tools.save_knowledge(
            "isolated.md", "## Body\n\nNothing connects here.\n",
        )
        saved = (kb / "wiki" / "isolated.md").read_text()
        assert tools._RELATED_BLOCK_START not in saved

    def test_save_compute_failure_does_not_block_save(
        self, tools_with_kb, monkeypatch,
    ):
        """If _compute_related_block raises, save_knowledge must still
        return success — related-block compilation is best-effort."""
        tools, kb = tools_with_kb

        def boom(*a, **k):
            raise RuntimeError("graph traversal blew up")

        monkeypatch.setattr(tools, "_compute_related_block", boom)
        result = tools.save_knowledge(
            "robust.md", "## Body\n\nstill saves.\n",
        )
        assert "Saved:" in result
        assert (kb / "wiki" / "robust.md").exists()


# ---------------------------------------------------------------------------
# Layer 4: round-trip — auto-injected wiki links become REFERENCES edges
# ---------------------------------------------------------------------------

class TestRelatedBlockBecomesReferencesEdges:

    def test_auto_links_create_references_edges(
        self, tools_with_kb, monkeypatch,
    ):
        """The wiki links emitted into the auto-generated related block
        should round-trip through _build_wiki_link_edges and become real
        REFERENCES edges on the graph after save_knowledge completes."""
        tools, kb = tools_with_kb
        canned = (
            tools._RELATED_BLOCK_START
            + "\n\n## Related\n\n- [[cortisol]]\n- [[stress]]\n\n"
            + tools._RELATED_BLOCK_END
        )
        monkeypatch.setattr(
            tools, "_compute_related_block", lambda *a, **k: canned,
        )
        tools.save_knowledge(
            "hpa-axis.md",
            "## Overview\n\nStress hormone regulation via the HPA axis.\n",
        )

        from knowledge.graph import EdgeType, NodeType
        graph = tools.kb_index.graph
        chunks = {
            nid: n for nid, n in graph.nodes.items()
            if n.node_type == NodeType.CHUNK
        }
        outgoing_refs = []
        for e in graph.edges.values():
            if e.edge_type != EdgeType.REFERENCES:
                continue
            src = chunks.get(e.source_id)
            if src and src.filename == "wiki/hpa-axis.md":
                outgoing_refs.append((e, chunks.get(e.target_id)))
        assert outgoing_refs, (
            "expected REFERENCES edges from the auto-injected wiki links"
        )
        target_files = {
            tgt.filename for _, tgt in outgoing_refs if tgt
        }
        # At least one of the canned link targets resolves to a real chunk
        assert {"wiki/cortisol.md", "wiki/stress.md"} & target_files
