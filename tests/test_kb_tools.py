"""Tests for knowledge base tools."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from agent.tools import (
    KBTools,
    FileTools,
    ShellTools,
    ToolResult,
    reset_budget,
    set_available_tokens,
    _current_kb_loads,
    _KB_MAX_LOADS_PER_RESPONSE,
)


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temp knowledge and canon dirs with sample files."""
    kb_dir = tmp_path / "knowledge"
    canon_dir = tmp_path / "canon"
    kb_dir.mkdir()
    canon_dir.mkdir()

    # Create a sample knowledge file
    (kb_dir / "test.md").write_text(
        "# Architecture\n\nSystem design overview.\n\n"
        "## Data Flow\n\nData flows from UI to API to model.\n\n"
        "## Vector Pipeline\n\nLanceDB handles vector storage.\n"
    )

    # Create a canon file
    (canon_dir / "canon_doc.md").write_text(
        "# Canon Document\n\nThis is read-only.\n\n"
        "## Section One\n\nCanon content here.\n"
    )

    return kb_dir, canon_dir


@pytest.fixture
def kb_tools(temp_dirs):
    """Create KBTools with temp dirs."""
    kb_dir, canon_dir = temp_dirs
    return KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)


@pytest.fixture(autouse=True)
def reset_budget_fixture():
    """Reset budget before each test."""
    reset_budget()
    yield
    reset_budget()


class TestListKnowledge:
    """Test list_knowledge tool."""

    def test_lists_files(self, kb_tools, temp_dirs):
        result = kb_tools.list_knowledge()
        assert "test.md" in result
        assert "canon_doc.md" in result

    def test_shows_canon_tag(self, kb_tools, temp_dirs):
        result = kb_tools.list_knowledge()
        assert "canon" in result

    def test_empty_kb(self, tmp_path):
        kb_dir = tmp_path / "empty_kb"
        canon_dir = tmp_path / "empty_canon"
        kb_dir.mkdir()
        canon_dir.mkdir()
        tools = KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)
        result = tools.list_knowledge()
        assert "empty" in result.lower()


class TestReadKnowledge:
    """Test read_knowledge tool (heading tree display)."""

    def test_file_not_found(self, kb_tools):
        result = kb_tools.read_knowledge("nonexistent.md")
        assert "not found" in result.lower()

    def test_shows_token_count(self, kb_tools):
        result = kb_tools.read_knowledge("test.md")
        assert "tokens" in result.lower()

    def test_empty_file(self, kb_tools, temp_dirs):
        kb_dir, _ = temp_dirs
        (kb_dir / "empty.md").write_text("")
        result = kb_tools.read_knowledge("empty.md")
        assert "no section content" in result


class TestReadKnowledgeSection:
    """Test read_knowledge_section with budget enforcement."""

    def test_loads_h1_section(self, kb_tools):
        result = kb_tools.read_knowledge_section("test.md", "Architecture")
        assert "Architecture" in result or "System design" in result
        assert "LOADED" in result

    def test_loads_h2_subsection(self, kb_tools):
        result = kb_tools.read_knowledge_section("test.md", "Data Flow")
        assert "Data Flow" in result or "data flows" in result.lower()
        assert "LOADED" in result

    def test_file_not_found(self, kb_tools):
        result = kb_tools.read_knowledge_section("nonexistent.md", "anything")
        assert "not found" in result.lower()

    def test_section_not_found(self, kb_tools):
        result = kb_tools.read_knowledge_section("test.md", "Nonexistent Section")
        assert "not found" in result.lower() or "Available" in result

    def test_budget_enforcement_max_loads(self, kb_tools):
        """After max loads, further loads are refused."""
        from agent import tools as tools_mod
        tools_mod._current_kb_loads = tools_mod._KB_MAX_LOADS_PER_RESPONSE
        result = kb_tools.read_knowledge_section("test.md", "Architecture")
        assert "REFUSED" in result
        assert "limit" in result.lower()

    def test_budget_enforcement_low_tokens(self, kb_tools):
        """When available tokens below minimum, loads are refused."""
        from agent import tools as tools_mod
        tools_mod._current_available_tokens = tools_mod._KB_MIN_REMAINING_TOKENS - 1
        result = kb_tools.read_knowledge_section("test.md", "Architecture")
        assert "REFUSED" in result
        assert "budget" in result.lower()

    def test_budget_increments(self, kb_tools):
        """Each successful load increments the counter."""
        from agent import tools as tools_mod
        assert tools_mod._current_kb_loads == 0
        kb_tools.read_knowledge_section("test.md", "Architecture")
        assert tools_mod._current_kb_loads == 1

    def test_case_insensitive(self, kb_tools):
        result = kb_tools.read_knowledge_section("test.md", "architecture")
        assert "LOADED" in result

    def test_provenance_header_h1(self, kb_tools):
        """H1 load prepends structured [SECTION: ...] header."""
        result = kb_tools.read_knowledge_section("test.md", "Architecture")
        first = result.splitlines()[0]
        # A2 emits canonical <source>:<relpath> in the SECTION header.
        assert first.startswith("[SECTION: knowledge:test.md | ")
        assert "Architecture" in first
        assert "LOADED" in first
        assert "(COMPLETE)" in first or "(TRUNCATED" in first

    def test_provenance_header_h2(self, kb_tools):
        """H2 subsection load includes parent heading in ancestry path."""
        result = kb_tools.read_knowledge_section("test.md", "Data Flow")
        first = result.splitlines()[0]
        assert first.startswith("[SECTION: knowledge:test.md | ")
        assert "Architecture > Data Flow" in first
        assert "LOADED" in first

    def test_complete_marker_short_section(self, kb_tools):
        """Short sections always carry the COMPLETE marker."""
        result = kb_tools.read_knowledge_section("test.md", "Architecture")
        assert "(COMPLETE)" in result.splitlines()[0]

    def test_truncated_marker_long_section(self, tmp_path):
        """Sections that exceed _KB_FILE_MAX_TOKENS report TRUNCATED with offset."""
        from agent import tools as tools_mod
        kb_dir = tmp_path / "knowledge"
        canon_dir = tmp_path / "canon"
        kb_dir.mkdir()
        canon_dir.mkdir()
        # Build a section with ~12k tokens so a single 8k-token load truncates.
        big = "# Big\n\n" + ("Lorem ipsum dolor sit amet. " * 4000)
        (kb_dir / "big.md").write_text(big)
        tools = tools_mod.KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)
        result = tools.read_knowledge_section("big.md", "Big")
        first = result.splitlines()[0]
        assert "TRUNCATED" in first
        assert "offset=" in first

    def test_offset_continuation(self, tmp_path):
        """Calling with offset returns later content past the offset."""
        from agent import tools as tools_mod
        kb_dir = tmp_path / "knowledge"
        canon_dir = tmp_path / "canon"
        kb_dir.mkdir()
        canon_dir.mkdir()
        big = "# Big\n\n" + ("Lorem ipsum dolor sit amet. " * 4000)
        (kb_dir / "big.md").write_text(big)
        tools = tools_mod.KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)
        first_call = tools.read_knowledge_section("big.md", "Big")
        # Extract suggested next offset from the marker
        import re as _re
        m = _re.search(r"offset=(\d+)", first_call.splitlines()[0])
        assert m is not None
        next_offset = m.group(1)
        # Reset budget so the second call isn't refused
        tools_mod.reset_budget()
        second_call = tools.read_knowledge_section("big.md", "Big", next_offset)
        first_line = second_call.splitlines()[0]
        assert "LOADED" in first_line
        assert f"offset {int(next_offset):,}" in first_line or "(COMPLETE)" in first_line

    def test_no_mid_word_cut(self, tmp_path):
        """Truncated sections never cut mid-word.

        Uses a section big enough to force truncation, then verifies that
        the very last token of the body is a complete dictionary word from
        the input (not a partial slice of one).
        """
        from agent import tools as tools_mod
        kb_dir = tmp_path / "knowledge"
        canon_dir = tmp_path / "canon"
        kb_dir.mkdir()
        canon_dir.mkdir()
        big = "# Big\n\n" + ("Lorem ipsum dolor sit amet consectetur. " * 4000)
        (kb_dir / "big.md").write_text(big)
        tools = tools_mod.KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)
        result = tools.read_knowledge_section("big.md", "Big")
        body = "\n".join(result.splitlines()[1:]).strip()
        valid_words = {"Lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
                       "Lorem.", "ipsum.", "dolor.", "sit.", "amet.", "consectetur."}
        # Strip trailing punctuation for word check
        last_token = body.rstrip().split()[-1].rstrip(".,;:!?")
        assert last_token in {w.rstrip(".,;:!?") for w in valid_words}, \
            f"Body ended with partial word: {last_token!r}"


class TestSearchKnowledge:
    """Test search_knowledge tool (section-level results)."""

    def test_no_index(self, kb_tools):
        result = kb_tools.search_knowledge("architecture")
        assert "not available" in result.lower()

    def test_with_mock_index(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        mock_index = MagicMock()
        mock_index.search.return_value = [
            {
                "filename": "test.md",
                "source": "knowledge",
                "heading": "Architecture",
                "summary": "System overview",
                "score": 0.85,
                "chunk_index": 0,
                "section_count": 3,
            }
        ]
        tools = KBTools(kb_index=mock_index, kb_dir=kb_dir, canon_dir=canon_dir)
        result = tools.search_knowledge("architecture")
        assert "test.md" in result
        assert "Architecture" in result

    def test_empty_results(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        mock_index = MagicMock()
        mock_index.search.return_value = []
        tools = KBTools(kb_index=mock_index, kb_dir=kb_dir, canon_dir=canon_dir)
        result = tools.search_knowledge("nonexistent topic xyz")
        assert "No matches" in result

    def test_renders_position_suffix(self, temp_dirs):
        """Search hits include [chunk_index+1/section_count] position suffix."""
        kb_dir, canon_dir = temp_dirs
        mock_index = MagicMock()
        mock_index.search.return_value = [
            {
                "filename": "test.md",
                "source": "knowledge",
                "heading": "Architecture",
                "summary": "System overview",
                "score": 0.85,
                "chunk_index": 0,
                "section_count": 3,
            },
            {
                "filename": "test.md",
                "source": "knowledge",
                "heading": "Data Flow",
                "summary": "How data moves",
                "score": 0.72,
                "chunk_index": 2,
                "section_count": 3,
            },
        ]
        tools = KBTools(kb_index=mock_index, kb_dir=kb_dir, canon_dir=canon_dir)
        result = tools.search_knowledge("architecture")
        assert "[1/3]" in result
        assert "[3/3]" in result


class TestSaveKnowledge:
    """Test save_knowledge tool. P0.4: all writes land under wiki/."""

    def test_saves_file_into_wiki_subdir(self, kb_tools, temp_dirs):
        """Bare filename is silently normalized to wiki/<filename>."""
        kb_dir, _ = temp_dirs
        result = kb_tools.save_knowledge("new_file.md", "# New File\n\nContent here.", tags="test")
        assert "Saved" in result
        # File must NOT be at root, must be under wiki/
        assert not (kb_dir / "new_file.md").exists()
        assert (kb_dir / "wiki" / "new_file.md").exists()

    def test_blocks_canon_write(self, kb_tools, temp_dirs):
        result = kb_tools.save_knowledge("canon_doc.md", "overwriting canon", tags="bad")
        assert "canon" in result.lower()

    def test_blocks_explicit_canon_path(self, kb_tools, temp_dirs):
        """Explicit canon/ prefix is rejected outright with a clear message."""
        result = kb_tools.save_knowledge("canon/whatever.md", "x", tags="")
        assert "canon" in result.lower()
        assert "read-only" in result.lower()

    def test_blocks_raw_path(self, kb_tools, temp_dirs):
        """raw/ prefix is rejected — raw is read-only source material."""
        kb_dir, _ = temp_dirs
        result = kb_tools.save_knowledge("raw/notes.md", "x", tags="")
        assert "raw" in result.lower()
        assert "read-only" in result.lower()
        # File must not have been created
        assert not (kb_dir / "raw" / "notes.md").exists()
        assert not (kb_dir / "wiki" / "raw" / "notes.md").exists()

    def test_blocks_path_traversal(self, kb_tools, temp_dirs):
        result = kb_tools.save_knowledge("../etc/passwd", "x", tags="")
        assert "Cannot save" in result
        assert ".." in result

    def test_blocks_absolute_path(self, kb_tools, temp_dirs):
        result = kb_tools.save_knowledge("/etc/passwd", "x", tags="")
        assert "Cannot save" in result

    def test_blocks_empty_filename(self, kb_tools, temp_dirs):
        result = kb_tools.save_knowledge("", "x", tags="")
        assert "Cannot save" in result
        result = kb_tools.save_knowledge("   ", "x", tags="")
        assert "Cannot save" in result

    def test_explicit_wiki_prefix_is_idempotent(self, kb_tools, temp_dirs):
        """save_knowledge('wiki/foo.md') writes to wiki/foo.md, not wiki/wiki/foo.md."""
        kb_dir, _ = temp_dirs
        result = kb_tools.save_knowledge(
            "wiki/cortisol.md", "## Topic\n\nbody", tags=""
        )
        assert "Saved" in result
        assert (kb_dir / "wiki" / "cortisol.md").exists()
        assert not (kb_dir / "wiki" / "wiki" / "cortisol.md").exists()

    def test_creates_parent_dirs_under_wiki(self, kb_tools, temp_dirs):
        """Nested paths are placed under wiki/ with parents created."""
        kb_dir, _ = temp_dirs
        result = kb_tools.save_knowledge("sub/dir/new.md", "# Nested\n\nContent", tags="test")
        assert "Saved" in result
        assert (kb_dir / "wiki" / "sub" / "dir" / "new.md").exists()
        assert not (kb_dir / "sub" / "dir" / "new.md").exists()


class TestSaveKnowledgeNormalizer:
    """Direct unit tests for KBTools._normalize_wiki_path()."""

    def test_bare_filename_gets_wiki_prefix(self):
        path, err = KBTools._normalize_wiki_path("foo.md")
        assert err is None
        assert path == "wiki/foo.md"

    def test_already_under_wiki_stays(self):
        path, err = KBTools._normalize_wiki_path("wiki/foo.md")
        assert err is None
        assert path == "wiki/foo.md"

    def test_nested_path_goes_under_wiki(self):
        path, err = KBTools._normalize_wiki_path("neuro/dmn.md")
        assert err is None
        assert path == "wiki/neuro/dmn.md"

    def test_raw_prefix_rejected(self):
        path, err = KBTools._normalize_wiki_path("raw/notes.md")
        assert path is None
        assert "raw" in err.lower()

    def test_canon_prefix_rejected(self):
        path, err = KBTools._normalize_wiki_path("canon/whatever.md")
        assert path is None
        assert "canon" in err.lower()

    def test_traversal_rejected(self):
        path, err = KBTools._normalize_wiki_path("../escape.md")
        assert path is None
        assert ".." in err
        path, err = KBTools._normalize_wiki_path("a/../b.md")
        assert path is None

    def test_absolute_rejected(self):
        path, err = KBTools._normalize_wiki_path("/abs.md")
        assert path is None

    def test_empty_rejected(self):
        path, err = KBTools._normalize_wiki_path("")
        assert path is None
        path, err = KBTools._normalize_wiki_path("   ")
        assert path is None
        path, err = KBTools._normalize_wiki_path(None)
        assert path is None

    def test_backslashes_normalized(self):
        path, err = KBTools._normalize_wiki_path("sub\\dir\\file.md")
        assert err is None
        assert path == "wiki/sub/dir/file.md"

    # R9: leading `knowledge/` prefix strip. The model carries this over
    # from canonical tool-output paths (`knowledge:wiki/foo.md` sometimes
    # printed with a slash). Without the strip, the tier check below
    # prepended `wiki/` and files landed in `knowledge/wiki/knowledge/wiki/`.
    def test_knowledge_wiki_prefix_collapsed(self):
        path, err = KBTools._normalize_wiki_path("knowledge/wiki/cortisol.md")
        assert err is None
        assert path == "wiki/cortisol.md"

    def test_knowledge_memory_prefix_collapsed(self):
        path, err = KBTools._normalize_wiki_path("knowledge/memory/journal.md")
        assert err is None
        assert path == "memory/journal.md"

    def test_knowledge_raw_still_refused(self):
        """After stripping `knowledge/`, the `raw/` refusal must still fire."""
        path, err = KBTools._normalize_wiki_path("knowledge/raw/notes.md")
        assert path is None
        assert "raw" in err.lower()

    def test_knowledge_canon_still_refused(self):
        """After stripping `knowledge/`, the `canon/` refusal must still fire."""
        path, err = KBTools._normalize_wiki_path("knowledge/canon/quote.md")
        assert path is None
        assert "canon" in err.lower()

    def test_only_one_leading_knowledge_stripped(self):
        """Deeply nested input (already-corrupted path from the bug) should
        strip exactly one leading `knowledge/` — not recursively unwind the
        whole mistake, since that would hide genuine nesting."""
        path, err = KBTools._normalize_wiki_path(
            "knowledge/wiki/knowledge/wiki/legacy.md"
        )
        assert err is None
        assert path == "wiki/knowledge/wiki/legacy.md"

    def test_bare_knowledge_md_treated_as_wiki_page(self):
        """A file literally named `knowledge.md` with no subpath must not
        be stripped to empty — it's a legit wiki page title."""
        path, err = KBTools._normalize_wiki_path("knowledge.md")
        assert err is None
        assert path == "wiki/knowledge.md"


class TestBudgetReset:
    """Test budget reset and token setting."""

    def test_reset_clears_loads(self):
        from agent import tools as tools_mod
        tools_mod._current_kb_loads = 5
        reset_budget()
        assert tools_mod._current_kb_loads == 0

    def test_set_available_tokens(self):
        from agent import tools as tools_mod
        set_available_tokens(50000)
        assert tools_mod._current_available_tokens == 50000

    def test_reset_clears_tool_tokens_used(self):
        from agent import tools as tools_mod
        tools_mod._current_tool_tokens_used = 12345
        reset_budget()
        assert tools_mod._current_tool_tokens_used == 0

    def test_set_context_window(self):
        from agent import tools as tools_mod
        tools_mod.set_context_window(200000)
        assert tools_mod._current_context_window == 200000

    def test_get_budget_state_shape(self):
        from agent import tools as tools_mod
        state = tools_mod.get_budget_state()
        for key in (
            "kb_loads",
            "kb_loads_max",
            "available_tokens",
            "context_window",
            "tool_tokens_used",
            "tool_token_cap",
        ):
            assert key in state


class TestAdaptiveToolBudget:
    """Adaptive token-aware refusal beyond the naive load count."""

    def test_refuses_when_tool_tokens_exceed_cap(self, kb_tools):
        """Once accumulated tool tokens exceed 50% of context window, refuse."""
        from agent import tools as tools_mod
        tools_mod.set_context_window(20000)  # cap = 10,000
        tools_mod._current_tool_tokens_used = 11000
        result = kb_tools.read_knowledge_section("test.md", "Architecture")
        assert "REFUSED" in result
        assert "tool budget" in result.lower()
        # Refusal must include both the used and cap numbers (honest).
        assert "11,000" in result
        assert "10,000" in result

    def test_under_cap_loads_succeed(self, kb_tools):
        """Loads succeed when accumulated tool tokens are well under cap."""
        from agent import tools as tools_mod
        tools_mod.set_context_window(200000)  # cap = 100,000
        tools_mod._current_tool_tokens_used = 1000
        result = kb_tools.read_knowledge_section("test.md", "Architecture")
        assert "REFUSED" not in result
        assert "LOADED" in result


class TestFileTools:
    """Test FileTools (kept from original)."""

    def test_read_file(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        tools = FileTools(kb_dir, canon_dir)
        result = tools.read_file("test.md")
        assert result.success
        assert "Architecture" in result.output

    def test_read_file_not_found(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        tools = FileTools(kb_dir, canon_dir)
        result = tools.read_file("nonexistent.md")
        assert not result.success

    def test_write_file(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        tools = FileTools(kb_dir, canon_dir)
        result = tools.write_file("new.md", "content")
        assert result.success
        assert (kb_dir / "new.md").read_text() == "content"

    def test_list_files(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        tools = FileTools(kb_dir, canon_dir)
        result = tools.list_files("knowledge")
        assert result.success


class TestGraphNeighbors:
    """Test graph_neighbors tool."""

    def test_no_graph(self, tmp_path):
        """Returns message when no graph available."""
        from knowledge.index import KBIndex
        index = KBIndex.__new__(KBIndex)
        index.graph = None
        tools = KBTools(kb_index=index)
        result = tools.graph_neighbors("test.md")
        assert "No knowledge graph" in result

    def test_finds_neighbors(self, tmp_path):
        """Finds neighbors for a file's sections."""
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="ai.md > Intro",
                   filename="ai.md", heading="Intro",
                   attributes={"source": "knowledge"})
        c2 = Node(id="c2", node_type=NodeType.CHUNK, name="ai.md > Methods",
                   filename="ai.md", heading="Methods",
                   attributes={"source": "knowledge"})
        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_edge(Edge(source_id="c1", target_id="c2",
                           edge_type=EdgeType.SIMILAR, weight=0.85))

        index = KBIndex.__new__(KBIndex)
        index.graph = graph
        tools = KBTools(kb_index=index)

        result = tools.graph_neighbors("ai.md")
        assert "Intro" in result
        assert "similar" in result.lower()

    def test_heading_filter(self, tmp_path):
        """Filters by heading when provided."""
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="ai.md > Intro",
                   filename="ai.md", heading="Intro",
                   attributes={"source": "knowledge"})
        graph.add_node(c1)

        index = KBIndex.__new__(KBIndex)
        index.graph = graph
        tools = KBTools(kb_index=index)

        result = tools.graph_neighbors("ai.md", "Intro")
        assert "Intro" in result

    def test_file_not_found(self, tmp_path):
        """Returns message when file has no sections."""
        from knowledge.graph import KnowledgeGraph
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        index = KBIndex.__new__(KBIndex)
        index.graph = graph
        tools = KBTools(kb_index=index)

        result = tools.graph_neighbors("nonexistent.md")
        assert "No sections" in result


class TestGraphTraverse:
    """Test graph_traverse tool."""

    def test_no_graph(self, tmp_path):
        from knowledge.index import KBIndex
        index = KBIndex.__new__(KBIndex)
        index.graph = None
        tools = KBTools(kb_index=index)
        result = tools.graph_traverse("test.md")
        assert "No knowledge graph" in result

    def test_traversal_with_depth(self, tmp_path):
        """BFS traversal from a section."""
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="ai.md > Intro",
                   filename="ai.md", heading="Intro", attributes={"source": "knowledge"})
        c2 = Node(id="c2", node_type=NodeType.CHUNK, name="math.md > Algebra",
                   filename="math.md", heading="Algebra", attributes={"source": "canon"})
        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_edge(Edge(source_id="c1", target_id="c2",
                           edge_type=EdgeType.CROSS_DOMAIN, weight=0.7))

        index = KBIndex.__new__(KBIndex)
        index.graph = graph
        tools = KBTools(kb_index=index)

        result = tools.graph_traverse("ai.md", "Intro", "2")
        assert "depth 2" in result.lower() or "depth" in result.lower()

    def test_invalid_depth_defaults_to_2(self, tmp_path):
        """Invalid depth value defaults to 2."""
        from knowledge.graph import KnowledgeGraph, Node, NodeType
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="ai.md > Intro",
                   filename="ai.md", heading="Intro", attributes={"source": "knowledge"})
        graph.add_node(c1)

        index = KBIndex.__new__(KBIndex)
        index.graph = graph
        index.table = None  # No table for fallback search
        tools = KBTools(kb_index=index)

        result = tools.graph_traverse("ai.md", "Intro", "abc")
        assert "depth 2" in result.lower() or "No" in result or "Intro" in result


class TestGraphSearch:
    """Test graph_search tool (hybrid vector + graph)."""

    def test_no_graph(self, tmp_path):
        from knowledge.index import KBIndex
        index = KBIndex.__new__(KBIndex)
        index.graph = None
        tools = KBTools(kb_index=index)
        result = tools.graph_search("test")
        assert "No knowledge graph" in result

    def test_search_with_graph(self, tmp_path):
        """Hybrid search combines vector results with graph neighbors."""
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="ai.md > Intro",
                   filename="ai.md", heading="Intro", attributes={"source": "knowledge"})
        c2 = Node(id="c2", node_type=NodeType.CHUNK, name="ai.md > Methods",
                   filename="ai.md", heading="Methods", attributes={"source": "knowledge"})
        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_edge(Edge(source_id="c1", target_id="c2",
                           edge_type=EdgeType.SIMILAR, weight=0.85))

        # Mock index with search
        index = KBIndex.__new__(KBIndex)
        index.graph = graph

        # Mock search method
        def mock_search(query, top_k=5):
            return [{"path": "ai.md", "content": "AI intro", "score": 0.9,
                      "heading": "Intro", "filename": "ai.md"}]
        index.search = mock_search

        tools = KBTools(kb_index=index)
        result = tools.graph_search("artificial intelligence")
        assert "ai.md" in result


class TestGraphStats:
    """Test graph_stats tool."""

    def test_no_graph(self, tmp_path):
        """Returns message when no graph available."""
        from knowledge.index import KBIndex
        index = KBIndex.__new__(KBIndex)
        index.graph = None
        tools = KBTools(kb_index=index)
        result = tools.graph_stats()
        assert "No knowledge graph" in result

    def test_stats_output(self, tmp_path):
        """Stats shows node/edge counts and connectivity."""
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="ai.md > Intro",
                   filename="ai.md", heading="Intro",
                   attributes={"source": "knowledge"})
        c2 = Node(id="c2", node_type=NodeType.CHUNK, name="ai.md > Methods",
                   filename="ai.md", heading="Methods",
                   attributes={"source": "knowledge"})
        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_edge(Edge(source_id="c1", target_id="c2",
                           edge_type=EdgeType.SIMILAR, weight=0.85))

        index = KBIndex.__new__(KBIndex)
        index.graph = graph
        tools = KBTools(kb_index=index)

        result = tools.graph_stats()
        assert "Nodes: 2" in result
        assert "Edges: 1" in result
        assert "similar" in result.lower()
        assert "Connectivity" in result

    def test_stats_shows_inter_file_and_cross_domain(self, tmp_path):
        """Stats reports inter_file and cross_domain edge types with inter/intra ratio."""
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="ai.md > Intro",
                   filename="ai.md", heading="Intro",
                   attributes={"source": "knowledge"})
        c2 = Node(id="c2", node_type=NodeType.CHUNK, name="neuro.md > Findings",
                   filename="neuro.md", heading="Findings",
                   attributes={"source": "knowledge"})
        c3 = Node(id="c3", node_type=NodeType.CHUNK, name="ethos.md > Principles",
                   filename="ethos.md", heading="Principles",
                   attributes={"source": "canon"})

        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_node(c3)

        # INTER_FILE: same source, different file
        graph.add_edge(Edge(source_id="c1", target_id="c2",
                           edge_type=EdgeType.INTER_FILE, weight=0.62))
        # CROSS_DOMAIN: different source
        graph.add_edge(Edge(source_id="c1", target_id="c3",
                           edge_type=EdgeType.CROSS_DOMAIN, weight=0.65))
        # SIMILAR: intra-file (need another node in same file)
        c4 = Node(id="c4", node_type=NodeType.CHUNK, name="ai.md > Methods",
                   filename="ai.md", heading="Methods",
                   attributes={"source": "knowledge"})
        graph.add_node(c4)
        graph.add_edge(Edge(source_id="c1", target_id="c4",
                           edge_type=EdgeType.SIMILAR, weight=0.85))

        index = KBIndex.__new__(KBIndex)
        index.graph = graph
        tools = KBTools(kb_index=index)

        result = tools.graph_stats()
        assert "inter_file" in result
        assert "cross_domain" in result
        assert "similar" in result.lower()
        assert "Inter-file/cross-domain edges: 2" in result
        assert "Intra-file edges: 1" in result


class TestSaveKnowledgeFormatting:
    """Test save_knowledge auto-formatting: frontmatter, H1, TOC, dividers.

    P0.4: all writes land under wiki/ even when filename is bare. The
    catalog (index.md) and audit log (log.md) still live at kb_dir root —
    they are agent-managed metadata, not knowledge content.
    """

    def test_save_adds_frontmatter(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        tools = KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)
        result = tools.save_knowledge("test-file.md", "## Section One\n\nContent here.", tags="test, demo")
        assert "Saved" in result
        content = (kb_dir / "wiki" / "test-file.md").read_text()
        assert "---" in content
        assert "tags:" in content
        assert "updated:" in content
        assert "date-created:" not in content
        assert "last-modified:" not in content

    def test_save_adds_h1_heading(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        tools = KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)
        tools.save_knowledge("my-topic.md", "## Introduction\n\nHello world.")
        content = (kb_dir / "wiki" / "my-topic.md").read_text()
        assert "# my-topic" in content

    def test_save_adds_toc(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        tools = KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)
        tools.save_knowledge("toc-test.md", "## First\n\nText.\n\n## Second\n\nMore text.")
        content = (kb_dir / "wiki" / "toc-test.md").read_text()
        assert "[First]" in content
        assert "[Second]" in content

    def test_save_adds_section_dividers(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        tools = KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)
        tools.save_knowledge("dividers.md", "## Alpha\n\nA\n\n## Beta\n\nB")
        content = (kb_dir / "wiki" / "dividers.md").read_text()
        assert "---" in content

    def test_save_creates_log_at_root(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        tools = KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)
        tools.save_knowledge("logged.md", "## Content\n\nHello.", tags="test")
        # log.md is agent-managed metadata, stays at root
        assert (kb_dir / "log.md").exists()

    def test_save_creates_index_at_root(self, temp_dirs):
        kb_dir, canon_dir = temp_dirs
        tools = KBTools(kb_index=None, kb_dir=kb_dir, canon_dir=canon_dir)
        tools.save_knowledge("indexed.md", "## Topic\n\nSome content.")
        # index.md catalog is agent-managed metadata, stays at root
        assert (kb_dir / "index.md").exists()
        index_content = (kb_dir / "index.md").read_text()
        # The catalog must reference the new wiki/ path
        assert "indexed.md" in index_content
        assert "Knowledge (editable)" in index_content


class TestHeadingNormalization:
    """Test heading slug normalization for graph tools."""

    def test_normalize_spaces(self):
        from agent.tools import _normalize_heading
        assert _normalize_heading("Marcus Aurelius") == "marcus-aurelius"

    def test_normalize_hyphens(self):
        from agent.tools import _normalize_heading
        assert _normalize_heading("marcus-aurelius") == "marcus-aurelius"

    def test_normalize_special_chars(self):
        from agent.tools import _normalize_heading
        assert _normalize_heading("What's New?!") == "what-s-new"

    def test_normalize_mixed(self):
        from agent.tools import _normalize_heading
        assert _normalize_heading("Data Flow & Architecture") == "data-flow-architecture"


class TestGraphTraverseSkipsParentChild:
    """Test that graph_traverse skips PARENT_CHILD edges and caps output."""

    def test_traverse_skips_parent_child(self, tmp_path):
        """graph_traverse should not follow PARENT_CHILD edges."""
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="file.md > Intro",
                   filename="file.md", heading="Intro",
                   attributes={"source": "knowledge"})
        c2 = Node(id="c2", node_type=NodeType.CHUNK, name="file.md > Subsection",
                   filename="file.md", heading="Subsection",
                   attributes={"source": "knowledge"})
        c3 = Node(id="c3", node_type=NodeType.CHUNK, name="other.md > Related",
                   filename="other.md", heading="Related",
                   attributes={"source": "knowledge"})
        graph.add_node(c1)
        graph.add_node(c2)
        graph.add_node(c3)

        # PARENT_CHILD: structural hierarchy — should be SKIPPED in traversal
        graph.add_edge(Edge(source_id="c1", target_id="c2",
                           edge_type=EdgeType.PARENT_CHILD, weight=1.0))
        # INTER_FILE: semantic — should be FOLLOWED
        graph.add_edge(Edge(source_id="c1", target_id="c3",
                           edge_type=EdgeType.INTER_FILE, weight=0.65))

        index = KBIndex.__new__(KBIndex)
        index.graph = graph
        index._file_count = 0
        tools = KBTools(kb_index=index)

        result = tools.graph_traverse("file.md", "Intro", "2")

        # Should find INTER_FILE edge to c3 but NOT PARENT_CHILD edge to c2
        assert "inter_file" in result.lower()
        assert "other.md" in result
        assert "parent_child" not in result.lower() or "Related" in result
        # c2 (Subsection) should NOT appear as a traversal result
        assert "Subsection" not in result

    def test_traverse_caps_at_max_results(self, tmp_path):
        """graph_traverse truncates output when too many results."""
        from knowledge.graph import KnowledgeGraph, Node, Edge, EdgeType, NodeType
        from knowledge.index import KBIndex

        graph = KnowledgeGraph(tmp_path / "graph.json")
        c1 = Node(id="c1", node_type=NodeType.CHUNK, name="file.md > Root",
                   filename="file.md", heading="Root",
                   attributes={"source": "knowledge"})
        graph.add_node(c1)

        # Add 200 INTER_FILE neighbors
        for i in range(200):
            node = Node(id=f"n{i}", node_type=NodeType.CHUNK,
                       name=f"other{i}.md > Section",
                       filename=f"other{i}.md", heading="Section",
                       attributes={"source": "knowledge"})
            graph.add_node(node)
            graph.add_edge(Edge(source_id="c1", target_id=f"n{i}",
                               edge_type=EdgeType.INTER_FILE, weight=0.6))

        index = KBIndex.__new__(KBIndex)
        index.graph = graph
        index._file_count = 0
        tools = KBTools(kb_index=index)

        result = tools.graph_traverse("file.md", "Root", "2")
        # D1: pagination footer replaces the old "truncated" notice — the
        # caller now sees an explicit offset/limit hint instead.
        assert "showing edges 0-100 of 200" in result
        assert "call again with offset=100" in result


# ---------------------------------------------------------------------------
# P1.1: filename resolver waterfall + index-backed section enumeration +
#        fuzzy not-found suggestions.
# ---------------------------------------------------------------------------

class _StubIndex:
    """Minimal KBIndex stand-in exposing only what the resolver and
    section enumeration paths consume."""

    def __init__(self, indexed_files=None, sections_by_file=None,
                 heading_tree_by_file=None):
        self._files = indexed_files or []
        self._sections = sections_by_file or {}
        self._trees = heading_tree_by_file or {}

    def list_indexed_filenames(self):
        return list(self._files)

    def list_sections(self, filename, source="knowledge"):
        return list(self._sections.get((filename, source), []))

    def get_heading_tree(self, filename, source="knowledge"):
        return self._trees.get((filename, source))


class TestResolveKbFilename:
    """P1.1: _resolve_kb_filename should accept loose filenames."""

    def test_exact_path_in_knowledge(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (kb / "wiki").mkdir()
        (kb / "wiki" / "page.md").write_text("# X")
        tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
        result = tools._resolve_kb_filename("wiki/page.md")
        assert result is not None
        rel, source, path = result
        assert rel == "wiki/page.md"
        assert source == "knowledge"
        assert path.exists()

    def test_exact_path_in_canon(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (canon / "doc.md").write_text("# Doc")
        tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
        result = tools._resolve_kb_filename("doc.md")
        assert result is not None
        assert result[1] == "canon"

    def test_strip_knowledge_prefix(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (kb / "wiki").mkdir()
        (kb / "wiki" / "page.md").write_text("# X")
        tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
        # Agent passes the redundant 'knowledge/' prefix
        result = tools._resolve_kb_filename("knowledge/wiki/page.md")
        assert result is not None
        assert result[0] == "wiki/page.md"
        assert result[1] == "knowledge"

    def test_strip_canon_prefix(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (canon / "doc.md").write_text("# Doc")
        tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
        result = tools._resolve_kb_filename("canon/doc.md")
        assert result is not None
        assert result[0] == "doc.md"
        assert result[1] == "canon"

    def test_substring_match_basename(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (kb / "wiki").mkdir()
        (kb / "wiki" / "rare-page.md").write_text("# X")
        index = _StubIndex(indexed_files=[
            {"filename": "wiki/rare-page.md", "source": "knowledge", "tier": "wiki"},
            {"filename": "wiki/other.md", "source": "knowledge", "tier": "wiki"},
        ])
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        # Bare basename — no path provided
        result = tools._resolve_kb_filename("rare-page.md")
        assert result is not None
        assert result[0] == "wiki/rare-page.md"

    def test_substring_match_ambiguous_returns_none(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        index = _StubIndex(indexed_files=[
            {"filename": "wiki/page.md", "source": "knowledge", "tier": "wiki"},
            {"filename": "raw/page.md", "source": "knowledge", "tier": "raw"},
        ])
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        # Two candidates with same basename → refuse to auto-resolve
        assert tools._resolve_kb_filename("page.md") is None

    def test_path_traversal_rejected(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
        assert tools._resolve_kb_filename("../etc/passwd") is None
        assert tools._resolve_kb_filename("wiki/../../etc/passwd") is None

    def test_empty_returns_none(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
        assert tools._resolve_kb_filename("") is None
        assert tools._resolve_kb_filename("   ") is None


class TestSuggestFilenames:
    """P1.1: _suggest_filenames returns near-misses for not-found errors."""

    def test_basename_typo_suggested(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        index = _StubIndex(indexed_files=[
            {"filename": "wiki/cortisol.md", "source": "knowledge", "tier": "wiki"},
            {"filename": "wiki/dopamine.md", "source": "knowledge", "tier": "wiki"},
        ])
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        suggestions = tools._suggest_filenames("cortizol.md")
        assert "wiki/cortisol.md" in suggestions

    def test_no_index_returns_empty(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
        assert tools._suggest_filenames("cortizol.md") == []

    def test_no_match_returns_empty(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        index = _StubIndex(indexed_files=[
            {"filename": "wiki/cortisol.md", "source": "knowledge", "tier": "wiki"},
        ])
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        # Wildly different — below the 0.5 ratio cutoff
        assert tools._suggest_filenames("zzzzzzzzzzzz.md") == []


class TestReadKnowledgeWithResolver:
    """P1.1: read_knowledge uses the resolver waterfall."""

    def test_loose_basename_resolves(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (kb / "wiki").mkdir()
        (kb / "wiki" / "rare-page.md").write_text(
            "# Title\n\nBody.\n\n## Sub\n\nMore."
        )
        index = _StubIndex(
            indexed_files=[
                {"filename": "wiki/rare-page.md", "source": "knowledge", "tier": "wiki"},
            ],
            heading_tree_by_file={
                ("wiki/rare-page.md", "knowledge"): "tree-text",
            },
        )
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        result = tools.read_knowledge("rare-page.md")
        assert "FILE NOT FOUND" not in result
        assert "tree-text" in result

    def test_unknown_file_includes_suggestions(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        index = _StubIndex(indexed_files=[
            {"filename": "wiki/cortisol.md", "source": "knowledge", "tier": "wiki"},
        ])
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        result = tools.read_knowledge("cortizol.md")
        assert "FILE NOT FOUND" in result
        assert "Did you mean" in result
        assert "wiki/cortisol.md" in result


class TestReadKnowledgeSectionWithResolver:
    """P1.1: read_knowledge_section honors resolver + index section count."""

    def test_resolver_finds_relocated_file(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (kb / "wiki").mkdir()
        (kb / "wiki" / "moved.md").write_text(
            "# Title\n\nIntro body.\n\n## Sub\n\nSub body.\n"
        )
        index = _StubIndex(indexed_files=[
            {"filename": "wiki/moved.md", "source": "knowledge", "tier": "wiki"},
        ])
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        # Agent passes bare basename — resolver promotes it to wiki/moved.md
        result = tools.read_knowledge_section("moved.md", "Title")
        assert "FILE NOT FOUND" not in result
        # A2: SECTION header reports the canonical <source>:<relpath> form.
        assert "[SECTION: knowledge:wiki/moved.md" in result

    def test_marker_uses_indexed_section_count(self, tmp_path):
        """When the index reports more sections than the disk-time chunker,
        the marker should reflect the indexed count (the agent saw that
        in read_knowledge / search_knowledge)."""
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (kb / "wiki").mkdir()
        (kb / "wiki" / "page.md").write_text(
            "# Title\n\nBody.\n\n## Sub\n\nMore.\n"
        )
        index = _StubIndex(
            indexed_files=[
                {"filename": "wiki/page.md", "source": "knowledge", "tier": "wiki"},
            ],
            sections_by_file={
                ("wiki/page.md", "knowledge"): [
                    {"heading": "Title", "chunk_index": 0, "token_count": 10,
                     "summary": "", "tier": "wiki"},
                    {"heading": "Title > Sub", "chunk_index": 1,
                     "token_count": 5, "summary": "", "tier": "wiki"},
                    {"heading": "Title > Extra", "chunk_index": 2,
                     "token_count": 5, "summary": "", "tier": "wiki"},
                ],
            },
        )
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        result = tools.read_knowledge_section("wiki/page.md", "Title")
        # Indexed total is 3, even though disk re-chunk would yield 1
        assert "/3" in result.split("LOADED")[0]

    def test_section_not_found_lists_indexed_headings(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (kb / "wiki").mkdir()
        (kb / "wiki" / "page.md").write_text("# Title\n\nBody.\n")
        index = _StubIndex(
            indexed_files=[
                {"filename": "wiki/page.md", "source": "knowledge", "tier": "wiki"},
            ],
            sections_by_file={
                ("wiki/page.md", "knowledge"): [
                    {"heading": "Title", "chunk_index": 0, "token_count": 10,
                     "summary": "", "tier": "wiki"},
                    {"heading": "Title > Phantom", "chunk_index": 1,
                     "token_count": 5, "summary": "", "tier": "wiki"},
                ],
            },
        )
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        result = tools.read_knowledge_section("wiki/page.md", "nonexistent")
        assert "SECTION NOT FOUND" in result
        # Hint comes from indexed sections, not the disk re-chunk
        assert "Title > Phantom" in result

    def test_unknown_file_gives_fuzzy_hint(self, tmp_path):
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        index = _StubIndex(indexed_files=[
            {"filename": "wiki/cortisol.md", "source": "knowledge", "tier": "wiki"},
        ])
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        result = tools.read_knowledge_section("cortizol.md", "anything")
        assert "FILE NOT FOUND" in result
        assert "Did you mean" in result
        assert "wiki/cortisol.md" in result


# ---------------------------------------------------------------------------
# P1.2: heading-leak defense.
#   - Tool parser decodes \n / \t / \r in quoted args
#   - save_knowledge sanitizes content escapes when content is single-line
#   - chunker splits on literal escape tokens that survive into the heading
# ---------------------------------------------------------------------------

class TestSanitizeContentEscapes:
    """save_knowledge._sanitize_content_escapes converts literal \\n into
    real newlines when content arrives single-line with embedded escapes."""

    def test_literal_escapes_decoded_when_single_line(self):
        from agent.tools import KBTools
        out = KBTools._sanitize_content_escapes(
            "# heading\\n\\n## sub\\n- bullet\\n"
        )
        assert "\n" in out
        assert "\\n" not in out
        # Decoded shape — heading on its own line followed by blank line + sub
        assert out.startswith("# heading\n\n## sub")

    def test_real_newlines_left_alone(self):
        from agent.tools import KBTools
        original = "# heading\n\n## sub\n- bullet\n"
        out = KBTools._sanitize_content_escapes(original)
        assert out == original

    def test_no_backslash_passes_through(self):
        from agent.tools import KBTools
        original = "no escapes here at all"
        assert KBTools._sanitize_content_escapes(original) == original

    def test_does_not_decode_when_real_newlines_present(self):
        """Heuristic: if the content already has multiple real newlines we
        assume it's well-formed and leave any literal `\\n` alone (could be
        a code sample or literal `\\n` reference)."""
        from agent.tools import KBTools
        original = "# heading\n\nsome text\nmore text\nliteral \\n stays"
        out = KBTools._sanitize_content_escapes(original)
        assert "literal \\n stays" in out

    def test_empty_content_returned_as_is(self):
        from agent.tools import KBTools
        assert KBTools._sanitize_content_escapes("") == ""
        assert KBTools._sanitize_content_escapes(None) is None


class TestSaveKnowledgeRescuesEscapedContent:
    """End-to-end: save_knowledge writes a properly multi-line file even
    when the model passed a single-line escaped string."""

    def test_save_decodes_literal_newlines_on_disk(self, tmp_path):
        from agent.tools import KBTools
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)

        result = tools.save_knowledge(
            "leak-test.md",
            "# Title\\n\\n## Sub\\n- bullet one\\n- bullet two\\n",
            "test",
        )
        assert "Saved" in result or "saved" in result.lower()
        on_disk = (kb / "wiki" / "leak-test.md").read_text(encoding="utf-8")
        assert "\\n" not in on_disk
        # Real headings on their own lines
        assert "\n# Title" in on_disk or on_disk.startswith("# Title") or "\n# leak-test" in on_disk
        assert "\n## Sub" in on_disk
        # Bullets rendered as a real list
        assert "\n- bullet one" in on_disk


class TestChunkerHeadingLeakDefense:
    """If a heading line contains literal `\\n` tokens, the chunker must
    NOT swallow the entire body into the heading text."""

    def test_split_on_level_handles_collapsed_body(self):
        from knowledge.chunker import _split_on_level
        # The whole body is one physical line; the heading regex captures
        # everything after `# ` greedily.
        body = "# daily-logs\\n\\n## 2026-04-17\\n- did the thing"
        sections = _split_on_level(body, level=1)
        # We must end up with at least one section whose heading is just
        # 'daily-logs' (not the full collapsed body)
        assert any(h == "daily-logs" for h, _ in sections), sections

    def test_build_heading_tree_handles_collapsed_body(self):
        from knowledge.chunker import build_heading_tree
        body = "# daily-logs\\n\\n## 2026-04-17\\n- did the thing"
        tree = build_heading_tree(body, "daily.md")
        # Root has one child whose heading is the cleaned 'daily-logs'
        assert tree.children, "expected a child node for the H1"
        assert tree.children[0].heading == "daily-logs"

    def test_clean_heading_passes_through_unchanged(self):
        from knowledge.chunker import _split_on_level
        body = "# heading\n\nbody line\n\n## sub\n\nmore"
        sections = _split_on_level(body, level=1)
        headings = [h for h, _ in sections]
        assert "heading" in headings


# NOTE: The legacy `[TOOL: ...]` regex parser was deleted in A1 (native
# Ollama tool calling). Args now arrive as real dicts via the SDK, so the
# in-app escape-decoding step is no longer reachable. Tests for
# tool_parser.parse_tool_calls / _decode_escapes were removed with the
# parser. The end-to-end save round-trip is still pinned by
# tests/integration/test_save_roundtrip.py.


# ---------------------------------------------------------------------------
# P1.4: difflib fuzzy suggestions on SECTION NOT FOUND.
# ---------------------------------------------------------------------------

class TestFuzzySectionSuggestions:
    """_fuzzy_section_suggestions ranks near-miss heading names."""

    def test_typo_returns_closest_heading(self):
        from agent.tools import KBTools
        suggestions = KBTools._fuzzy_section_suggestions(
            "marcus aureleus",
            ["marcus-aurelius", "epictetus", "seneca"],
        )
        assert suggestions[0] == "marcus-aurelius"

    def test_substring_match_preferred(self):
        from agent.tools import KBTools
        suggestions = KBTools._fuzzy_section_suggestions(
            "data flow",
            ["Architecture > Data Flow", "Architecture > Vector Pipeline"],
        )
        assert suggestions[0] == "Architecture > Data Flow"

    def test_unrelated_query_returns_empty(self):
        from agent.tools import KBTools
        suggestions = KBTools._fuzzy_section_suggestions(
            "zzzqqqxxx",
            ["overview", "details"],
        )
        assert suggestions == []

    def test_empty_inputs_return_empty(self):
        from agent.tools import KBTools
        assert KBTools._fuzzy_section_suggestions("", ["a"]) == []
        assert KBTools._fuzzy_section_suggestions("a", []) == []

    def test_limits_to_three_suggestions(self):
        from agent.tools import KBTools
        suggestions = KBTools._fuzzy_section_suggestions(
            "section",
            ["section-one", "section-two", "section-three", "section-four", "section-five"],
        )
        assert len(suggestions) <= 3

    def test_leaf_segment_matched_for_deep_paths(self):
        """Short query 'sub' should match 'Title > Long Heading > Sub'."""
        from agent.tools import KBTools
        suggestions = KBTools._fuzzy_section_suggestions(
            "sub",
            ["Title > Long Heading > Sub", "Other > Unrelated"],
        )
        assert suggestions and suggestions[0] == "Title > Long Heading > Sub"


class TestSectionNotFoundSurfacesFuzzyHint:
    """End-to-end: read_knowledge_section emits 'Did you mean: ...' on miss."""

    def test_emits_did_you_mean_for_typo(self, tmp_path):
        from agent.tools import KBTools
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (kb / "wiki").mkdir()
        (kb / "wiki" / "stoics.md").write_text(
            "# Stoics\n\nIntro.\n\n## Marcus Aurelius\n\nMeditations.\n"
            "## Epictetus\n\nDiscourses.\n"
        )
        index = _StubIndex(
            indexed_files=[
                {"filename": "wiki/stoics.md", "source": "knowledge", "tier": "wiki"},
            ],
            sections_by_file={
                ("wiki/stoics.md", "knowledge"): [
                    {"heading": "Stoics", "chunk_index": 0, "token_count": 5,
                     "summary": "", "tier": "wiki"},
                    {"heading": "Stoics > Marcus Aurelius", "chunk_index": 1,
                     "token_count": 5, "summary": "", "tier": "wiki"},
                    {"heading": "Stoics > Epictetus", "chunk_index": 2,
                     "token_count": 5, "summary": "", "tier": "wiki"},
                ],
            },
        )
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        result = tools.read_knowledge_section("wiki/stoics.md", "marcus aureleus")
        assert "SECTION NOT FOUND" in result
        assert "Did you mean" in result
        assert "Marcus Aurelius" in result

    def test_no_hint_when_no_close_match(self, tmp_path):
        from agent.tools import KBTools
        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        kb.mkdir()
        canon.mkdir()
        (kb / "wiki").mkdir()
        (kb / "wiki" / "page.md").write_text("# Title\n\nBody.\n")
        index = _StubIndex(
            indexed_files=[
                {"filename": "wiki/page.md", "source": "knowledge", "tier": "wiki"},
            ],
            sections_by_file={
                ("wiki/page.md", "knowledge"): [
                    {"heading": "Title", "chunk_index": 0, "token_count": 5,
                     "summary": "", "tier": "wiki"},
                ],
            },
        )
        tools = KBTools(kb_index=index, kb_dir=kb, canon_dir=canon)
        result = tools.read_knowledge_section("wiki/page.md", "zzzqqqxxx")
        assert "SECTION NOT FOUND" in result
        # No suggestion line when nothing crosses the cutoff
        assert "Did you mean" not in result