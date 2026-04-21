"""Tests for knowledge base chunker."""

import pytest
from knowledge.chunker import (
    HeadingNode,
    _build_context_header,
    _doc_summary,
    build_heading_tree,
    chunk_file,
    format_heading_tree,
    enrich_tree_summaries,
    _detect_heading_level,
    _split_on_level,
    _parse_frontmatter,
)


class TestParseFrontmatter:
    """Test YAML frontmatter extraction."""

    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("Hello world")
        assert meta == {}
        assert body == "Hello world"

    def test_simple_frontmatter(self):
        text = "---\ntitle: Test\ntags: ai, ml\n---\nBody content"
        meta, body = _parse_frontmatter(text)
        assert meta["title"] == "Test"
        assert "Body content" in body

    def test_empty_frontmatter(self):
        text = "---\n---\nJust body"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert "Just body" in body


class TestDetectHeadingLevel:
    """Test heading level detection."""

    def test_h1(self):
        assert _detect_heading_level("# Title\nBody") == 1

    def test_h2(self):
        assert _detect_heading_level("## Section\nBody") == 2

    def test_h3(self):
        assert _detect_heading_level("### Sub\nBody") == 3

    def test_mixed_levels(self):
        # Should return the highest (lowest number) level
        text = "## Section\n### Sub\nBody"
        assert _detect_heading_level(text) == 2

    def test_no_headings(self):
        assert _detect_heading_level("Just plain text") is None


class TestSplitOnLevel:
    """Test heading-level splitting."""

    def test_h1_sections(self):
        text = "# First\nContent 1\n# Second\nContent 2"
        sections = _split_on_level(text, 1)
        assert len(sections) == 2
        assert sections[0][0] == "First"
        assert sections[1][0] == "Second"

    def test_preamble(self):
        text = "Preamble text\n# First\nContent"
        sections = _split_on_level(text, 1)
        assert sections[0][0] == "(preamble)"
        assert "Preamble" in sections[0][1]

    def test_h2_sections(self):
        text = "## Alpha\nA content\n## Beta\nB content"
        sections = _split_on_level(text, 2)
        assert len(sections) == 2
        assert sections[0][0] == "Alpha"

    def test_strips_trailing_dividers(self):
        text = "# Section\nContent\n---\n# Next\nMore"
        sections = _split_on_level(text, 1)
        assert not sections[0][1].rstrip().endswith("---")


class TestBuildHeadingTree:
    """Test heading tree construction."""

    def test_empty_file(self):
        tree = build_heading_tree("", "test.md")
        assert tree.level == 0
        assert tree.heading == "test"
        assert tree.subtree_tokens == 0

    def test_no_headings(self):
        tree = build_heading_tree("Just some text here.", "doc.md")
        assert tree.heading == "doc"
        assert tree.own_tokens > 0
        assert tree.subtree_tokens == tree.own_tokens
        assert len(tree.children) == 0

    def test_h1_headings(self):
        text = "# Title\nSome content\n## Subtitle\nMore content"
        tree = build_heading_tree(text, "test.md")
        assert tree.heading == "test"
        assert len(tree.children) == 1
        assert tree.children[0].heading == "Title"
        assert tree.children[0].level == 1

    def test_nested_headings(self):
        text = "# Parent\n## Child\n### Grandchild\nContent"
        tree = build_heading_tree(text, "nested.md")
        parent = tree.children[0]
        assert parent.heading == "Parent"
        assert len(parent.children) == 1
        assert parent.children[0].heading == "Child"

    def test_subtree_tokens(self):
        text = "# Title\nSome content here\n## Section\nMore content"
        tree = build_heading_tree(text, "test.md")
        assert tree.subtree_tokens > 0
        # subtree_tokens should be >= own_tokens for any node with children
        for child in tree.children:
            assert child.subtree_tokens >= child.own_tokens

    def test_with_frontmatter(self):
        text = "---\ntitle: Test\n---\n# Heading\nContent"
        tree = build_heading_tree(text, "test.md")
        assert tree.heading == "test"
        assert len(tree.children) == 1


class TestFormatHeadingTree:
    """Test heading tree rendering."""

    def test_empty_tree(self):
        tree = HeadingNode(level=0, heading="empty")
        result = format_heading_tree(tree)
        assert "empty" in result
        assert "0 tokens" in result

    def test_tree_with_children(self):
        tree = HeadingNode(level=0, heading="doc", subtree_tokens=500)
        child = HeadingNode(level=1, heading="Section", own_tokens=500, subtree_tokens=500)
        tree.children.append(child)
        result = format_heading_tree(tree)
        assert "doc" in result
        assert "Section" in result
        assert "# Section" in result

    def test_tree_with_summary(self):
        tree = HeadingNode(level=0, heading="doc", subtree_tokens=300, summary="A summary")
        result = format_heading_tree(tree)
        assert "A summary" in result


class TestEnrichTreeSummaries:
    """Test summary attachment to tree nodes."""

    def test_attach_summary(self):
        tree = HeadingNode(level=0, heading="doc")
        child = HeadingNode(level=1, heading="Architecture")
        tree.children.append(child)

        summaries = {"architecture": "Covers system design"}
        enrich_tree_summaries(tree, summaries)
        assert child.summary == "Covers system design"

    def test_case_insensitive(self):
        tree = HeadingNode(level=0, heading="doc")
        child = HeadingNode(level=1, heading="Architecture")
        tree.children.append(child)

        summaries = {"ARCHITECTURE": "Covers system design"}
        enrich_tree_summaries(tree, summaries)
        assert child.summary == "Covers system design"


class TestChunkFile:
    """Test file chunking."""

    def test_empty_file(self):
        result = chunk_file("", "empty.md")
        assert result == []

    def test_no_headings(self):
        text = "Just some plain text without any headings."
        result = chunk_file(text, "plain.md")
        assert len(result) == 1
        assert result[0]["heading"] == "plain"
        assert result[0]["token_count"] > 0

    def test_h1_sections(self):
        text = "# First\nContent one\n# Second\nContent two"
        result = chunk_file(text, "test.md")
        assert len(result) == 2
        assert result[0]["heading"] == "First"
        assert result[1]["heading"] == "Second"
        assert all("chunk_index" in c for c in result)

    def test_h2_sections(self):
        text = "## Alpha\nAlpha content\n## Beta\nBeta content"
        result = chunk_file(text, "sub.md")
        assert len(result) == 2
        assert result[0]["heading"] == "Alpha"

    def test_with_frontmatter(self):
        text = "---\ntitle: Test\n---\n# Section\nContent"
        result = chunk_file(text, "fm.md")
        assert len(result) >= 1
        assert result[0]["heading"] == "Section"

    def test_max_tokens_splitting(self):
        # Create text that exceeds a small token budget
        text = "# Big Section\n" + "Word " * 200
        result = chunk_file(text, "big.md", max_tokens=100)
        # Should be split into multiple chunks
        assert len(result) > 1

    def test_chunk_indices_sequential(self):
        text = "# A\nContent\n# B\nMore\n# C\nFinal"
        result = chunk_file(text, "seq.md")
        indices = [c["chunk_index"] for c in result]
        assert indices == list(range(len(result)))

    def test_nested_headings_in_h1_chunk(self):
        text = "# Parent\nParent intro\n## Child\nChild content\n## Another\nMore"
        result = chunk_file(text, "nested.md")
        # H2 sections should stay inside the H1 chunk
        assert len(result) == 1
        assert result[0]["heading"] == "Parent"
        assert "Child content" in result[0]["content"]

    def test_context_header_fields_present(self):
        text = "# First\nContent one\n# Second\nContent two"
        result = chunk_file(text, "test.md")
        assert len(result) == 2
        for c in result:
            assert "filename" in c
            assert c["filename"] == "test.md"
            assert "total_chunks" in c
            assert c["total_chunks"] == 2
            assert "doc_summary" in c

    def test_context_header_no_headings(self):
        text = "This is the first line.\nMore content here."
        result = chunk_file(text, "plain.md")
        assert len(result) == 1
        assert result[0]["filename"] == "plain.md"
        assert result[0]["total_chunks"] == 1
        assert result[0]["doc_summary"] == "This is the first line."

    def test_context_header_position_correct(self):
        text = "# A\nContent\n# B\nMore\n# C\nFinal"
        result = chunk_file(text, "seq.md")
        assert len(result) == 3
        for i, c in enumerate(result):
            assert c["chunk_index"] == i
            assert c["total_chunks"] == 3


class TestDocSummary:
    """Test doc_summary extraction."""

    def test_first_meaningful_line(self):
        assert _doc_summary("Hello world\nMore text") == "Hello world"

    def test_skip_headings(self):
        assert _doc_summary("# Title\nReal content") == "Real content"

    def test_skip_frontmatter_divider(self):
        assert _doc_summary("---\nActual line") == "Actual line"

    def test_skip_toc_links(self):
        assert _doc_summary("[Intro](#intro)\nReal text") == "Real text"

    def test_skip_table_rows(self):
        assert _doc_summary("| Col1 | Col2 |\nReal content") == "Real content"

    def test_truncation(self):
        long_line = "x" * 200
        result = _doc_summary(long_line)
        assert len(result) == 153  # 150 + "..."
        assert result.endswith("...")

    def test_empty_file(self):
        assert _doc_summary("") == "(no summary)"


class TestBuildContextHeader:
    """Test context header generation."""

    def test_basic_header(self):
        header = _build_context_header(
            filename="ai/guide.md",
            position=3,
            total=10,
            heading="Architecture",
            doc_summary="Covers system design",
        )
        assert "file: ai/guide.md" in header
        assert "position: 3/10" in header
        assert "doc_summary: Covers system design" in header
        # No ancestors for top-level heading
        assert "ancestors" not in header

    def test_ancestors_from_heading_path(self):
        header = _build_context_header(
            filename="ai/guide.md",
            position=5,
            total=10,
            heading="Architecture > Frontend > Components",
            doc_summary="System overview",
        )
        assert "ancestors: Architecture > Frontend" in header
        assert "Components" not in header.split("ancestors:")[1].split("\n")[0]

    def test_two_level_ancestors(self):
        header = _build_context_header(
            filename="doc.md",
            position=1,
            total=5,
            heading="Intro > Details",
            doc_summary="Overview",
        )
        assert "ancestors: Intro" in header

    def test_no_doc_summary_when_empty(self):
        header = _build_context_header(
            filename="doc.md",
            position=1,
            total=5,
            heading="Section",
            doc_summary="(no summary)",
        )
        assert "doc_summary" not in header

    def test_yaml_delimiters(self):
        header = _build_context_header(
            filename="f.md", position=1, total=1,
            heading="X", doc_summary="Y",
        )
        lines = header.strip().split("\n")
        assert lines[0] == "---"
        assert lines[-1] == "---"