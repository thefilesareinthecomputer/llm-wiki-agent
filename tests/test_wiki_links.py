"""Tests for the wiki-link parser and REFERENCES edge builder (P2.1).

Three layers under test:
  1. parse_links     — pure regex layer
  2. resolve_link    — index-aware target resolution
  3. _build_wiki_link_edges — end-to-end: indexed chunks → REFERENCES edges
                              with link_text + target_anchor provenance
"""

from pathlib import Path

import pytest

from knowledge.wiki_links import (
    parse_links,
    resolve_link,
    normalize_anchor,
    _normalize_target,
)


# ---------------------------------------------------------------------------
# Layer 1: parse_links
# ---------------------------------------------------------------------------

class TestParseWikiLinks:
    """Obsidian-style [[link]] forms."""

    def test_bare_wiki_link(self):
        links = parse_links("See [[cortisol]] for more.")
        assert len(links) == 1
        assert links[0]["target"] == "cortisol.md"
        assert links[0]["anchor"] == ""
        assert links[0]["display"] == "cortisol"
        assert links[0]["kind"] == "wiki"

    def test_wiki_link_with_display(self):
        links = parse_links("See [[cortisol|stress hormone]] for more.")
        assert len(links) == 1
        assert links[0]["target"] == "cortisol.md"
        assert links[0]["display"] == "stress hormone"

    def test_wiki_link_with_anchor(self):
        links = parse_links("See [[cortisol#regulation]] above.")
        assert len(links) == 1
        assert links[0]["target"] == "cortisol.md"
        assert links[0]["anchor"] == "regulation"

    def test_wiki_link_with_anchor_and_display(self):
        links = parse_links("See [[cortisol#regulation|regulation]] above.")
        assert len(links) == 1
        assert links[0]["target"] == "cortisol.md"
        assert links[0]["anchor"] == "regulation"
        assert links[0]["display"] == "regulation"

    def test_wiki_link_already_has_md_extension(self):
        links = parse_links("See [[cortisol.md]] for more.")
        assert links[0]["target"] == "cortisol.md"

    def test_image_embed_skipped(self):
        """![[file.png]] is an Obsidian image embed — not a reference."""
        links = parse_links("Embed: ![[diagram.png]] inline.")
        assert links == []

    def test_multiple_wiki_links_ordered(self):
        body = "Refer to [[A]] and then [[B]] then [[C]]."
        links = parse_links(body)
        assert [l["target"] for l in links] == ["A.md", "B.md", "C.md"]


class TestParseMarkdownLinks:
    """Standard markdown links to local .md files."""

    def test_basic_markdown_link(self):
        links = parse_links("See [cortisol notes](cortisol.md) above.")
        assert len(links) == 1
        assert links[0]["target"] == "cortisol.md"
        assert links[0]["display"] == "cortisol notes"
        assert links[0]["kind"] == "markdown"

    def test_markdown_link_with_anchor(self):
        links = parse_links("See [regulation](cortisol.md#regulation).")
        assert links[0]["target"] == "cortisol.md"
        assert links[0]["anchor"] == "regulation"

    def test_markdown_link_with_path(self):
        links = parse_links("See [doc](wiki/cortisol.md).")
        assert links[0]["target"] == "wiki/cortisol.md"

    def test_markdown_link_with_relative_dot(self):
        links = parse_links("See [doc](./cortisol.md).")
        assert links[0]["target"] == "cortisol.md"

    def test_external_http_link_skipped(self):
        links = parse_links("See [Google](https://google.com).")
        assert links == []

    def test_mailto_link_skipped(self):
        links = parse_links("[Email](mailto:foo@bar.com)")
        assert links == []

    def test_image_link_skipped(self):
        """![alt](file.png) is an image, not a reference."""
        links = parse_links("Image: ![alt](pic.png) here.")
        assert links == []

    def test_anchor_only_link_skipped(self):
        """[heading](#section) is an in-page anchor — no cross-file edge."""
        links = parse_links("See [above](#overview).")
        assert links == []


class TestParseLinksIgnoresCode:
    """Links inside code blocks/spans must not pollute the graph."""

    def test_inline_code_ignored(self):
        body = "Use the syntax `[[wiki-link]]` to link, but [[real-page]] is real."
        links = parse_links(body)
        assert len(links) == 1
        assert links[0]["target"] == "real-page.md"

    def test_fenced_code_ignored(self):
        body = (
            "Real link: [[outside]]\n\n"
            "```\n[[inside-code]]\n```\n\n"
            "Another real: [[also-outside]]"
        )
        links = parse_links(body)
        targets = [l["target"] for l in links]
        assert "outside.md" in targets
        assert "also-outside.md" in targets
        assert "inside-code.md" not in targets


class TestParseLinksMixed:
    def test_wiki_and_markdown_in_same_body(self):
        body = (
            "First [[wiki-page]] and then "
            "[markdown link](other.md#sec) followed by "
            "[[third|with display]]."
        )
        links = parse_links(body)
        assert [l["target"] for l in links] == [
            "wiki-page.md", "other.md", "third.md",
        ]
        assert [l["kind"] for l in links] == [
            "wiki", "markdown", "wiki",
        ]

    def test_empty_body(self):
        assert parse_links("") == []
        assert parse_links(None) == []  # type: ignore[arg-type]

    def test_no_links(self):
        assert parse_links("Just plain text with no links at all.") == []


class TestNormalizeTarget:
    def test_appends_md_extension(self):
        assert _normalize_target("cortisol") == "cortisol.md"

    def test_keeps_existing_md(self):
        assert _normalize_target("cortisol.md") == "cortisol.md"

    def test_strips_relative_dot_slash(self):
        assert _normalize_target("./cortisol.md") == "cortisol.md"

    def test_strips_leading_slash(self):
        assert _normalize_target("/cortisol.md") == "cortisol.md"

    def test_external_returns_empty(self):
        assert _normalize_target("https://x.com") == ""
        assert _normalize_target("mailto:x@y.com") == ""

    def test_anchor_only_returns_empty(self):
        assert _normalize_target("#section") == ""

    def test_blank_returns_empty(self):
        assert _normalize_target("") == ""
        assert _normalize_target("   ") == ""


class TestNormalizeAnchor:
    def test_lowercase(self):
        assert normalize_anchor("Marcus Aurelius") == "marcus-aurelius"

    def test_collapses_whitespace_and_underscores(self):
        assert normalize_anchor("marcus  aurelius_meditations") == "marcus-aurelius-meditations"

    def test_strips_outer_dashes(self):
        assert normalize_anchor("--marcus--") == "marcus"

    def test_empty(self):
        assert normalize_anchor("") == ""


# ---------------------------------------------------------------------------
# Layer 2: resolve_link
# ---------------------------------------------------------------------------

class TestResolveLink:
    """Index-aware target resolution."""

    def test_exact_path_match(self):
        link = {"target": "wiki/cortisol.md"}
        files = [
            {"filename": "wiki/cortisol.md", "source": "knowledge"},
            {"filename": "wiki/dopamine.md", "source": "knowledge"},
        ]
        assert resolve_link(link, files) == ("wiki/cortisol.md", "knowledge")

    def test_basename_unique_match(self):
        link = {"target": "cortisol.md"}
        files = [
            {"filename": "wiki/cortisol.md", "source": "knowledge"},
            {"filename": "wiki/dopamine.md", "source": "knowledge"},
        ]
        assert resolve_link(link, files) == ("wiki/cortisol.md", "knowledge")

    def test_basename_ambiguous_returns_none(self):
        """Two files with the same basename — refuse to silently pick one."""
        link = {"target": "page.md"}
        files = [
            {"filename": "wiki/page.md", "source": "knowledge"},
            {"filename": "raw/page.md", "source": "knowledge"},
        ]
        assert resolve_link(link, files) is None

    def test_path_suffix_match(self):
        link = {"target": "cortisol.md"}
        files = [
            {"filename": "wiki/health/cortisol.md", "source": "knowledge"},
        ]
        assert resolve_link(link, files) == ("wiki/health/cortisol.md", "knowledge")

    def test_canon_target_returns_canon_source(self):
        link = {"target": "stoicism.md"}
        files = [
            {"filename": "stoicism.md", "source": "canon"},
        ]
        assert resolve_link(link, files) == ("stoicism.md", "canon")

    def test_no_match_returns_none(self):
        link = {"target": "nonexistent.md"}
        files = [{"filename": "wiki/cortisol.md", "source": "knowledge"}]
        assert resolve_link(link, files) is None

    def test_empty_inputs_return_none(self):
        assert resolve_link({"target": ""}, [{"filename": "a.md", "source": "x"}]) is None
        assert resolve_link({"target": "a.md"}, []) is None


# ---------------------------------------------------------------------------
# Layer 3: end-to-end edge builder
# ---------------------------------------------------------------------------

@pytest.fixture
def index_with_links(tmp_path, monkeypatch):
    """Build a real KBIndex over a tiny KB containing wiki links.

    Uses the same construction pattern as conftest.client_with_init:
    construct KBIndex(), then swap _embedding_fn for a fake to avoid
    any network calls.
    """
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
        "# cortisol\n\nStress hormone.\n\n## regulation\n\nHPA axis governs it.\n"
    )
    (kb / "wiki" / "stress.md").write_text(
        "# stress\n\nSee [[cortisol]] for the hormone, "
        "and [[cortisol#regulation|HPA axis]] for the loop.\n\n"
        "Also [the dopamine page](dopamine.md#reward) mentions it.\n"
    )
    (kb / "wiki" / "dopamine.md").write_text(
        "# dopamine\n\nReward neurotransmitter.\n\n## reward\n\nVTA signaling.\n"
    )

    index = KBIndex()
    index.db = lancedb.connect(str(tmp_path / "lancedb"))
    index._embedding_fn = FakeEmbeddingFunction()
    index.build_index(extract_entities=False, llm_summaries=False, force=True)
    return index


class TestWikiLinkEdgesEndToEnd:
    """_build_wiki_link_edges emits REFERENCES edges with provenance."""

    def test_wiki_link_creates_reference_edge(self, index_with_links):
        from knowledge.graph import EdgeType, NodeType
        graph = index_with_links.graph
        ref_edges = [
            e for e in graph.edges.values()
            if e.edge_type == EdgeType.REFERENCES
        ]
        assert ref_edges, "expected at least one REFERENCES edge from wiki links"

        # The stress page links to cortisol — find that edge and check provenance
        chunk_lookup = {nid: n for nid, n in graph.nodes.items()
                        if n.node_type == NodeType.CHUNK}

        def _is_from(edge, filename):
            n = chunk_lookup.get(edge.source_id)
            return n is not None and n.filename == filename

        def _is_to(edge, filename):
            n = chunk_lookup.get(edge.target_id)
            return n is not None and n.filename == filename

        stress_to_cortisol = [
            e for e in ref_edges
            if _is_from(e, "wiki/stress.md") and _is_to(e, "wiki/cortisol.md")
        ]
        assert stress_to_cortisol, "expected wiki/stress.md → wiki/cortisol.md edge"

    def test_edge_carries_link_text_provenance(self, index_with_links):
        from knowledge.graph import EdgeType, NodeType
        graph = index_with_links.graph
        chunk_lookup = {nid: n for nid, n in graph.nodes.items()
                        if n.node_type == NodeType.CHUNK}

        for e in graph.edges.values():
            if e.edge_type != EdgeType.REFERENCES:
                continue
            src = chunk_lookup.get(e.source_id)
            tgt = chunk_lookup.get(e.target_id)
            if not src or not tgt:
                continue
            if src.filename == "wiki/stress.md" and tgt.filename == "wiki/cortisol.md":
                assert "link_text" in e.attributes
                assert "target_file" in e.attributes
                assert e.attributes["target_file"] == "wiki/cortisol.md"
                assert e.attributes.get("link_kind") in ("wiki", "markdown")
                return
        pytest.fail("never found a stress→cortisol REFERENCES edge to inspect")

    def test_anchor_preserved_in_edge_attributes(self, tmp_path, monkeypatch):
        """`[[cortisol#regulation]]` must record `target_anchor=regulation`
        on the edge even when the target file is small enough that the
        chunker collapses sections. The anchor is the agent's stated intent
        and lint passes (P3.2) need it to flag heading drift."""
        import lancedb
        import knowledge.index as kbi
        from knowledge.index import KBIndex
        from knowledge.graph import EdgeType
        from tests.conftest import FakeEmbeddingFunction

        kb = tmp_path / "knowledge"
        canon = tmp_path / "canon"
        (kb / "wiki").mkdir(parents=True)
        canon.mkdir()
        monkeypatch.setattr(kbi, "KB_DIR", kb)
        monkeypatch.setattr(kbi, "CANON_DIR", canon)
        monkeypatch.setattr(kbi, "LANCEDB_DIR", tmp_path / "lancedb")

        (kb / "wiki" / "cortisol.md").write_text(
            "# cortisol\n\nStress hormone.\n\n## regulation\n\nHPA axis.\n"
        )
        (kb / "wiki" / "stress.md").write_text(
            "# stress\n\nOnly anchored: [[cortisol#regulation|HPA axis]].\n"
        )

        index = KBIndex()
        index.db = lancedb.connect(str(tmp_path / "lancedb"))
        index._embedding_fn = FakeEmbeddingFunction()
        index.build_index(extract_entities=False, llm_summaries=False, force=True)

        anchored = [
            e for e in index.graph.edges.values()
            if e.edge_type == EdgeType.REFERENCES
            and e.attributes.get("target_anchor") == "regulation"
        ]
        assert anchored, (
            "expected REFERENCES edge with target_anchor='regulation' "
            "preserved in attributes regardless of chunk granularity"
        )
        for e in anchored:
            assert e.attributes.get("link_text") == "HPA axis"
            assert e.attributes.get("target_file") == "wiki/cortisol.md"

    def test_markdown_link_creates_reference_edge(self, index_with_links):
        """[the dopamine page](dopamine.md#reward) — markdown form."""
        from knowledge.graph import EdgeType, NodeType
        graph = index_with_links.graph
        chunk_lookup = {nid: n for nid, n in graph.nodes.items()
                        if n.node_type == NodeType.CHUNK}

        markdown_edges = [
            e for e in graph.edges.values()
            if e.edge_type == EdgeType.REFERENCES
            and e.attributes.get("link_kind") == "markdown"
        ]
        assert markdown_edges, "expected at least one markdown-form REFERENCES edge"

    def test_no_self_reference_edges(self, index_with_links):
        """A chunk that links to itself shouldn't get a self-edge."""
        from knowledge.graph import EdgeType
        graph = index_with_links.graph
        for e in graph.edges.values():
            if e.edge_type == EdgeType.REFERENCES:
                assert e.source_id != e.target_id

    def test_unresolvable_link_emits_no_edge(self, tmp_path, monkeypatch):
        """[[ghost-page]] when ghost-page.md doesn't exist → no edge."""
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

        (kb / "wiki" / "alone.md").write_text(
            "# alone\n\nReferences [[ghost-page]] which doesn't exist.\n"
        )

        index = KBIndex()
        index.db = lancedb.connect(str(tmp_path / "lancedb"))
        index._embedding_fn = FakeEmbeddingFunction()
        index.build_index(extract_entities=False, llm_summaries=False, force=True)

        from knowledge.graph import EdgeType
        ref_edges = [
            e for e in index.graph.edges.values()
            if e.edge_type == EdgeType.REFERENCES
        ]
        assert ref_edges == []


# ---------------------------------------------------------------------------
# Edge.attributes round-trip (P2.1 added the field)
# ---------------------------------------------------------------------------

class TestEdgeAttributesRoundTrip:
    def test_edge_attributes_default_empty(self):
        from knowledge.graph import Edge, EdgeType
        e = Edge(source_id="a", target_id="b", edge_type=EdgeType.SIMILAR)
        assert e.attributes == {}

    def test_edge_attributes_serialize(self):
        from knowledge.graph import Edge, EdgeType
        e = Edge(
            source_id="a", target_id="b", edge_type=EdgeType.REFERENCES,
            weight=1.0, evidence="wiki", attributes={"link_text": "see X"},
        )
        d = e.to_dict()
        assert d["attributes"] == {"link_text": "see X"}

    def test_edge_attributes_deserialize_with_field(self):
        from knowledge.graph import Edge
        d = {
            "source_id": "a", "target_id": "b", "edge_type": "references",
            "weight": 1.0, "evidence": "x",
            "attributes": {"link_text": "see X"},
        }
        e = Edge.from_dict(d)
        assert e.attributes == {"link_text": "see X"}

    def test_edge_attributes_deserialize_legacy_without_field(self):
        """Legacy persisted edges (pre-P2.1) had no `attributes` key —
        must still deserialize, with attributes defaulting to {}."""
        from knowledge.graph import Edge
        d = {
            "source_id": "a", "target_id": "b", "edge_type": "similar",
            "weight": 0.8, "evidence": "x",
        }
        e = Edge.from_dict(d)
        assert e.attributes == {}
