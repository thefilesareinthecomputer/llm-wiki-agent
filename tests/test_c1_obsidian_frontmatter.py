"""C1 — Obsidian-vault-valid frontmatter, wiki-link conversion,
deterministic Related footer.

Covers:

1. ``_build_frontmatter`` emits ``aliases``, ``tags`` (YAML block lists),
   ``created``, ``updated`` (date-only), ``source``, ``tier`` (Obsidian's
   expected key set). Legacy ``date-created`` / ``last-modified`` are read
   when upgrading old files but are **not** written back.
2. ``created`` is preserved across edits when the existing file already
   has it (Obsidian convention: created is set once).
3. ``_convert_markdown_links_to_wiki`` rewrites in-vault markdown links to
   ``[[slug]]`` / ``[[slug|Display]]`` while leaving canon, external,
   anchor, image, and code-fenced/code-span links untouched.
4. ``save_knowledge`` writes a memory-tier page with ``tier: memory`` in
   the frontmatter when the filename lands under ``memory/``.
5. ``_compute_related_block`` produces a stable ordering for tied weights
   so the auto-generated Related footer is deterministic.
6. End-to-end snapshot: a saved page is parseable by PyYAML and has all
   the keys an Obsidian vault expects.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ----- _build_frontmatter ---------------------------------------------------

def test_frontmatter_emits_obsidian_keys():
    from agent.tools import _build_frontmatter
    import yaml

    fm = _build_frontmatter(
        tags="alpha,beta",
        date_created="2026-04-01",
        aliases=["First Alias"],
        source="knowledge",
        tier="wiki",
    )
    body = fm.split("---")[1].strip()
    parsed = yaml.safe_load(body)
    assert parsed["aliases"] == ["First Alias"]
    assert parsed["tags"] == ["alpha", "beta"]
    assert parsed["source"] == "knowledge"
    assert parsed["tier"] == "wiki"
    assert "created" in parsed
    assert "updated" in parsed
    assert "date-created" not in parsed
    assert "last-modified" not in parsed
    assert "T" not in str(parsed["updated"])  # date-only, not ISO datetime


def test_frontmatter_preserves_existing_created():
    from agent.tools import _build_frontmatter
    import yaml

    existing = {
        "created": "2024-01-15",
        "tags": ["legacy"],
        "aliases": ["Old Name"],
        "source": "knowledge",
        "tier": "wiki",
    }
    fm = _build_frontmatter(
        tags="new",
        date_created="2026-04-18",
        existing_meta=existing,
    )
    parsed = yaml.safe_load(fm.split("---")[1].strip())
    # Created stays at the original value, not the new date_created arg.
    # PyYAML auto-parses ISO dates to datetime.date, which is correct YAML
    # and correct Obsidian-frontmatter behavior; compare via str().
    assert str(parsed["created"]) == "2024-01-15"
    # Aliases merge — old + nothing new
    assert "Old Name" in parsed["aliases"]
    # Tags merge — both old and new appear, deduped
    assert "new" in parsed["tags"]
    assert "legacy" in parsed["tags"]


def test_frontmatter_carries_through_legacy_date_created():
    """A file written before C1 only has `date-created`. The next save must
    promote it to the new `created` key without losing the original date."""
    from agent.tools import _build_frontmatter
    import yaml

    existing = {"date-created": "2025-06-01"}
    fm = _build_frontmatter(
        tags="x",
        date_created="2026-04-18",
        existing_meta=existing,
    )
    parsed = yaml.safe_load(fm.split("---")[1].strip())
    assert str(parsed["created"]) == "2025-06-01"


# ----- markdown -> wiki link conversion -------------------------------------

@pytest.mark.parametrize("src,expected", [
    ("See [Cortisol](wiki/cortisol.md).", "See [[cortisol|Cortisol]]."),
    (
        "[cortisol](wiki/cortisol.md)",
        "[[cortisol]]",
    ),
    (
        "[Section](knowledge/wiki/foo.md#intro)",
        "[[foo#intro|Section]]",
    ),
    (
        "[memory note](memory/yesterday.md)",
        "[[yesterday|memory note]]",
    ),
    # canon links must stay markdown so citation semantics hold
    ("[Source](canon/mind/quotes.md)", "[Source](canon/mind/quotes.md)"),
    # external link
    ("[Google](https://example.com)", "[Google](https://example.com)"),
    # image embed
    ("![alt](wiki/diagram.png)", "![alt](wiki/diagram.png)"),
    # mailto + anchor
    ("[Mail](mailto:a@b.c)", "[Mail](mailto:a@b.c)"),
    ("[Top](#top)", "[Top](#top)"),
    # already a wiki-link — left alone
    ("[[already]]", "[[already]]"),
])
def test_convert_markdown_links_to_wiki(src, expected):
    from agent.tools import _convert_markdown_links_to_wiki
    assert _convert_markdown_links_to_wiki(src) == expected


def test_convert_skips_code_blocks():
    from agent.tools import _convert_markdown_links_to_wiki
    src = (
        "Outside [Foo](wiki/foo.md) link.\n"
        "```python\n"
        "x = '[Bar](wiki/bar.md)'\n"
        "```\n"
        "After [Baz](wiki/baz.md)."
    )
    out = _convert_markdown_links_to_wiki(src)
    assert "[[foo|Foo]]" in out
    assert "[[baz|Baz]]" in out
    # Inside the fence stays raw
    assert "[Bar](wiki/bar.md)" in out
    assert "[[bar" not in out


def test_convert_skips_inline_code():
    from agent.tools import _convert_markdown_links_to_wiki
    src = "Use `[Sample](wiki/sample.md)` then [Real](wiki/real.md)."
    out = _convert_markdown_links_to_wiki(src)
    # Inline code preserved literally
    assert "`[Sample](wiki/sample.md)`" in out
    # Real link converted
    assert "[[real|Real]]" in out


# ----- save_knowledge end-to-end snapshot -----------------------------------

def _kb_dirs(tmp_path):
    kb = tmp_path / "knowledge"
    canon = tmp_path / "canon"
    (kb / "wiki").mkdir(parents=True)
    (kb / "memory").mkdir(parents=True)
    canon.mkdir()
    return kb, canon


def test_save_knowledge_writes_obsidian_valid_frontmatter(tmp_path):
    """The on-disk file must parse cleanly with PyYAML and carry every
    Obsidian-relevant frontmatter key."""
    import yaml
    from agent.tools import KBTools

    kb, canon = _kb_dirs(tmp_path)
    tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
    result = tools.save_knowledge(
        "obsidian-test.md",
        "## Body\n\nSome content with [Cortisol](wiki/cortisol.md).",
        tags="hormones, demo",
    )
    assert result.startswith("Saved")
    body = (kb / "wiki" / "obsidian-test.md").read_text()

    # Frontmatter parses cleanly
    parts = body.split("---", 2)
    assert len(parts) >= 3
    parsed = yaml.safe_load(parts[1])

    for key in ("tags", "created", "updated", "source", "tier"):
        assert key in parsed, f"missing required frontmatter key: {key}"
    assert parsed["tier"] == "wiki"
    assert parsed["source"] == "knowledge"
    assert "hormones" in parsed["tags"]
    assert "demo" in parsed["tags"]
    assert "date-created" not in parsed
    assert "last-modified" not in parsed
    assert "T" not in str(parsed["updated"])

    # Wiki-link conversion applied
    assert "[[cortisol|Cortisol]]" in body
    assert "[Cortisol](wiki/cortisol.md)" not in body


def test_save_knowledge_memory_tier_routing(tmp_path):
    """Pages saved under memory/ must carry tier: memory in frontmatter."""
    import yaml
    from agent.tools import KBTools

    kb, canon = _kb_dirs(tmp_path)
    tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
    tools.save_knowledge(
        "memory/today.md",
        "## Notes\n\nCaptured a thought.",
        tags="journal",
    )
    body = (kb / "memory" / "today.md").read_text()
    parsed = yaml.safe_load(body.split("---", 2)[1])
    assert parsed["tier"] == "memory"


def test_save_knowledge_preserves_created_on_edit(tmp_path):
    import yaml
    from agent.tools import KBTools

    kb, canon = _kb_dirs(tmp_path)
    tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)

    # First save sets `created`.
    tools.save_knowledge("draft.md", "## v1\n\nFirst.", tags="x")
    first = (kb / "wiki" / "draft.md").read_text()
    first_meta = yaml.safe_load(first.split("---", 2)[1])
    created_v1 = first_meta["created"]

    # Second save must keep the same `created` even though the body changed.
    tools.save_knowledge("draft.md", "## v2\n\nSecond.", tags="x")
    second = (kb / "wiki" / "draft.md").read_text()
    second_meta = yaml.safe_load(second.split("---", 2)[1])
    assert second_meta["created"] == created_v1


def test_save_knowledge_canon_links_left_as_markdown(tmp_path):
    """Canon citations stay as plain markdown — they're not wiki-linkable
    inside the writable vault."""
    from agent.tools import KBTools

    kb, canon = _kb_dirs(tmp_path)
    tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
    tools.save_knowledge(
        "cite.md",
        "## Cite\n\nSee [Quotes](canon/mind/quotes.md) for context.",
    )
    body = (kb / "wiki" / "cite.md").read_text()
    assert "[Quotes](canon/mind/quotes.md)" in body
    assert "[[quotes" not in body


# ----- deterministic Related block ------------------------------------------

def test_related_block_ordering_is_stable_for_tied_weights(tmp_path):
    """When two graph neighbors share the same total weight, the Related
    footer must order them deterministically by filename so re-saving the
    same page never thrashes the file on disk."""
    from knowledge.graph import (
        Edge,
        EdgeType,
        KnowledgeGraph,
        Node,
        NodeType,
    )
    from agent.tools import KBTools

    graph = KnowledgeGraph(persist_path=tmp_path / "graph.json")
    own = Node(
        id="own", node_type=NodeType.CHUNK,
        name="own", filename="wiki/own.md", heading="own",
        attributes={"source": "knowledge"},
    )
    graph.add_node(own)
    for fname in ("wiki/zebra.md", "wiki/apple.md", "wiki/mango.md"):
        nid = fname.replace("/", "_")
        graph.add_node(Node(
            id=nid, node_type=NodeType.CHUNK,
            name=fname, filename=fname, heading=fname,
            attributes={"source": "knowledge"},
        ))
        graph.add_edge(Edge(
            source_id="own", target_id=nid,
            edge_type=EdgeType.SIMILAR,
            weight=0.5,  # identical weights so only the tiebreak matters
        ))

    class _StubKBIndex:
        def __init__(self, g):
            self.graph = g

    tools = KBTools(
        kb_index=_StubKBIndex(graph),
        kb_dir=tmp_path / "knowledge",
        canon_dir=tmp_path / "canon",
    )

    block = tools._compute_related_block("wiki/own.md", top_n=3)
    # Filenames sort alphabetically -> apple, mango, zebra
    apple_pos = block.find("apple")
    mango_pos = block.find("mango")
    zebra_pos = block.find("zebra")
    assert apple_pos != -1 and mango_pos != -1 and zebra_pos != -1
    assert apple_pos < mango_pos < zebra_pos


# ----- _build_file_content has new signature defaults -----------------------

def test_build_file_content_default_kwargs_are_back_compat():
    """Existing callers that pass only positional (filename, content, tags)
    must keep working."""
    from agent.tools import _build_file_content

    out = _build_file_content("test.md", "## Body\n\nfoo")
    assert out.startswith("---")
    assert "tier: wiki" in out
    assert "source: knowledge" in out


def test_normalize_tags_flattens_nested_lists():
    from agent.tools import _normalize_tags

    assert _normalize_tags([["philosophy", "neuroscience"], "stoicism"]) == (
        "philosophy,neuroscience,stoicism"
    )


def test_save_knowledge_flattens_nested_tag_argument(tmp_path):
    import yaml
    from agent.tools import KBTools

    kb, canon = _kb_dirs(tmp_path)
    tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
    out = tools.save_knowledge(
        "nested-tags.md",
        "## Body\n\nx.",
        tags=[["philosophy", "neuroscience"], "stoicism"],
    )
    assert out.startswith("Saved")
    body = (kb / "wiki" / "nested-tags.md").read_text()
    parsed = yaml.safe_load(body.split("---", 2)[1])
    assert parsed["tags"] == ["philosophy", "neuroscience", "stoicism"]


def test_save_knowledge_refuses_tags_that_collapse_to_empty(tmp_path):
    from agent.tools import KBTools

    kb, canon = _kb_dirs(tmp_path)
    tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
    out = tools.save_knowledge("empty-tags.md", "## Body\n\nx.", tags=[[[]]])
    assert "refusing" in out.lower()
    assert "flat" in out.lower()


def test_save_knowledge_refuses_serialized_list_string_as_tag(tmp_path):
    from agent.tools import KBTools

    kb, canon = _kb_dirs(tmp_path)
    tools = KBTools(kb_index=None, kb_dir=kb, canon_dir=canon)
    out = tools.save_knowledge(
        "blob-tag.md",
        "## Body\n\nx.",
        tags=["['a', 'b']"],
    )
    assert "refusing" in out.lower()
