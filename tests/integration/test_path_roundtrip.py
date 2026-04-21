"""Integration: anything ``folder_tree()`` advertises must resolve.

A2 unified the filename convention to ``<source>:<relpath>``. Bug 2 was
that ``folder_tree()`` emitted one shape and the resolver expected
another, so every graph_traverse / graph_neighbors from a copied path
returned "Ambiguous: ... matched 153 sections."

This integration test seeds two real files (one canon, one wiki),
builds the index + graph, asks ``folder_tree`` for the listing, then
calls ``graph_traverse`` against each filename it advertises. The
contract is: the resolver never says "Ambiguous" or "FILE NOT FOUND"
for a path the tool itself just emitted.
"""

from __future__ import annotations

import re

from .conftest import tool_call, content


def test_folder_tree_paths_resolve_in_graph_traverse(integration_env):
    env = integration_env
    kb_dir = env["kb_dir"]
    canon_dir = env["canon_dir"]
    kb_index = env["kb_index"]
    tools = env["tools"]

    (canon_dir / "law.md").write_text(
        "# Law\n\n## Principle\n\nA principle.\n", encoding="utf-8"
    )
    (kb_dir / "wiki" / "page.md").write_text(
        "# Page\n\n## Section\n\nA section.\n", encoding="utf-8"
    )

    # Re-build now that real files exist on disk.
    kb_index.build_index(extract_entities=False)

    tree = tools.folder_tree("all")
    # Folder headers carry the source label; the per-file canonical path
    # appears in list_knowledge() / search_knowledge() output further
    # down. The test contract is "anything advertised resolves" — verify
    # both layers.
    assert "[canon] canon/" in tree
    assert "[knowledge] knowledge/" in tree
    assert "wiki/" in tree

    listing = tools.list_knowledge()
    from agent import kb_paths

    expected_paths = [
        kb_paths.to_canonical("canon", "law.md"),
        kb_paths.to_canonical("knowledge", "wiki/page.md"),
    ]
    for canonical in expected_paths:
        assert canonical in listing, (
            f"list_knowledge output did not include canonical path "
            f"{canonical!r}. Got:\n{listing}"
        )

    # Now feed each canonical path back into graph_traverse and confirm
    # the resolver accepts it (no Ambiguous, no FILE NOT FOUND).
    for canonical in expected_paths:
        out = tools.graph_traverse(canonical, depth=1)
        assert "Ambiguous" not in out, (
            f"graph_traverse({canonical!r}) returned Ambiguous: {out!r}"
        )
        assert "FILE NOT FOUND" not in out, (
            f"graph_traverse({canonical!r}) returned FILE NOT FOUND: {out!r}"
        )


def test_legacy_paths_still_resolve(integration_env):
    """Backward compatibility: the resolver must keep accepting bare
    relpaths so older agent prompts and links don't break overnight."""
    env = integration_env
    kb_dir = env["kb_dir"]
    canon_dir = env["canon_dir"]
    kb_index = env["kb_index"]
    tools = env["tools"]

    (canon_dir / "legacy.md").write_text(
        "# Legacy\n\n## Section\n\nLegacy body.\n", encoding="utf-8"
    )
    kb_index.build_index(extract_entities=False)

    # Bare basename, plus tier-prefixed legacy form.
    for arg in ("legacy.md", "canon/legacy.md"):
        out = tools.graph_traverse(arg, depth=1)
        assert "FILE NOT FOUND" not in out, (
            f"graph_traverse({arg!r}) failed legacy resolution: {out!r}"
        )
