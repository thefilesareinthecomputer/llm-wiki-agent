"""B1 — sidebar pagination + memory tier.

Two tightly scoped checks:

1. ``GET /conversations`` honors ``limit`` / ``offset`` and exposes the full
   count via ``X-Total-Count`` so the UI can show "show N more" without an
   extra round-trip.
2. The medallion tiering recognizes ``knowledge/memory/`` as a distinct tier
   between raw (bronze) and wiki (silver), and the folder tree advertises it
   when the directory exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. /conversations pagination + X-Total-Count
# ---------------------------------------------------------------------------


def _seed_conversations(client, n: int) -> list[str]:
    ids = []
    for i in range(n):
        resp = client.post("/conversations", json={})
        ids.append(resp.json()["id"])
    return ids


def test_default_conversations_endpoint_unchanged_for_legacy_callers(
    client_with_init,
):
    """Legacy callers (and tests) call ``GET /conversations`` and unpack a
    flat list. Pagination must not break that contract.
    """
    _seed_conversations(client_with_init, 3)
    resp = client_with_init.get("/conversations")
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) >= 3


def test_conversations_limit_truncates_and_exposes_total(client_with_init):
    _seed_conversations(client_with_init, 12)
    resp = client_with_init.get("/conversations?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 5
    total = int(resp.headers.get("X-Total-Count", "0"))
    assert total >= 12, (
        "X-Total-Count must reflect the unpaginated total so the sidebar "
        "knows whether to render a 'show more' affordance."
    )


def test_conversations_offset_pages_correctly(client_with_init):
    _seed_conversations(client_with_init, 6)
    page1 = client_with_init.get("/conversations?limit=3&offset=0").json()
    page2 = client_with_init.get("/conversations?limit=3&offset=3").json()
    assert len(page1) == 3
    assert len(page2) >= 3
    page1_ids = {c["id"] for c in page1}
    page2_ids = {c["id"] for c in page2}
    assert page1_ids.isdisjoint(page2_ids), (
        "offset must yield non-overlapping pages or pagination is broken"
    )


def test_conversations_sorted_most_recent_first(client_with_init):
    """Sidebar contract: the conversation with the most recent activity is
    on top so the user always sees their current thread first.
    """
    from web.app import memory_store

    convs = _seed_conversations(client_with_init, 3)
    # Touch the OLDEST conv last so it becomes the most recent.
    memory_store.add_turn("user", "ping", conversation_id=convs[0])

    resp = client_with_init.get("/conversations?limit=10")
    body = resp.json()
    ids = [c["id"] for c in body]
    assert ids[0] == convs[0], (
        "Touching the oldest conversation should bubble it to the top of "
        f"the sidebar. Order received: {ids}, expected first: {convs[0]}"
    )


# ---------------------------------------------------------------------------
# 2. Memory tier surfaces in folder_tree + tier classification
# ---------------------------------------------------------------------------


def test_compute_tier_recognizes_memory_subdir():
    """``knowledge/memory/<anything>`` resolves to the memory tier so the
    rest of the stack (search weights, badges, frontmatter) treats it
    distinctly from raw / wiki.
    """
    from knowledge.index import KBIndex, TIER_MEMORY, TIER_WIKI, TIER_RAW

    assert KBIndex._compute_tier("knowledge", "memory/threads/2026-04-18.md") == TIER_MEMORY
    assert KBIndex._compute_tier("knowledge", "memory") == TIER_MEMORY
    assert KBIndex._compute_tier("knowledge", "wiki/cortisol.md") == TIER_WIKI
    assert KBIndex._compute_tier("knowledge", "raw/source.md") == TIER_RAW


def test_tier_search_weights_order_memory_between_raw_and_wiki():
    """Search ranking must reflect tier authority: canon > wiki > memory > raw."""
    from knowledge.index import (
        TIER_SEARCH_WEIGHTS, TIER_CANON, TIER_WIKI, TIER_MEMORY, TIER_RAW,
    )
    assert (
        TIER_SEARCH_WEIGHTS[TIER_CANON]
        > TIER_SEARCH_WEIGHTS[TIER_WIKI]
        > TIER_SEARCH_WEIGHTS[TIER_MEMORY]
        > TIER_SEARCH_WEIGHTS[TIER_RAW]
    ), TIER_SEARCH_WEIGHTS


def test_folder_tree_labels_memory_directory(tmp_path):
    """When ``knowledge/memory/`` exists with content, the rendered tree
    advertises the [memory] tier badge so the agent learns the folder is
    real and addressable.
    """
    from knowledge.graph import (
        KnowledgeGraph, build_folder_tree, format_folder_tree,
    )

    kb_dir = tmp_path / "knowledge"
    canon_dir = tmp_path / "canon"
    (kb_dir / "wiki").mkdir(parents=True)
    (kb_dir / "memory").mkdir(parents=True)
    (kb_dir / "raw").mkdir(parents=True)
    canon_dir.mkdir(parents=True)
    (kb_dir / "wiki" / "cortisol.md").write_text("# Cortisol\n")
    (kb_dir / "memory" / "thread-2026-04-18.md").write_text("# Thread\n")
    (kb_dir / "raw" / "source.md").write_text("# Source\n")

    graph = KnowledgeGraph(persist_path=tmp_path / "graph.json")
    for node, edge in build_folder_tree(kb_dir, canon_dir):
        if node:
            graph.add_node(node)
        if edge:
            graph.add_edge(edge)

    tree = format_folder_tree(graph, source="knowledge")
    assert "[memory]" in tree, (
        "knowledge/memory/ must show up with a [memory] tier badge so the "
        f"agent can navigate to it. Got tree:\n{tree}"
    )
    assert "[wiki]" in tree
    assert "[raw]" in tree


def test_folder_tree_root_label_mentions_memory_tier(tmp_path):
    """The knowledge-root header line in the rendered tree must mention
    that the memory tier exists alongside wiki/raw, so the agent reads
    the legend right at the top of the tree.
    """
    from knowledge.graph import (
        KnowledgeGraph, build_folder_tree, format_folder_tree,
    )

    kb_dir = tmp_path / "knowledge"
    canon_dir = tmp_path / "canon"
    kb_dir.mkdir()
    canon_dir.mkdir()
    (kb_dir / "wiki").mkdir()
    (kb_dir / "wiki" / "page.md").write_text("# Page\n")

    graph = KnowledgeGraph(persist_path=tmp_path / "graph.json")
    for node, edge in build_folder_tree(kb_dir, canon_dir):
        if node:
            graph.add_node(node)
        if edge:
            graph.add_edge(edge)

    tree = format_folder_tree(graph, source="knowledge")
    assert "memory tier" in tree.lower(), (
        f"Knowledge-root legend should advertise the memory tier; got:\n{tree}"
    )
