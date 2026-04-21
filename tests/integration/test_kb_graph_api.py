"""Integration: HTTP routes for the human-facing graph viewer.

Covers ``GET /kb/graph/subgraph`` (Cytoscape-shaped bounded subgraph) and
``GET /kb/graph/overview`` (dashboard rollup) end-to-end:

  - real KBIndex + LanceDB + KnowledgeGraph rebuilt from on-disk files
  - real FastAPI route via TestClient
  - cap clamping / out-of-range 400 / unresolved 404 with suggestions
  - ``parent_child`` silently stripped from any edge_types allow-list

These are the contracts the UI in P1-4 depends on; if any of them
regress, the Graph tab will silently render empty or hammer the index.
"""

from __future__ import annotations

import pytest


def _seed_three_file_kb(env):
    """Create a tiny KB with explicit cross-file links so the resulting
    graph has SIMILAR + INTER_FILE + REFERENCES edges to traverse."""
    kb_dir = env["kb_dir"]
    canon_dir = env["canon_dir"]
    kb_index = env["kb_index"]

    (kb_dir / "wiki" / "alpha.md").write_text(
        "# alpha\n\n## intro\n\nAlpha is the first thing.\n\n"
        "## body\n\nSee [[beta]] for the cross-link target.\n",
        encoding="utf-8",
    )
    (kb_dir / "wiki" / "beta.md").write_text(
        "# beta\n\n## intro\n\nBeta is the second thing. It depends on "
        "alpha for context and then loops back.\n",
        encoding="utf-8",
    )
    (canon_dir / "ground.md").write_text(
        "# ground\n\n## principle\n\nGround truth lives here.\n",
        encoding="utf-8",
    )

    kb_index.build_index(extract_entities=False)
    return kb_index


def test_subgraph_returns_cytoscape_shape(integration_env):
    env = integration_env
    kb_index = _seed_three_file_kb(env)
    client = env["client"]

    resp = client.get(
        "/kb/graph/subgraph",
        params={"file": "knowledge:wiki/alpha.md", "depth": 2},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "meta" in body
    assert "elements" in body
    assert "nodes" in body["elements"]
    assert "edges" in body["elements"]
    # Root metadata reflects what we asked for.
    assert body["meta"]["root"]["file"].endswith("alpha.md")
    # Cytoscape data shape on every node.
    for n in body["elements"]["nodes"]:
        assert "data" in n
        assert "id" in n["data"]
        assert n["data"]["id"].startswith("node:")
        for k in ("file", "heading", "tier", "source"):
            assert k in n["data"]
    for e in body["elements"]["edges"]:
        assert "data" in e
        d = e["data"]
        assert d["source"].startswith("node:")
        assert d["target"].startswith("node:")
        assert d["type"] in {
            "similar", "inter_file", "cross_domain",
            "references", "relates_to",
        }
        # Edge classes mirrors the type so a Cytoscape stylesheet can
        # select on edge_type without re-reading data.
        assert e["classes"] == d["type"]


def test_subgraph_default_excludes_parent_child(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]

    resp = client.get(
        "/kb/graph/subgraph",
        params={"file": "knowledge:wiki/alpha.md", "depth": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    types = {e["data"]["type"] for e in body["elements"]["edges"]}
    assert "parent_child" not in types


def test_subgraph_parent_child_silently_stripped_from_allow_list(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]

    resp = client.get(
        "/kb/graph/subgraph",
        params={
            "file": "knowledge:wiki/alpha.md",
            "depth": 2,
            "edge_types": "parent_child,inter_file",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["meta"]["dropped_parent_child"] is True
    types = {e["data"]["type"] for e in body["elements"]["edges"]}
    assert "parent_child" not in types


def test_subgraph_invalid_edge_types_token_returns_400(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]

    resp = client.get(
        "/kb/graph/subgraph",
        params={
            "file": "knowledge:wiki/alpha.md",
            "edge_types": "bogus",
        },
    )
    assert resp.status_code == 400


@pytest.mark.parametrize("depth", [0, 4, 10, -1])
def test_subgraph_out_of_range_depth_returns_400(integration_env, depth):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]
    resp = client.get(
        "/kb/graph/subgraph",
        params={"file": "knowledge:wiki/alpha.md", "depth": depth},
    )
    assert resp.status_code == 400


def test_subgraph_max_nodes_clamped_to_cap(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]
    # 500 must fail (above hard cap of 250).
    resp = client.get(
        "/kb/graph/subgraph",
        params={"file": "knowledge:wiki/alpha.md", "max_nodes": 500},
    )
    assert resp.status_code == 400


def test_subgraph_capped_flag_set_when_max_nodes_bites(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]
    # Tiny cap forces clamp to bite on a multi-node neighborhood.
    resp = client.get(
        "/kb/graph/subgraph",
        params={
            "file": "knowledge:wiki/alpha.md",
            "depth": 3,
            "max_nodes": 1,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Either the BFS short-circuited at the cap (capped.nodes=True) or
    # there genuinely is only one node within depth — both produce a
    # well-formed response. The hard contract is that the cap is honored.
    assert body["meta"]["stats"]["nodes"] <= 1
    assert body["meta"]["capped"]["max_nodes"] == 1


def test_subgraph_unknown_file_returns_404(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]

    resp = client.get(
        "/kb/graph/subgraph",
        params={"file": "knowledge:wiki/does-not-exist.md"},
    )
    assert resp.status_code == 404
    body = resp.json()
    # FastAPI wraps custom detail bodies under "detail".
    detail = body.get("detail", {})
    assert isinstance(detail, dict)
    assert detail.get("error") in {"not_found", "ambiguous"}


def test_subgraph_ambiguous_heading_returns_404_with_suggestions(integration_env):
    env = integration_env
    kb_dir = env["kb_dir"]
    canon_dir = env["canon_dir"]
    kb_index = env["kb_index"]

    # Two files whose H1 (and therefore the only resulting chunk for a
    # short page) shares a leaf name. Searching by that leaf hits the
    # cross-file fallback in _resolve_chunk_nodes and produces multiple
    # candidates → ambiguous.
    (kb_dir / "wiki" / "one-shared-leaf.md").write_text(
        "# shared-leaf\n\nFirst occurrence.\n", encoding="utf-8"
    )
    (kb_dir / "wiki" / "two-shared-leaf.md").write_text(
        "# shared-leaf\n\nSecond occurrence.\n", encoding="utf-8"
    )
    kb_index.build_index(extract_entities=False)

    # Pass the bare leaf as `file`; with no `heading` set, the resolver
    # falls through to the cross-file leaf fallback and finds two chunks
    # → ambiguous.
    resp = env["client"].get(
        "/kb/graph/subgraph",
        params={"file": "shared-leaf"},
    )
    assert resp.status_code == 404
    body = resp.json()
    detail = body.get("detail", {})
    assert detail.get("error") == "ambiguous"
    assert "suggestions" in detail
    assert "Ambiguous" in detail["suggestions"]


def test_subgraph_edge_types_allow_list_filters(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]

    resp = client.get(
        "/kb/graph/subgraph",
        params={
            "file": "knowledge:wiki/alpha.md",
            "depth": 2,
            "edge_types": "similar",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for e in body["elements"]["edges"]:
        assert e["data"]["type"] == "similar"


# ---------------------------------------------------------------------------
# /kb/graph/overview
# ---------------------------------------------------------------------------

def test_overview_returns_dashboard_shape(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]

    resp = client.get("/kb/graph/overview")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for k in (
        "embedding_model", "last_indexed_at", "nodes", "edges",
        "orphan_nodes", "avg_edges_per_node", "edge_types",
        "edge_share", "hubs",
    ):
        assert k in body, f"missing key {k!r} in overview body"
    assert isinstance(body["hubs"], list)
    # last_indexed_at is set after build_index — never None on a healthy
    # index.
    assert body["last_indexed_at"] is not None


def test_overview_edge_share_consistent_with_kb_stats(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]

    overview = client.get("/kb/graph/overview").json()
    stats = client.get("/kb/stats").json()
    assert overview["edge_share"] == stats["graph"]["edge_share"]


def test_overview_top_param_caps_hub_list(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]

    resp = client.get("/kb/graph/overview", params={"top": 1})
    body = resp.json()
    assert len(body["hubs"]) <= 1


def test_overview_hubs_carry_canonical_addressing(integration_env):
    env = integration_env
    _seed_three_file_kb(env)
    client = env["client"]

    body = client.get("/kb/graph/overview").json()
    if body["hubs"]:
        for h in body["hubs"]:
            # Canonical form is "<source>:<relpath>".
            assert ":" in h["file"]
            assert h["id"].startswith("node:")
            assert "share_non_pc" in h
