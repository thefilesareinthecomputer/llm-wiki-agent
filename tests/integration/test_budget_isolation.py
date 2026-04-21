"""Integration: per-class budgets keep writes alive even when explore
exhausts. This is A3's whole reason to exist — bug 3 was that a shared
cap let exploration starve the actual save the user asked for, and the
forced-summary stream then dropped the save call entirely.

Scenario: explore_budget + 2 distinct ``search_knowledge`` (explore)
calls followed by one ``save_knowledge`` (write) call. The runtime must:
  - run at most ``CLASS_BUDGETS["explore"]`` explore calls
  - REFUSE the last 2 explore calls with a structured message
  - still execute the save_knowledge call (write budget intact)
  - leave a real file on disk
"""

from __future__ import annotations

from .conftest import content, tool_call


def test_write_budget_survives_exhausted_explore(integration_env):
    env = integration_env
    kb_dir = env["kb_dir"]
    from agent.tools import CLASS_BUDGETS
    explore_budget = CLASS_BUDGETS["explore"]

    iterations: list = []
    # Fire 2 extra explore calls past the budget so the refusal path is
    # guaranteed to trigger regardless of what the budget is configured at.
    total_explore_attempts = explore_budget + 2
    for i in range(total_explore_attempts):
        iterations.append([
            tool_call("search_knowledge", query=f"audit-probe-{i}")
        ])
    iterations.append([
        tool_call(
            "save_knowledge",
            filename="post-exhaustion.md",
            content=(
                "## Summary\n\n"
                "This page is saved AFTER the explore budget has already "
                "been exhausted in the same turn. If the per-class budget "
                "isolation (A3) regresses, this file will not exist.\n"
            ),
            tags="audit, regression",
        )
    ])
    iterations.append([content("Saved post-exhaustion.md.")])

    turn = env["run"]("Audit then save.", iterations)

    # Per-class budget: at most CLASS_BUDGETS["explore"] actually executed.
    explore_executed = [
        d for d in turn.tool_done
        if d.get("tool") == "search_knowledge" and d.get("executed")
    ]
    assert len(explore_executed) <= explore_budget, (
        f"explore budget breached: {len(explore_executed)} / cap {explore_budget}"
    )

    # And we must see at least one REFUSED explore result so the model
    # learns *why* it was cut off rather than guessing.
    refused = [
        r for r in turn.tool_results
        if r.get("tool") == "search_knowledge"
        and "REFUSED:" in (r.get("result") or "")
    ]
    assert refused, (
        "expected at least one REFUSED search_knowledge result once the "
        "explore budget exhausted; got tool_results: "
        f"{[r.get('result', '')[:80] for r in turn.tool_results]}"
    )

    # The save MUST have fired even though explore hit zero.
    save_done = [
        d for d in turn.tool_done
        if d.get("tool") == "save_knowledge" and d.get("executed")
    ]
    assert save_done, (
        "save_knowledge did not execute after explore was exhausted — "
        "this is bug 3 regressing"
    )

    saved = kb_dir / "wiki" / "post-exhaustion.md"
    assert saved.exists(), (
        "save_knowledge reported executed but produced no file on disk"
    )
    body = saved.read_text(encoding="utf-8")
    assert "post-exhaustion.md" in body or "post-exhaustion" in body
    assert "If the per-class budget isolation" in body


def test_explore_budget_remaining_advertised_in_results(integration_env):
    """Every framed tool result must include the remaining budget so the
    model sees its own runway and can stop voluntarily before the cap."""
    env = integration_env
    iterations = [
        [tool_call("search_knowledge", query="advertise-1")],
        [tool_call("search_knowledge", query="advertise-2")],
        [content("Done.")],
    ]
    turn = env["run"]("warm up the explore counter", iterations)

    for r in turn.tool_results:
        text = r.get("result", "")
        assert "[remaining_budget:" in text, (
            f"tool_result missing remaining_budget annotation: {text[:160]!r}"
        )


# ---------------------------------------------------------------------------
# P1-1: orient budget class is independent of explore.
# ---------------------------------------------------------------------------

def test_orient_class_exists_with_expected_budget():
    """P1-1: ``orient`` is a top-level class in CLASS_BUDGETS, the three
    cheap orientation tools live in it, and the budget is the documented
    cap."""
    from agent.tools import CLASS_BUDGETS, TOOL_CLASSES

    assert "orient" in CLASS_BUDGETS
    assert CLASS_BUDGETS["orient"] == 5
    for name in ("graph_stats", "list_knowledge", "folder_tree"):
        assert TOOL_CLASSES[name] == "orient", (
            f"{name} should belong to the orient class, got {TOOL_CLASSES[name]}"
        )
    # Heavy tools must stay in explore.
    for name in (
        "read_knowledge_section", "graph_traverse", "graph_neighbors",
        "describe_node", "search_knowledge", "graph_search",
        "read_knowledge",
    ):
        assert TOOL_CLASSES[name] == "explore", (
            f"{name} should remain in explore, got {TOOL_CLASSES[name]}"
        )


def test_orient_exhaustion_does_not_block_explore(integration_env):
    """Burn the orient budget on graph_stats / list_knowledge / folder_tree
    calls, then run a real explore call. The explore call MUST still execute —
    if a regression collapses orient back into explore, the search call will
    refuse with a budget error."""
    env = integration_env
    from agent.tools import CLASS_BUDGETS
    orient_budget = CLASS_BUDGETS["orient"]

    iterations = []
    # Exhaust + overshoot the orient budget.
    for i in range(orient_budget + 2):
        # Rotate the tool names so the dedup guard doesn't kick in before
        # the budget guard.
        names = ("graph_stats", "list_knowledge", "folder_tree")
        name = names[i % len(names)]
        iterations.append([tool_call(name)])
    # Then make an explore call — must execute.
    iterations.append([tool_call("search_knowledge", query="orient-vs-explore")])
    iterations.append([content("Done.")])

    turn = env["run"]("orient first then explore", iterations)

    # Orient budget honored — at most `orient_budget` orient calls executed.
    orient_executed = [
        d for d in turn.tool_done
        if d.get("tool") in ("graph_stats", "list_knowledge", "folder_tree")
        and d.get("executed")
    ]
    assert len(orient_executed) <= orient_budget, (
        f"orient budget breached: {len(orient_executed)} / cap {orient_budget}"
    )

    # And the explore call survived the orient drain.
    explore_done = [
        d for d in turn.tool_done
        if d.get("tool") == "search_knowledge" and d.get("executed")
    ]
    assert explore_done, (
        "search_knowledge did not execute after orient was exhausted — "
        "orient class must be independent of explore"
    )


def test_remaining_budget_lists_orient_class(integration_env):
    """The structured remaining_budget annotation surfaced to the model must
    list the orient class so it can pace itself."""
    env = integration_env
    iterations = [
        [tool_call("graph_stats")],
        [content("Done.")],
    ]
    turn = env["run"]("inspect remaining_budget", iterations)

    text = turn.tool_results[0].get("result", "")
    assert "[remaining_budget:" in text
    assert "orient" in text, (
        f"orient class missing from remaining_budget annotation: {text[:200]!r}"
    )
