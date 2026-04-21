"""Eval: stops_after_budget_exhaustion (post A3 per-class budgets).

Replaces the old "max_tool_iterations=4" ceiling with the per-class
budgets introduced in A3:

  CLASS_BUDGETS = {orient: 5, explore: 10, write: 2, maintenance: 3}

Two layers:

  1. Per-class cap - a scripted model that wants to call more explore
     tools than the budget allows will only get ``CLASS_BUDGETS["explore"]``
     executions. Anything past that must come back as a structured
     "REFUSED: explore budget exhausted" tool result so the model knows
     why.
  2. Sufficiency prompt - the L4 SUFFICIENCY block is the prompt-side
     complement to the runtime cap and must land in the assembled
     system prompt.

The model's *behavioral* decision to stop after 2 tools (per the
SUFFICIENCY block) is not enforced here - that is a quality target,
not a contract. What this eval pins is the floor: the loop is bounded
and the stop-cue is delivered.
"""

from __future__ import annotations


def test_explore_budget_caps_at_configured_limit(eval_run):
    """Script explore_budget + 2 explore-class tool calls. Only
    ``CLASS_BUDGETS["explore"]`` should actually run.

    The overflow calls must come back as REFUSED tool results so the
    model understands why further calls were rejected, instead of
    stalling silently.
    """
    from agent.tools import CLASS_BUDGETS
    from tests.evals.conftest import content, tool_call

    explore_budget = CLASS_BUDGETS["explore"]

    # Each iteration emits one explore-class call. We vary the search query
    # so the within-turn dedup guard doesn't suppress the call before the
    # per-class budget gets a chance to refuse. After the budget is exhausted
    # further calls are refused but still appear in tool_calls (as
    # bookkeeping). The last iteration is the model's summary.
    total_attempts = explore_budget + 2
    iterations = [
        [tool_call("search_knowledge", query=f"audit-probe-{i}")]
        for i in range(total_attempts)
    ]
    iterations.append([content("Final summary after the budget caps.")])

    turn = eval_run("Run a deep audit of the KB.", iterations)

    # Tool results are framed as "[TOOL_RESULT: ...]\n<body>\n[remaining_budget: ...]".
    # The body either contains "REFUSED: <class> budget exhausted" (budget hit),
    # "SKIPPED REPEAT CALL" (within-turn dedup), or the actual tool output.
    executed_results = [
        r for r in turn.tool_results
        if "REFUSED:" not in (r.get("result") or "")
        and "SKIPPED REPEAT CALL" not in (r.get("result") or "")
    ]
    refused_results = [
        r for r in turn.tool_results
        if "REFUSED:" in (r.get("result") or "")
    ]

    # graph_stats is also de-duplicated within a turn, so only the first
    # call really executes. Subsequent identical calls come back as
    # SKIPPED REPEAT or REFUSED depending on whether the budget hit
    # first. Either way: zero blow-past.
    assert len(executed_results) <= explore_budget, (
        f"explore budget breached: {len(executed_results)} executions, "
        f"cap is {explore_budget}"
    )
    # And we expect at least some refusals once the budget runs out.
    assert refused_results, (
        "expected REFUSED tool results once the explore budget exhausted"
    )


def test_sufficiency_block_in_system_prompt():
    """The L4 SUFFICIENCY block is the prompt-side complement to the
    runtime cap. Verify it lands in the assembled system prompt."""
    from web.app import _build_system_prompt

    prompt = _build_system_prompt(kb_context="", tools_enabled=True)
    assert "SUFFICIENCY" in prompt, (
        "L4 must contain the SUFFICIENCY block so the model is told to "
        "answer with what it has after 2 tool calls."
    )
    assert "answer-first" in prompt.lower()
    assert ("two explore calls" in prompt.lower()
            or "2 tool calls" in prompt
            or "2 tools" in prompt), (
        "SUFFICIENCY block must reference the answer-after-N-tools rule."
    )


def test_per_class_budgets_advertised_in_l4():
    """L4 must spell out the per-class budget so the model can pace itself."""
    from web.app import _build_system_prompt
    from agent.tools import CLASS_BUDGETS

    prompt = _build_system_prompt(kb_context="", tools_enabled=True)
    assert "PER-TURN BUDGETS" in prompt
    for cls, n in CLASS_BUDGETS.items():
        assert f"{cls}" in prompt, f"L4 missing budget block for class {cls!r}"
        assert str(n) in prompt, (
            f"L4 missing budget value {n} for class {cls!r}"
        )
