"""Eval: compile_trigger.

The L4 prompt nudges the agent to run `compile_knowledge(source)` +
`save_knowledge()` whenever it loads the same raw/ source twice in a
session. This eval has two layers:

  1. Prompt audit — the WIKI AUTHORING LOOP block must be present in
     the L4 system prompt and reference both compile_knowledge and
     save_knowledge by name.
  2. Tool wiring — when a scripted model emits a `compile_knowledge`
     call, the dispatcher in `_execute_tool` must route it to the
     real KBTools method (no AttributeError, no silent drop).

The behavioral fire-the-tool decision belongs to the model; this eval
guarantees the agent has both the instruction and the working tool.
"""


def test_l4_contains_wiki_authoring_loop_nudge():
    """The L4 prompt must mention the WIKI AUTHORING LOOP and name both
    tools the agent is supposed to chain."""
    from web.app import _build_system_prompt

    prompt = _build_system_prompt(kb_context="", tools_enabled=True)
    assert "WIKI AUTHORING LOOP" in prompt, (
        "L4 prompt must contain the WIKI AUTHORING LOOP nudge so the agent "
        "knows when to compile a wiki page."
    )
    assert "compile_knowledge" in prompt
    assert "save_knowledge" in prompt
    assert "raw/" in prompt, (
        "Nudge must reference raw/ so the agent recognizes the trigger condition."
    )


def test_compile_knowledge_dispatch_works(eval_run):
    """Scripted model emits a compile_knowledge call. Verify it
    actually executes (vs. failing silently in the dispatcher)."""
    from tests.evals.conftest import content, tool_call

    turn = eval_run(
        "Compile a wiki page from this raw source.",
        [
            [tool_call(
                "compile_knowledge",
                source="raw/agent-development/AGENT-NOTES.md",
            )],
            [content("Plan received. I will now save the page.")],
        ],
    )

    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0]["tool"] == "compile_knowledge"
    assert len(turn.tool_results) == 1
    # The tool either returns a compilation plan, a "no source found" message,
    # or a graph-not-available stub. Any of those means the dispatcher routed
    # the call. What we are guarding against is a silent drop or AttributeError.
    result = turn.tool_results[0].get("result", "")
    assert result, "compile_knowledge must return non-empty output through the dispatcher."
