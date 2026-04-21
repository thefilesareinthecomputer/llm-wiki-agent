"""R11: `token_usage` SSE must carry the real rolling total and the real
model context window, not the `len(full_response)//4` / hardcoded 256000
values the original implementation shipped.

Two regression anchors:

1. The **final** `token_usage` event must report `used` that scales with
   the full message list (system prompt + history + tool results), not
   just the assistant's output. An empty assistant response with heavy
   tool output must still show non-trivial `used`.
2. The `total` field must equal the live model's context window. With
   the test gateway fixed on `qwen3:0.6b`, that is 40000 per
   `MODEL_CONTEXT_WINDOWS` — NOT 256000.
3. Mid-loop `token_usage` events must fire after each tool round so the
   UI counter updates live, not only at end-of-turn.
"""

from __future__ import annotations

from .conftest import content, tool_call


def test_token_usage_reports_real_context_window(integration_env):
    env = integration_env
    # One explore + final content so we exercise both the mid-loop and
    # final emit paths.
    turn = env["run"](
        "Give me a quick probe.",
        [
            [tool_call("search_knowledge", query="stoicism")],
            [content("Short answer.")],
        ],
    )

    assert turn.token_usage, (
        "no token_usage SSE events observed — the UI counter would be blank. "
        "The tool-loop and final emits are both required."
    )

    from web.app import _context_window_for
    expected_total = _context_window_for("qwen3:0.6b")

    # Every single event must declare the correct model window. If any
    # event smuggles back the old hardcoded 256000, the UI will over-
    # report headroom and the user won't see compaction pressure coming.
    for evt in turn.token_usage:
        assert evt.get("total") == expected_total, (
            f"token_usage emitted total={evt.get('total')}; expected "
            f"{expected_total} (the actual qwen3:0.6b context window). "
            "This is R11 regressing — do not hardcode 256000."
        )
        assert evt.get("total") != 256000, (
            "token_usage still hardcodes 256000 — regressing R11"
        )


def test_token_usage_final_is_rolling_not_response_tokens(integration_env):
    """Final `used` must be a rolling total across the whole message
    list, not `len(full_response)//4`. With a trivial assistant reply
    and a real tool result, the rolling total dwarfs the response-only
    count — that gap is exactly what the original bug hid."""
    env = integration_env
    turn = env["run"](
        "Probe the KB.",
        [
            [tool_call("search_knowledge", query="anything")],
            [content("ok.")],
        ],
    )

    assert turn.token_usage, "no token_usage SSE events observed"
    final = [u for u in turn.token_usage if u.get("phase") == "final"]
    assert final, "no final-phase token_usage event emitted"
    final_evt = final[-1]

    # `response_tokens` (the old bug) would be ~len("ok.")/4 == 0.
    # The rolling total MUST include system prompt + tool result framing +
    # user message, so it should be comfortably above a hundred tokens.
    assert final_evt.get("used", 0) > 50, (
        f"final token_usage reported used={final_evt.get('used')}; "
        "this looks like the old len(full_response)//4 bug. The value "
        "must be _count_messages_tokens(messages) — the rolling total."
    )


def test_token_usage_mid_loop_emits_per_tool_round(integration_env):
    """Mid-loop `token_usage` events must fire after each tool round
    with `phase: 'tool_loop'` so the UI updates incrementally instead
    of jumping from 0 to the final number at end-of-turn."""
    env = integration_env
    turn = env["run"](
        "Run two tool probes then answer.",
        [
            [tool_call("search_knowledge", query="one")],
            [tool_call("search_knowledge", query="two")],
            [content("Done.")],
        ],
    )

    tool_loop_events = [
        u for u in turn.token_usage if u.get("phase") == "tool_loop"
    ]
    assert len(tool_loop_events) >= 2, (
        "expected >=2 tool_loop token_usage events (one per tool round); "
        f"got {len(tool_loop_events)}. Mid-loop counter updates are part "
        "of R11 — if this breaks, the UI goes back to a stale counter "
        "during long sessions."
    )

    # The rolling total should monotonically climb across rounds because
    # each round adds a tool message to the list. It is allowed to stay
    # flat only if compaction triggered (unlikely at this scale) or the
    # same message count was re-measured.
    useds = [u["used"] for u in tool_loop_events]
    assert useds == sorted(useds), (
        f"mid-loop used values not monotonic: {useds}. Each tool round "
        "should add tokens; a drop means the counter is being reset."
    )
