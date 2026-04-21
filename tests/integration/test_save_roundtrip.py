"""Integration: native tool_call -> save_knowledge -> file on disk.

This is the test that would have caught A1's bug 1: the homebrew regex
parser silently dropped any save_knowledge whose body contained ``(`` /
``)`` / tables / brackets, and the model fabricated "Page saved" because
it never saw a tool_result back.

We script the chat stream to emit a real Ollama-shaped tool_call,
exercise the full FastAPI tool loop, then assert on disk bytes. If any
future change breaks the parse or the dispatch, this test fails before
the user does.
"""

from __future__ import annotations

from .conftest import content, tool_call


HARD_CONTENT = """## Overview

Slowly Changing Dimensions (SCD Type 2) preserve history (with full row
versioning) so analysts can answer "what did the customer look like on
2024-03-15?" without losing today's state.

## Tables

| dim   | scd_type | reason                                     |
|-------|----------|--------------------------------------------|
| dim_customer | 2 | preserve address + tier (audit trail)   |
| dim_date     | 1 | overwrite (no history needed)           |

## Code Fence

```sql
SELECT customer_id, current_address
FROM dim_customer
WHERE current_flag = TRUE
  AND effective_date <= '2024-03-15'
  AND (end_date IS NULL OR end_date > '2024-03-15');
```

## Brackets

The `[[wiki-link]]` syntax stays verbatim, and so does a markdown link
like [Kimball Method](https://kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/).

## Edge Cases

- Function call notation in prose: `f(x) = x ** 2` is fine.
- Curly braces: `{ "json": "ok" }` should not trip anything.
- Square arrays: `[1, 2, 3]` is just markdown text.
"""


def test_save_knowledge_with_complex_content_lands_on_disk(integration_env):
    """A1 / bug 1 regression. Save a wiki page whose body is full of the
    characters the old regex parser choked on; verify the file is real,
    contains the exact body bytes we asked for, and that the agent saw a
    tool_result back (not a silent drop)."""
    env = integration_env
    kb_dir = env["kb_dir"]

    iterations = [
        [tool_call(
            "save_knowledge",
            filename="dw-to-mep.md",
            content=HARD_CONTENT,
            tags="data-warehouse, kimball, wiki",
        )],
        [content("Saved dw-to-mep.md to wiki/.")],
    ]

    turn = env["run"]("Save the wiki page on dimensional modeling.", iterations)

    saved = kb_dir / "wiki" / "dw-to-mep.md"
    assert saved.exists(), (
        f"save_knowledge did not create the file. tool_results seen: "
        f"{turn.tool_results!r}"
    )

    body = saved.read_text(encoding="utf-8")
    # Spot-check every problematic substring from the original content.
    for needle in [
        "(SCD Type 2)",
        "| dim_customer | 2 |",
        "```sql",
        "WHERE current_flag = TRUE",
        "[[wiki-link]]",
        "[Kimball Method](https://kimballgroup.com",
        "f(x) = x ** 2",
        "{ \"json\": \"ok\" }",
        "[1, 2, 3]",
    ]:
        assert needle in body, (
            f"Saved file is missing the expected substring {needle!r}.\n"
            f"On-disk body was:\n{body}"
        )

    # Auto-frontmatter wraps the agent's body — verify the wrapper landed.
    assert body.startswith("---\n"), "frontmatter wrapper missing"
    assert "tags:" in body.split("---", 2)[1]

    # The runtime must have surfaced a real tool_result for the save call,
    # otherwise the model has no way to know whether the write succeeded.
    save_results = turn.results_for("save_knowledge")
    assert save_results, "no tool_result for save_knowledge — silent-drop regression"
    result_text = save_results[0].get("result", "")
    assert "wiki/dw-to-mep.md" in result_text, (
        f"save_knowledge result didn't reference the saved path: {result_text!r}"
    )

    # And the run wasn't classified as a failed/refused dispatch.
    executed = turn.executed("save_knowledge")
    assert executed, "tool_done shows save_knowledge as not executed"
