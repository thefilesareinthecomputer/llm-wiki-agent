"""Eval: mep_identity_present.

Single prompt-shape audit. Verifies the new mind-en-place agent
identity, the OVERVIEW context block, and the CORE COMMANDS rules all
land in the assembled system prompt. Also checks that the operational
tool-failure rules (the survivors from the previous prompt) live under
the OPERATIONAL HONESTY header so they are not lost in the rewrite.

This is the contract that prevents an accidental revert to the prior
LLM Wiki Agent/Obsidian framing - if any of these strings disappear from the
prompt, this eval fails and the change is visible.
"""

from __future__ import annotations


def _prompt() -> str:
    from web.app import _build_system_prompt

    # tools_enabled covers the full L4 surface; kb_context empty is fine
    # because we only audit identity / behavior / honesty layers here.
    return _build_system_prompt(kb_context="", tools_enabled=True)


def test_l0_role_block_present():
    """ROLE block: synthesis engine + dev coach + SRE for my brain."""
    prompt = _prompt()
    assert "mind-en-place" in prompt.lower(), "L0 must declare the mind-en-place identity."
    assert "Synthesis engine" in prompt
    assert "Development coach" in prompt
    assert "SRE for my brain" in prompt


def test_l1_overview_block_present():
    """OVERVIEW: the always-on context that lets the agent decode
    user-daily-reflections, mep-* files, and the medallion KB shape."""
    prompt = _prompt()
    assert "OVERVIEW" in prompt
    assert "Mind en Place" in prompt or "mind en place" in prompt.lower()
    # The three KB themes the agent must be able to name.
    assert "COMMITMENTS" in prompt
    assert "EXPERIENCE" in prompt
    assert "WISDOM" in prompt
    # Archive-structure tier markers - if these vanish the agent loses
    # the ability to interpret filename prefixes.
    assert "mep-*.md" in prompt
    assert "user-*.md" in prompt
    assert "vault-*.md" in prompt


def test_l1_core_commands_block_present():
    """CORE COMMANDS A-E. Each section header must be present."""
    prompt = _prompt()
    assert "CORE COMMANDS" in prompt
    assert "A. TRUTH" in prompt
    assert "B. NO PLACATION" in prompt
    assert "C. NO MIRRORING" in prompt
    assert "D. NO MANIPULATION" in prompt
    assert "E. CONTRADICTIONS" in prompt


def test_l1_operational_honesty_survives_rewrite():
    """The 6 tool-failure rules from the previous prompt are the
    operational tail that the tool framing relies on. They must
    survive under a clearly labeled header."""
    prompt = _prompt()
    assert "OPERATIONAL HONESTY" in prompt, (
        "Tool-failure rules must live under the OPERATIONAL HONESTY header so "
        "they are visible alongside the new CORE COMMANDS, not buried."
    )
    lower = prompt.lower()
    # The single-retry policy and 'did you mean' loophole are the rule
    # the model is most likely to violate; pin both.
    assert "did you mean" in lower
    assert "retry exactly once" in lower or "exactly one retry" in lower
    # No-fabrication and no-cross-file-blurring rules.
    assert "fabricate" in lower
    assert "merge" in lower or "mix content" in lower


def test_l1_objective_and_task_persona_present():
    """OBJECTIVE + TASK persona: candid, terse, no AI pizzazz, no em
    dashes. Pins the persona surface so a future prompt edit does not
    silently swap the voice back to the prior frame."""
    prompt = _prompt()
    assert "OBJECTIVE" in prompt
    assert "TASK" in prompt
    # Persona references that ground the voice.
    assert "Q from James Bond" in prompt or "Alfred from Batman" in prompt or "Jarvis" in prompt
    # Anti-pattern callouts the persona explicitly bans.
    assert "em dashes" in prompt.lower()
    assert "emojis" in prompt.lower()


def test_old_identity_removed():
    """The prior poetic framing was replaced by user
    direction. Make sure no fragment leaked through the rewrite."""
    prompt = _prompt()
    lower = prompt.lower()
    # These two phrases were unique to the old L0/L1; either one resurfacing
    # means the rewrite was partially reverted.
    assert "keeper of a knowledge vault" not in lower
    assert "obsidian: dark mirror" not in lower
