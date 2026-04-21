"""Tests for the canonical KB path helpers and the grep-enforced rule
that raw ``filename ==`` comparisons live only inside src/agent/kb_paths.py."""

from pathlib import Path

import pytest

from agent.kb_paths import (
    KBPathError,
    from_canonical,
    is_canonical,
    parse,
    relpath_of,
    source_of,
    split_canonical,
    to_canonical,
    to_canonical_lenient,
)


class TestRoundTrip:
    def test_canon_roundtrip(self):
        c = to_canonical("canon", "mind-en-place/vault.md")
        assert c == "canon:mind-en-place/vault.md"
        assert from_canonical(c) == ("canon", "mind-en-place/vault.md")

    def test_knowledge_roundtrip(self):
        c = to_canonical("knowledge", "wiki/foo.md")
        assert c == "knowledge:wiki/foo.md"
        assert from_canonical(c) == ("knowledge", "wiki/foo.md")

    def test_raw_under_knowledge(self):
        c = to_canonical("knowledge", "raw/notes/bar.md")
        assert c == "knowledge:raw/notes/bar.md"

    def test_normalises_leading_slash(self):
        assert to_canonical("canon", "/foo/bar.md") == "canon:foo/bar.md"

    def test_normalises_dotslash(self):
        assert to_canonical("canon", "./foo.md") == "canon:foo.md"

    def test_normalises_backslashes(self):
        assert to_canonical("canon", r"foo\bar.md") == "canon:foo/bar.md"


class TestRejectInvalid:
    def test_rejects_bad_source(self):
        with pytest.raises(KBPathError):
            to_canonical("memory", "x.md")  # not a real source root yet

    def test_rejects_dotdot_segments(self):
        with pytest.raises(KBPathError):
            to_canonical("canon", "../escape.md")

    def test_rejects_empty_relpath(self):
        with pytest.raises(KBPathError):
            to_canonical("canon", "")

    def test_from_canonical_rejects_bare(self):
        with pytest.raises(KBPathError):
            from_canonical("foo.md")

    def test_from_canonical_rejects_unknown_source(self):
        with pytest.raises(KBPathError):
            from_canonical("garbage:foo.md")


class TestParseLenient:
    def test_parses_canonical(self):
        assert parse("knowledge:wiki/x.md") == ("knowledge", "wiki/x.md")

    def test_handles_legacy_canon_prefix(self):
        assert parse("canon/mind-en-place/x.md") == ("canon", "mind-en-place/x.md")

    def test_handles_legacy_knowledge_prefix(self):
        assert parse("knowledge/wiki/y.md") == ("knowledge", "wiki/y.md")

    def test_legacy_wiki_prefix_treated_as_knowledge(self):
        assert parse("wiki/foo.md") == ("knowledge", "wiki/foo.md")

    def test_legacy_raw_prefix_treated_as_knowledge(self):
        assert parse("raw/notes/z.md") == ("knowledge", "raw/notes/z.md")

    def test_bare_relpath_uses_default(self):
        assert parse("loose.md") == ("knowledge", "loose.md")

    def test_bare_relpath_default_override(self):
        assert parse("loose.md", default_source="canon") == ("canon", "loose.md")

    def test_to_canonical_lenient(self):
        assert to_canonical_lenient("wiki/foo.md") == "knowledge:wiki/foo.md"

    def test_rejects_traversal_in_legacy(self):
        with pytest.raises(KBPathError):
            parse("knowledge/../etc/passwd")


class TestHelpers:
    def test_is_canonical(self):
        assert is_canonical("canon:foo.md") is True
        assert is_canonical("foo.md") is False
        assert is_canonical("") is False
        assert is_canonical("garbage:foo.md") is False

    def test_split_canonical_returns_none_on_failure(self):
        assert split_canonical("foo.md") is None
        assert split_canonical("canon:foo.md") == ("canon", "foo.md")

    def test_relpath_of_strips_prefix(self):
        assert relpath_of("knowledge:wiki/x.md") == "wiki/x.md"
        assert relpath_of("wiki/x.md") == "wiki/x.md"

    def test_source_of(self):
        assert source_of("canon:y.md") == "canon"
        assert source_of("foo.md") is None


# ---------------------------------------------------------------------------
# Architectural guard: keep raw ``.filename ==`` / ``filename =`` LanceDB
# comparisons inside the small set of files allowed to do them. Any new code
# touching node.filename or the LanceDB filename column should route through
# kb_paths so the canonical convention stays unbroken.
# ---------------------------------------------------------------------------

ALLOWED_RAW_FILENAME_COMPARE_FILES = {
    # Modules that are by-design talking to the raw LanceDB column / graph
    # node attribute. New entries require a code review.
    "src/agent/kb_paths.py",
    "src/knowledge/index.py",
    "src/knowledge/graph.py",
    "src/knowledge/wiki_links.py",
    # Existing tools.py uses node.filename for resolver / related-block
    # logic; the resolver itself routes through kb_paths but the in-place
    # comparisons remain.
    "src/agent/tools.py",
}


def test_no_raw_filename_compares_outside_kb_paths():
    """Grep-based regression: any new ``filename ==`` comparison outside the
    allowlist is a bug — route it through ``kb_paths`` instead."""
    import re

    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "src"
    pattern = re.compile(r"\.filename\s*==|filename\s*=\s*'")
    offenders: list[str] = []
    for py in src_root.rglob("*.py"):
        rel = str(py.relative_to(repo_root))
        if rel in ALLOWED_RAW_FILENAME_COMPARE_FILES:
            continue
        text = py.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")

    assert not offenders, (
        "Raw filename comparisons should route through src/agent/kb_paths.py.\n"
        "Offending lines:\n  - " + "\n  - ".join(offenders)
    )
