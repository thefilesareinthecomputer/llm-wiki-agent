"""Canonical KB path helpers.

The agent and the index use a single canonical filename format everywhere:

    <source>:<relpath>

Examples:

    canon:mind-en-place/vault-philosophy-quotes.md
    knowledge:wiki/dw-to-mep.md
    knowledge:raw/technology/Data Warehouse Toolkit.md

Why one format: prior to this convention the heading-tree printer emitted
``canon/foo.md`` (with the source folded into the path), the LanceDB column
stored ``foo.md`` plus a separate ``source`` field, and the wiki-link resolver
accepted bare ``foo.md``. The agent copied paths from one printer into a
resolver that expected a different shape and graph traversal blew up with
"matched 153 sections" ambiguity errors. This module is the single point of
truth for splitting/joining ``(source, relpath)`` pairs.

Rules:

- ``source`` is always one of ``"canon"`` or ``"knowledge"``. The medallion
  tier (``canon``/``wiki``/``raw``) is computed from ``(source, relpath)`` by
  ``KBIndex._compute_tier`` and is NOT part of the canonical path.
- ``relpath`` is forward-slash, never starts with ``/``, never contains ``..``.
  It is the path under the ``canon/`` or ``knowledge/`` source root that
  LanceDB stores in its ``filename`` column.
- Bare relpaths (no ``:``) coming from the agent are accepted by ``parse``
  with a best-effort source guess but ``to_canonical`` always produces the
  full prefixed form for output.

Anywhere outside this module that compares ``node.filename`` directly to a
user/agent-supplied string is a bug; route through ``parse`` instead. The
test suite has a regression that greps for bare ``.filename ==`` comparisons
outside this file (see ``tests/test_kb_paths.py::test_no_raw_filename_compares_outside_kb_paths``).
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Optional

VALID_SOURCES = ("canon", "knowledge")
TIER_PREFIXES = ("wiki", "raw")  # subdirs under knowledge/ that the agent often inlines


class KBPathError(ValueError):
    """Raised when a string cannot be coerced into a canonical KB path."""


def to_canonical(source: str, relpath: str) -> str:
    """Build the canonical ``<source>:<relpath>`` string.

    ``source`` must be ``"canon"`` or ``"knowledge"``. ``relpath`` is
    normalized (forward slashes, no leading ``/``, no ``..`` segments).
    Trailing whitespace and a leading ``./`` are stripped.
    """
    src = (source or "").strip().lower()
    if src not in VALID_SOURCES:
        raise KBPathError(
            f"Invalid source '{source}' (must be one of {VALID_SOURCES})."
        )
    rel = _normalize_relpath(relpath)
    if not rel:
        raise KBPathError("Empty relpath.")
    return f"{src}:{rel}"


def from_canonical(canonical: str) -> tuple[str, str]:
    """Parse a canonical ``<source>:<relpath>`` string into ``(source, relpath)``.

    Raises :class:`KBPathError` when the string lacks the prefix or has an
    unknown source. Use :func:`parse` instead when accepting user/agent input,
    which falls back to a best-effort guess for legacy bare relpaths.
    """
    if not canonical or ":" not in canonical:
        raise KBPathError(
            f"Not a canonical KB path: '{canonical}' "
            f"(expected '<source>:<relpath>' format)."
        )
    src, _, rel = canonical.partition(":")
    src = src.strip().lower()
    if src not in VALID_SOURCES:
        raise KBPathError(
            f"Invalid source '{src}' in '{canonical}' "
            f"(must be one of {VALID_SOURCES})."
        )
    rel = _normalize_relpath(rel)
    if not rel:
        raise KBPathError(f"Empty relpath in '{canonical}'.")
    return src, rel


def is_canonical(s: str) -> bool:
    """Return True iff ``s`` parses as a canonical ``<source>:<relpath>``."""
    if not s or ":" not in s:
        return False
    try:
        from_canonical(s)
    except KBPathError:
        return False
    return True


def parse(
    user_input: str,
    *,
    default_source: str = "knowledge",
) -> tuple[str, str]:
    """Best-effort coercion of an agent/user string into ``(source, relpath)``.

    Resolution order:

    1. If ``user_input`` already has the canonical ``<source>:<relpath>`` form,
       parse it directly.
    2. If ``user_input`` starts with ``canon/`` or ``knowledge/``, peel the
       prefix off and use it as the source.
    3. If the relpath starts with ``raw/`` or ``wiki/`` (legacy tier prefix),
       treat it as a knowledge-tier relpath; canon/ is the only path that
       lives under canon/.
    4. Fall back to ``default_source`` and the cleaned relpath.

    Raises :class:`KBPathError` when the input is empty / contains ``..`` /
    is otherwise unsafe.
    """
    if not user_input or not user_input.strip():
        raise KBPathError("Empty filename.")

    s = user_input.strip()

    # 1. canonical form
    if ":" in s:
        head, _, tail = s.partition(":")
        if head.strip().lower() in VALID_SOURCES:
            return from_canonical(s)
        # Not a known source — fall through; could be a Windows drive letter
        # or accidental colon in a heading. Don't blindly split.

    # 2. <source>/<relpath>
    cleaned = s.lstrip("/").replace("\\", "/")
    head, _, rest = cleaned.partition("/")
    if head in VALID_SOURCES and rest:
        return head, _normalize_relpath(rest)

    # 3. legacy tier-prefixed relpath under knowledge/
    if head in TIER_PREFIXES:
        return "knowledge", _normalize_relpath(cleaned)

    # 4. default source
    return default_source, _normalize_relpath(cleaned)


def to_canonical_lenient(user_input: str, *, default_source: str = "knowledge") -> str:
    """Convenience: parse + emit canonical, raising on unsafe input."""
    src, rel = parse(user_input, default_source=default_source)
    return to_canonical(src, rel)


def split_canonical(canonical: str) -> Optional[tuple[str, str]]:
    """Like :func:`from_canonical` but returns None on parse failure instead of raising."""
    try:
        return from_canonical(canonical)
    except KBPathError:
        return None


def relpath_of(canonical_or_relpath: str) -> str:
    """Extract just the relpath portion. Accepts either canonical or bare input."""
    if ":" in canonical_or_relpath:
        try:
            _, rel = from_canonical(canonical_or_relpath)
            return rel
        except KBPathError:
            pass
    return _normalize_relpath(canonical_or_relpath)


def source_of(canonical: str) -> Optional[str]:
    """Extract just the source portion of a canonical path. None when missing."""
    parsed = split_canonical(canonical)
    return parsed[0] if parsed else None


def _normalize_relpath(rel: str) -> str:
    """Normalize a relpath: forward slashes, no leading slash, no '..' segments."""
    if rel is None:
        return ""
    r = rel.strip().replace("\\", "/")
    if r.startswith("./"):
        r = r[2:]
    r = r.lstrip("/")
    if not r:
        return ""
    parts = [p for p in r.split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise KBPathError(
            f"Refusing relpath with '..' segment: '{rel}'."
        )
    # PurePosixPath normalises ``a//b`` -> ``a/b`` etc.
    return str(PurePosixPath(*parts))
