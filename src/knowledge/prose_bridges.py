"""Heuristic prose-pattern bridge extraction (P2.2).

When the agent writes prose that mentions two known KB pages in the same
sentence — e.g. "Cortisol regulates the stress response" where both
`cortisol.md` and `stress.md` exist — we emit a soft REFERENCES edge with
`link_kind="prose"` and the originating sentence as `evidence`. This gives
graph-traversal queries a path between conceptually-linked pages even when
the agent forgot to write an explicit `[[wiki link]]`.

Conservative rules — to avoid noise:

  1. Both endpoints must be **distinct** known pages (page slug or H1).
  2. Mentions must be **whole-word** matches, not substrings.
  3. Mentions inside link syntax (`[[...]]`, `[..](.md)`) are stripped
     first so we don't duplicate edges P2.1 already created.
  4. Mentions inside fenced code or inline code are stripped.
  5. Sentences must contain a **verb-like connector** between the two
     mentions (heuristic: any non-trivial word between them). Pure
     enumerations like "see cortisol, dopamine, serotonin" are skipped.
  6. A given (subject, object) pair gets at most one edge per source
     chunk — duplicates are deduped by the graph layer anyway, but we
     stop early to keep `evidence` tied to the first sentence.

The output schema mirrors `wiki_links.parse_links` so the index builder
can treat both as a uniform "explicit reference" stream.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Strip code first (same patterns as wiki_links to keep behaviour aligned)
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

# Strip link syntax — wiki_links.py owns those edges.
_WIKI_LINK_RE = re.compile(r"\[\[[^\[\]\n]+\]\]")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\[\]\n]+\]\([^()\n\s]+\)")

# Sentence boundary — naive but cheap. Splits on . ! ? followed by
# whitespace or end-of-string. Avoids splitting on common abbreviations
# by requiring the punctuation to be followed by whitespace/newline.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _strip_noise(body: str) -> str:
    """Replace code blocks and link syntax with whitespace so character
    offsets remain comparable to the original. We don't need offsets,
    but it keeps sentence splitting honest."""
    def _blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())

    body = _FENCED_CODE_RE.sub(_blank, body)
    body = _INLINE_CODE_RE.sub(_blank, body)
    body = _WIKI_LINK_RE.sub(_blank, body)
    body = _MARKDOWN_LINK_RE.sub(_blank, body)
    return body


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences. Trims and drops empties."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _build_mention_pattern(slug: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive pattern for a page slug.

    Slug is treated literally (no regex meta interpretation). Hyphens
    and underscores in the slug match either form in the prose so
    `cortisol-regulation` matches both 'cortisol-regulation' and
    'cortisol regulation'."""
    pieces = re.split(r"[-_\s]+", slug.strip())
    pieces = [re.escape(p) for p in pieces if p]
    if not pieces:
        return re.compile(r"(?!x)x")  # never matches
    body = r"[-_\s]+".join(pieces)
    # \b is greedy with non-word chars; use lookarounds for resilience
    return re.compile(rf"(?<![\w]){body}(?![\w])", re.IGNORECASE)


def _has_connector(
    sentence: str,
    span_a: tuple[int, int],
    span_b: tuple[int, int],
    other_mention_texts: Iterable[str] = (),
) -> bool:
    """Conservative: there must be at least one verb-like word between
    the two mentions for it to count as a relation, not an enumeration.

    'cortisol regulates stress'         → connector = 'regulates' → bridge
    'cortisol, stress, dopamine'        → connector = ',' / ' '   → no bridge
    'cortisol, dopamine, serotonin'     → middle is another page name
                                           → no bridge (enumeration)
    """
    lo = min(span_a[1], span_b[1])
    hi = max(span_a[0], span_b[0])
    if hi <= lo:
        return False
    middle = sentence[lo:hi]
    tokens = re.findall(r"[A-Za-z]{3,}", middle)
    if not tokens:
        return False
    GLUE = {"and", "or", "the", "with", "for", "from", "into", "onto"}
    other_lower = {t.lower() for t in other_mention_texts if t}
    meaningful = [
        t for t in tokens
        if t.lower() not in GLUE and t.lower() not in other_lower
    ]
    return bool(meaningful)


def compile_page_index(known_pages: Iterable[dict]) -> tuple[re.Pattern | None, dict]:
    """Build a single combined alternation regex over all known page slugs.

    Returns (combined_pattern, slug_to_page) where:
      - combined_pattern is a case-insensitive whole-word matcher for any
        slug or alias (or None if the corpus has fewer than 2 valid slugs).
      - slug_to_page maps the normalised match text → page dict. The
        match's lowered+whitespace-collapsed text is the lookup key.

    Compiled ONCE per build_index pass and reused across all chunks for
    O(N) scanning instead of O(pages × chunks).
    """
    from pathlib import Path

    pages = list(known_pages)
    if not pages:
        return None, {}

    slug_to_page: dict[str, dict] = {}
    raw_slugs: list[str] = []
    for p in pages:
        fname = p.get("filename") or ""
        if not fname:
            continue
        slugs: list[str] = [Path(fname).stem]
        for a in p.get("aliases") or []:
            if a:
                slugs.append(str(a))
        for s in slugs:
            norm = re.sub(r"[-_\s]+", " ", s.lower().strip())
            if not norm or len(norm) < 3:
                continue
            if norm not in slug_to_page:
                slug_to_page[norm] = p
                raw_slugs.append(s)

    if len(slug_to_page) < 2:
        return None, slug_to_page

    raw_slugs.sort(key=len, reverse=True)
    parts = []
    for s in raw_slugs:
        pieces = re.split(r"[-_\s]+", s.strip())
        pieces = [re.escape(p) for p in pieces if p]
        if not pieces:
            continue
        parts.append(r"[-_\s]+".join(pieces))
    if len(parts) < 2:
        return None, slug_to_page
    combined = re.compile(
        rf"(?<![\w])(?:{'|'.join(parts)})(?![\w])",
        re.IGNORECASE,
    )
    return combined, slug_to_page


def find_bridges(
    body: str,
    known_pages: Iterable[dict],
    *,
    compiled: tuple[re.Pattern | None, dict] | None = None,
) -> list[dict]:
    """Return prose-bridge candidates from a chunk body.

    Args:
        body: Raw chunk markdown.
        known_pages: Iterable of page dicts. See `compile_page_index`.
        compiled: Optional pre-compiled (pattern, slug_to_page) tuple from
            `compile_page_index` to avoid recompiling the alternation
            regex per chunk during a build pass.

    Returns: list of bridge dicts:
        {
            "subject_file": str,
            "subject_source": str,
            "object_file": str,
            "object_source": str,
            "evidence": str,         # the sentence
            "subject_match": str,    # exact text matched for subject
            "object_match": str,     # exact text matched for object
        }

    Bridges are returned in document order. The same (subject, object)
    pair appears at most once per `find_bridges` call — only the first
    qualifying sentence is recorded so `evidence` stays representative.
    """
    if not body:
        return []
    if compiled is not None:
        combined, slug_to_page = compiled
    else:
        combined, slug_to_page = compile_page_index(known_pages)
    if combined is None or len(slug_to_page) < 2:
        return []

    cleaned = _strip_noise(body)
    out: list[dict] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()

    for sentence in _split_sentences(cleaned):
        mentions: list[tuple[tuple[int, int], str, dict]] = []
        for m in combined.finditer(sentence):
            text = m.group(0)
            norm = re.sub(r"[-_\s]+", " ", text.lower().strip())
            page = slug_to_page.get(norm)
            if page is None:
                continue
            mentions.append(((m.start(), m.end()), text, page))
        if len(mentions) < 2:
            continue

        mentions.sort(key=lambda t: t[0][0])
        all_mention_texts = [t for _, t, _ in mentions]

        for i, (span_a, text_a, page_a) in enumerate(mentions):
            for j_off, (span_b, text_b, page_b) in enumerate(mentions[i + 1:], start=i + 1):
                # Same page → not a bridge
                if (page_a.get("filename"), page_a.get("source")) == (
                    page_b.get("filename"), page_b.get("source"),
                ):
                    continue
                # Other mentions in this sentence don't count as connectors
                others = [t for k, t in enumerate(all_mention_texts)
                          if k != i and k != j_off]
                if not _has_connector(sentence, span_a, span_b, others):
                    continue
                key = (
                    page_a.get("filename", ""),
                    page_a.get("source", ""),
                    page_b.get("filename", ""),
                    page_b.get("source", ""),
                )
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                out.append({
                    "subject_file": page_a.get("filename", ""),
                    "subject_source": page_a.get("source", ""),
                    "object_file": page_b.get("filename", ""),
                    "object_source": page_b.get("source", ""),
                    "evidence": sentence,
                    "subject_match": text_a,
                    "object_match": text_b,
                })

    return out
