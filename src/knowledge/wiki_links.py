"""Wiki-link parsing for the LLM Wiki Pattern.

Two link forms are recognized:

  1. Obsidian-style wiki links:
       [[page-name]]
       [[page-name|display text]]
       [[page-name#section]]
       [[page-name#section|display text]]

  2. Standard markdown links to local .md files (with optional fragment):
       [display text](page-name.md)
       [display text](page-name.md#section)
       [display text](path/to/page-name.md#section)

External links (http://, mailto:, image embeds with `!`, code spans, fenced
code blocks) are ignored.

Each parsed link returns:
  {
    "raw":       original matched substring
    "target":    canonicalised target filename (always ends in .md)
    "anchor":    optional heading anchor (after #) or "" when absent
    "display":   display text the agent wrote
    "kind":      "wiki" | "markdown"
    "start":     character offset of the match in the source body
  }

Resolution against the knowledge graph happens in `resolve_link()`, which
takes the parsed link dict + a list of indexed (filename, source) pairs
and returns the best-matching filename + source, or None when ambiguous /
unknown. P2.3 will surface link_text + target in graph rendering.
"""

from __future__ import annotations

import re
from pathlib import Path

# Match [[wiki-link]] but NOT image embeds ![[file.png]]
_WIKI_LINK_RE = re.compile(
    r"(?<!\!)\[\[([^\[\]\n|#]+)(?:#([^\[\]\n|]+))?(?:\|([^\[\]\n]+))?\]\]"
)

# Match [text](url) but NOT images ![text](url)
_MARKDOWN_LINK_RE = re.compile(
    r"(?<!\!)\[([^\[\]\n]+)\]\(([^()\n\s]+)(?:\s+\"[^\"]*\")?\)"
)

# Strip fenced code blocks and inline code so we don't pick up links inside
# of code samples.
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def _strip_code(body: str) -> str:
    """Replace code regions with whitespace of equal length so character
    offsets in the cleaned text still align with the original body."""
    def _blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())

    body = _FENCED_CODE_RE.sub(_blank, body)
    body = _INLINE_CODE_RE.sub(_blank, body)
    return body


def _normalize_target(target: str) -> str:
    """Canonicalise a link target into a .md filename.

    - Strips leading './' and '/'
    - Trims whitespace
    - Appends '.md' when missing
    - Returns "" when the target is empty / external-looking
    """
    t = target.strip()
    if not t:
        return ""
    # External links — skip
    if t.startswith(("http://", "https://", "mailto:", "ftp://", "tel:")):
        return ""
    # Anchor-only links (#section) — caller handles separately
    if t.startswith("#"):
        return ""
    if t.startswith("./"):
        t = t[2:]
    t = t.lstrip("/")
    if not t:
        return ""
    if not t.lower().endswith(".md"):
        t = f"{t}.md"
    return t


def parse_links(body: str) -> list[dict]:
    """Parse all wiki-style and markdown links from a markdown body.

    Returns a list ordered by character offset. Code regions are stripped
    before matching so links inside code samples don't pollute the graph.
    """
    if not body:
        return []
    cleaned = _strip_code(body)
    out: list[dict] = []

    for m in _WIKI_LINK_RE.finditer(cleaned):
        target = m.group(1).strip()
        anchor = (m.group(2) or "").strip()
        display = (m.group(3) or target).strip()
        norm = _normalize_target(target)
        if not norm:
            continue
        out.append({
            "raw": m.group(0),
            "target": norm,
            "anchor": anchor,
            "display": display,
            "kind": "wiki",
            "start": m.start(),
        })

    for m in _MARKDOWN_LINK_RE.finditer(cleaned):
        display = m.group(1).strip()
        url = m.group(2).strip()
        # Split off optional anchor
        if "#" in url:
            target_part, anchor = url.split("#", 1)
        else:
            target_part, anchor = url, ""
        norm = _normalize_target(target_part)
        if not norm:
            continue
        out.append({
            "raw": m.group(0),
            "target": norm,
            "anchor": anchor.strip(),
            "display": display,
            "kind": "markdown",
            "start": m.start(),
        })

    out.sort(key=lambda d: d["start"])
    return out


def resolve_link(
    link: dict, indexed_files: list[dict]
) -> tuple[str, str] | None:
    """Resolve a parsed link to (filename, source) against the index.

    Resolution order — first unambiguous hit wins:
      1. Exact path match (treating link target as relative path)
      2. Basename-only match — e.g. `[[cortisol]]` resolves to
         `wiki/cortisol.md` when there's exactly one cortisol.md indexed
      3. Path-suffix match — e.g. `wiki/cortisol.md` matches a file
         indexed as `wiki/cortisol.md`

    When more than one candidate matches, returns None — wiki link
    resolution refuses to silently pick a winner so the lint pass can
    flag the ambiguity later.
    """
    target = (link.get("target") or "").lower()
    if not target or not indexed_files:
        return None

    exact = [
        c for c in indexed_files
        if (c.get("filename") or "").lower() == target
    ]
    if len(exact) == 1:
        return exact[0]["filename"], exact[0].get("source", "knowledge")
    if len(exact) > 1:
        return None

    target_base = Path(target).name
    same_base = [
        c for c in indexed_files
        if Path(c.get("filename", "")).name.lower() == target_base
    ]
    if len(same_base) == 1:
        return same_base[0]["filename"], same_base[0].get("source", "knowledge")
    if len(same_base) > 1:
        return None

    suffix = [
        c for c in indexed_files
        if (c.get("filename") or "").lower().endswith("/" + target)
    ]
    if len(suffix) == 1:
        return suffix[0]["filename"], suffix[0].get("source", "knowledge")

    return None


def normalize_anchor(anchor: str) -> str:
    """Normalise an anchor for case-insensitive heading matching.

    Lowercase, strip, collapse whitespace and dashes. Used by the graph
    builder to match `[[stoics#marcus-aurelius]]` against an indexed
    heading like `Stoics > Marcus Aurelius`.
    """
    if not anchor:
        return ""
    a = anchor.strip().lower()
    a = re.sub(r"[\s_-]+", "-", a)
    return a.strip("-")
