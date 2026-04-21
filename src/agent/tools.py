"""Agent tools for knowledge base operations, file access, and shell execution.

KB Tools implement the token-shopping pattern: the agent sees a heading tree
with token costs, then selectively loads only the sections it needs. Budget
enforcement prevents overloading the context window.
"""

import difflib
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from agent.tokenizer import (
    count_tokens,
    estimate_tokens,
    slice_tokens,
    truncate_at_sentence_boundary,
    truncate_to_tokens,
)
from knowledge.graph import EdgeType, NodeType
from debug_log import get_logger, log_event
import subprocess
from knowledge.chunker import chunk_file
from agent import kb_paths


def _normalize_heading(heading: str) -> str:
    """Normalize a heading for fuzzy matching: lowercase, hyphenate spaces, strip specials."""
    h = heading.lower().strip()
    h = re.sub(r'[^a-z0-9]+', '-', h)
    h = h.strip('-')
    return h


def _format_edge_provenance(edge) -> str:
    """P2.3: render WHY two chunks are connected.

    For REFERENCES edges (wiki-link / markdown-link / prose bridge), surface
    the link_text + link_kind so the agent sees exactly what phrase made
    the connection.

    For SIMILAR edges (D3), surface ``intra_rank``/``intra_total`` from the
    edge attributes when present, e.g. " — rank 1/5 in file". Lets the
    agent read flat 0.78-0.87 cosine spreads as ordinal ranks instead of
    guessing whether 0.84 is signal or floor.

    Falls back to the freeform `evidence` string for other edge types.

    Returns a string like " — via 'link text' [wiki]", " — rank 1/5 in
    file", " — evidence text", or "" when no provenance is available.
    """
    attrs = getattr(edge, "attributes", None) or {}
    link_text = attrs.get("link_text") or ""
    link_kind = attrs.get("link_kind") or ""
    target_anchor = attrs.get("target_anchor") or ""

    if link_text:
        bits = [f"via '{link_text}'"]
        if link_kind:
            bits.append(f"[{link_kind}]")
        if target_anchor:
            bits.append(f"#{target_anchor}")
        out = " " + " ".join(bits)
        # For prose bridges include a snippet of the originating sentence.
        evidence = (getattr(edge, "evidence", "") or "").strip()
        if link_kind == "prose" and evidence:
            snippet = evidence if len(evidence) <= 140 else evidence[:137] + "..."
            out += f' — "{snippet}"'
        return f" —{out}"

    # D3: intra-file SIMILAR rank takes precedence over the generic
    # "semantic similarity (X)" evidence string — the rank carries more
    # signal than the score alone.
    intra_rank = attrs.get("intra_rank")
    intra_total = attrs.get("intra_total")
    if isinstance(intra_rank, int) and isinstance(intra_total, int) and intra_total > 0:
        return f" — rank {intra_rank}/{intra_total} in file"

    evidence = (getattr(edge, "evidence", "") or "").strip()
    if evidence:
        snippet = evidence if len(evidence) <= 140 else evidence[:137] + "..."
        return f" — {snippet}"
    return ""


def _split_combined_path(filename: str, heading: str) -> tuple[str, str]:
    """A4: when heading is empty and filename contains ' > ', split it.

    Left side becomes the file, right side the full heading path.
    Example: 'foo.md > section > subsection' -> ('foo.md', 'section > subsection')
    """
    if heading or " > " not in filename:
        return filename, heading
    left, _, right = filename.partition(" > ")
    return left.strip(), right.strip()


def _filename_match(node, want_relpath: str, want_source: Optional[str]) -> bool:
    """Compare a node's stored filename / source against a parsed canonical
    path. ``want_source`` may be ``None`` when the user only gave a bare relpath
    (we then fall back to relpath-only matching across all sources)."""
    if (node.filename or "").lower() != want_relpath.lower():
        return False
    if want_source is None:
        return True
    return (node.attributes or {}).get("source") == want_source


def _rank_candidates(
    candidates: list,
    needle: str,
    want_source: Optional[str],
    query: str = "",
    kb_index=None,
) -> list:
    """Rank ambiguous resolver candidates so the most plausible suggestion
    appears first in the disambiguation list.

    Ranking priority (stable sort, descending — best first):

      1. ``query=`` cosine rerank — when ``query`` is non-empty AND a KB
         index with an embedding fn is available, score each candidate by
         cosine similarity of ``embed(query)`` vs ``embed(heading + summary)``
         in a single batch call. Failures fall through silently to lexical
         ranking; the cosine score becomes the dominant sort key.
      2. Exact-slug match on the leaf segment of the heading (the part after
         the last `` > ``) against the normalised needle.
      3. Filename-locality — candidates whose ``attributes['source']``
         matches ``want_source`` rank above candidates from other tiers.
      4. Substring position — earlier substring hits in the heading score
         higher than later ones.

    Returns a fresh list; does not mutate the input.
    """
    if not candidates:
        return []
    norm_needle = _normalize_heading(needle)

    cosine_scores: dict[str, float] = {}
    if query and kb_index is not None and getattr(kb_index, "_embedding_fn", None):
        try:
            texts = [(query or "")[:8000]]
            for n in candidates:
                blob = (n.heading or "")
                if n.summary:
                    blob = (blob + " — " + n.summary)[:8000]
                texts.append(blob)
            vecs = kb_index._embedding_fn(texts)
            if vecs and len(vecs) == len(candidates) + 1:
                q_vec = list(vecs[0]) if vecs[0] is not None else None
                if q_vec:
                    for n, v in zip(candidates, vecs[1:]):
                        if v is None:
                            continue
                        cosine_scores[n.id] = KBTools._cosine(q_vec, list(v))
        except Exception:
            cosine_scores = {}

    def _key(n):
        heading_l = (n.heading or "").lower()
        leaf = heading_l.split(" > ")[-1] if " > " in heading_l else heading_l
        norm_leaf = _normalize_heading(leaf)
        exact_leaf = 1 if (norm_needle and norm_needle == norm_leaf) else 0
        src = (n.attributes or {}).get("source")
        local = 1 if (want_source and src == want_source) else 0
        substr_pos = heading_l.find(needle.lower()) if needle else -1
        substr_score = (1.0 / (1 + substr_pos)) if substr_pos >= 0 else 0.0
        cos = cosine_scores.get(n.id, 0.0)
        # Sort descending — negate every component.
        return (-cos, -exact_leaf, -local, -substr_score)

    return sorted(candidates, key=_key)


def _resolve_chunk_nodes(
    graph,
    filename: str,
    heading: str,
    caller: str = "graph_neighbors",
    query: str = "",
    kb_index=None,
) -> tuple[list, Optional[str]]:
    """Resolve (filename, heading) to chunk nodes using the locked order:

    1. A4 split — if heading is empty and filename contains ' > ', split it.
    2. Exact match — n.filename == filename AND normalized headings equal.
    3. Substring match — n.filename == filename AND normalized heading is substring.
    4. A3 cross-file heading fallback — only when heading != '' AND steps 1-3
       returned zero nodes. Searches all files for matching heading.

    Filename input may be canonical (``canon:foo.md`` / ``knowledge:wiki/x.md``)
    or any of the legacy forms (``canon/foo.md``, ``wiki/x.md``, bare
    ``foo.md``). All forms are normalised through ``kb_paths.parse`` and the
    optional ``source`` is then used as an extra filter against
    ``node.attributes['source']``.

    ``caller`` is the tool name to render in disambiguation suggestions
    (e.g. ``describe_node`` so the model copy-pastes back into the same
    tool, not always ``graph_neighbors``).

    ``query`` (optional) and ``kb_index`` (optional) enable a cosine
    rerank of the candidate list when more than one heading matches —
    see ``_rank_candidates`` for the scoring rules.

    Returns (nodes, disambiguation_message). When the cross-file fallback finds
    2-10 candidates, returns ([], message) with a "did you mean" list. When >10,
    returns ([], message) telling the model to narrow.
    """
    filename, heading = _split_combined_path(filename, heading)

    # ---- Canonical-path normalisation ------------------------------------
    # Accept canonical, legacy, and bare forms. ``want_source`` is None when
    # the caller didn't pin a tier (bare ``foo.md`` or unknown segment), in
    # which case we keep the legacy "match any source" behaviour.
    want_source: Optional[str] = None
    want_relpath: str = filename
    if filename:
        try:
            want_source, want_relpath = kb_paths.parse(filename)
        except kb_paths.KBPathError:
            want_source = None
            want_relpath = filename
        # Bare relpath without a recognised tier prefix — drop the source pin
        # so the resolver still finds files indexed under either source.
        if want_source == "knowledge" and not (
            want_relpath.startswith("wiki/") or want_relpath.startswith("raw/")
        ) and ":" not in filename and not filename.startswith("knowledge"):
            want_source = None

    chunk_nodes = [n for n in graph.nodes.values() if n.node_type == NodeType.CHUNK]

    if heading:
        norm_heading = _normalize_heading(heading)
        # Step 2: exact match
        exact = [n for n in chunk_nodes
                 if _filename_match(n, want_relpath, want_source)
                 and _normalize_heading(n.heading) == norm_heading]
        if exact:
            return exact, None
        # Step 3: substring match (filename equal + normalized heading substring)
        substr = [n for n in chunk_nodes
                  if _filename_match(n, want_relpath, want_source)
                  and (norm_heading in _normalize_heading(n.heading)
                       or heading.lower() in n.heading.lower())]
        if substr:
            return substr, None
        # Step 4: A3 cross-file heading fallback. Treat the original `filename`
        # input as a possible heading too (handles graph_neighbors("marcus-aurelius")).
        candidates: list = []
        seen_keys: set = set()
        for needle in (heading, filename):
            norm_needle = _normalize_heading(needle)
            if not norm_needle:
                continue
            for n in chunk_nodes:
                norm_n = _normalize_heading(n.heading)
                # Match the LEAF heading first (highest precision), then full path
                leaf = norm_n.split("-")[-1] if "-" in norm_n else norm_n
                last_segment = n.heading.split(" > ")[-1] if " > " in n.heading else n.heading
                norm_last = _normalize_heading(last_segment)
                if (norm_needle == norm_last
                        or norm_needle in norm_n
                        or needle.lower() in n.heading.lower()):
                    key = (n.filename, n.heading)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        candidates.append(n)
            if candidates:
                break
        if not candidates:
            return [], None
        if len(candidates) == 1:
            return candidates, None
        if len(candidates) <= 10:
            ranked = _rank_candidates(
                candidates, heading or filename, want_source, query, kb_index
            )
            lines = [
                f"Ambiguous: '{filename}' / '{heading}' matched {len(ranked)} sections. Did you mean:"
            ]
            for n in ranked:
                src = (n.attributes or {}).get("source", "knowledge")
                cano = kb_paths.to_canonical(src, n.filename or "")
                lines.append(f"  {caller}(filename='{cano}', heading='{n.heading}')")
            return [], "\n".join(lines)
        return [], (
            f"Ambiguous: '{filename}' / '{heading}' matched {len(candidates)} sections. "
            f"Narrow your query (pass a more specific heading or include the filename)."
        )

    # No heading — try filename-as-file first.
    nodes = [n for n in chunk_nodes
             if _filename_match(n, want_relpath, want_source)]
    if nodes:
        return nodes, None

    # A3 fallback: treat the filename input as a heading and search across files.
    norm_needle = _normalize_heading(filename)
    if not norm_needle:
        return [], None
    candidates: list = []
    seen_keys: set = set()
    for n in chunk_nodes:
        last_segment = n.heading.split(" > ")[-1] if " > " in n.heading else n.heading
        norm_last = _normalize_heading(last_segment)
        norm_n = _normalize_heading(n.heading)
        if (norm_needle == norm_last
                or norm_needle in norm_n
                or filename.lower() in n.heading.lower()):
            key = (n.filename, n.heading)
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append(n)
    if not candidates:
        return [], None
    if len(candidates) == 1:
        return candidates, None
    if len(candidates) <= 10:
        ranked = _rank_candidates(
            candidates, filename, want_source, query, kb_index
        )
        lines = [
            f"Ambiguous: '{filename}' matched {len(ranked)} sections. Did you mean:"
        ]
        for n in ranked:
            src = (n.attributes or {}).get("source", "knowledge")
            cano = kb_paths.to_canonical(src, n.filename or "")
            lines.append(f"  {caller}(filename='{cano}', heading='{n.heading}')")
        return [], "\n".join(lines)
    return [], (
        f"Ambiguous: '{filename}' matched {len(candidates)} sections. "
        f"Narrow your query (pass a more specific heading or include the filename)."
    )


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from content. Returns (metadata, body)."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        import yaml
        meta = yaml.safe_load(parts[1]) or {}
        return meta, parts[2].strip()
    except Exception:
        return {}, content


def _flatten_alias_input(raw) -> list[str]:
    """Flatten alias arguments; do **not** split a single string on commas."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, (list, tuple)):
        acc: list[str] = []
        for item in raw:
            acc.extend(_flatten_alias_input(item))
        return acc
    s = str(raw).strip()
    return [s] if s else []


def _flatten_tag_input(raw) -> list[str]:
    """Recursively flatten tag tool arguments into plain strings.

    Models sometimes pass ``[["a", "b"]]`` which would stringify to one bogus
    tag; this walks lists/tuples and splits comma-separated strings so
    ``_build_frontmatter`` always sees a flat tag list.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        # A pasted Python/JSON list repr must stay one token so validation can
        # reject it — splitting on comma would shred ``['a', 'b']`` into bogus
        # pseudo-tags.
        if (
            len(s) >= 2
            and s[0] == "["
            and s[-1] == "]"
            and ("'" in s or '"' in s)
        ):
            return [s]
        return [p.strip() for p in s.split(",") if p.strip()]
    if isinstance(raw, (list, tuple)):
        acc: list[str] = []
        for item in raw:
            acc.extend(_flatten_tag_input(item))
        return acc
    s = str(raw).strip()
    return [s] if s else []


def _normalize_tags(tags) -> str:
    """Normalize tags to a comma-separated string regardless of input shape.

    Native Ollama tool calls deliver list-typed JSON arguments straight from
    the model, so this function accepts both a list (preferred new shape) and
    a comma-separated string (legacy shape kept for backward compatibility).
    Nested lists are flattened and deduped case-insensitively while keeping
    the first-seen casing.
    """
    flat = _flatten_tag_input(tags)
    if not flat:
        return ""
    seen: set[str] = set()
    deduped: list[str] = []
    for t in flat:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(t)
    return ",".join(deduped)


def _normalize_frontmatter_date(val) -> str:
    """Return ``YYYY-MM-DD`` for frontmatter ``created`` / ``updated``."""
    if val is None:
        return ""
    if isinstance(val, datetime):
        return val.date().isoformat()
    if isinstance(val, date):
        return val.isoformat()
    s = str(val).strip()
    if len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
        return s[:10]
    return s


def _render_yaml_block_string_list(key: str, items: list[str]) -> list[str]:
    """Render ``key`` as a YAML block list of scalars (Obsidian-safe)."""
    if not items:
        return []
    lines = [f"{key}:"]
    for item in items:
        raw = str(item).strip()
        if not raw:
            continue
        need_quote = (
            raw[0] in "'\""
            or any(c in raw for c in ':{}[]#,&*|>!%')
            or " " in raw
            or "\n" in raw
            or raw.startswith("-")
        )
        if need_quote:
            esc = raw.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'  - "{esc}"')
        else:
            lines.append(f"  - {raw}")
    return lines


def _tags_save_validation_error(raw_tags) -> str | None:
    """Refuse writes when the model sent a non-empty tag payload that
    flattened to nothing (e.g. ``[[[]]]``), or left a serialized list blob."""
    if raw_tags in (None, "", []):
        return None
    flat = _flatten_tag_input(raw_tags)
    if not flat:
        return (
            "save_knowledge: refusing to write — `tags` flattened to empty "
            "(nested lists / wrong JSON shape). Pass a flat array of strings, "
            "e.g. [\"philosophy\", \"neuroscience\"], or a comma-separated string."
        )
    for t in flat:
        u = t.strip()
        if u.startswith("[") and u.endswith("]") and ("'" in u or '"' in u):
            return (
                "save_knowledge: refusing to write — tag looks like a serialized "
                f"Python/JSON list: {t[:80]!r}. Pass plain tag strings only."
            )
    return None


def _wiki_to_canon_markdown_href(
    kb_dir: Path,
    canon_dir: Path,
    wiki_output_relpath: str,
    canon_relpath: str,
    anchor: str = "",
) -> str:
    """Filesystem-relative ``../../canon/...`` href from a wiki page to canon."""
    wiki_path = (kb_dir / wiki_output_relpath).resolve()
    canon_path = (canon_dir / canon_relpath).resolve()
    rel = os.path.relpath(canon_path, start=wiki_path.parent)
    href = rel.replace("\\", "/")
    if anchor:
        frag = anchor.strip().lstrip("#")
        if frag:
            href = f"{href}#{frag}"
    return href


_MARKDOWN_CANONICAL_TARGET_RE = re.compile(
    r"\[[^\]]+\]\(([^)]+)\)",
)


def _build_frontmatter(
    tags: str = "",
    date_created: str = "",
    existing_meta: dict | None = None,
    aliases: list[str] | None = None,
    source: str = "knowledge",
    tier: str = "wiki",
) -> str:
    """Build Obsidian-vault-valid YAML frontmatter.

    Emits ``aliases``, ``tags`` (YAML block lists), ``created``, ``updated``
    (date-only ``YYYY-MM-DD``), ``source``, and ``tier``. Reads legacy
    ``date-created`` / ``last-modified`` from ``existing_meta`` when upgrading
    older files but does **not** write those keys back (single date scheme).

    ``created`` is set once on first write and preserved across edits;
    ``updated`` is bumped to today's date on every save.
    """
    now_date = datetime.now().strftime("%Y-%m-%d")
    existing_meta = existing_meta or {}

    # `created` must survive across edits. Prefer `created`, then legacy keys.
    created_raw = (
        existing_meta.get("created")
        or existing_meta.get("date-created")
        or date_created
        or now_date
    )
    created = _normalize_frontmatter_date(created_raw)

    # Carry through aliases, source, tier, and project from any existing
    # frontmatter so manual edits in Obsidian don't get clobbered.
    existing_aliases = existing_meta.get("aliases")
    if isinstance(existing_aliases, str):
        existing_aliases = [existing_aliases]
    elif not isinstance(existing_aliases, list):
        existing_aliases = []
    merged_aliases: list[str] = []
    seen_a = set()
    for a in _flatten_alias_input(aliases) + _flatten_alias_input(existing_aliases):
        s = str(a).strip()
        if s and s.lower() not in seen_a:
            seen_a.add(s.lower())
            merged_aliases.append(s)

    final_source = existing_meta.get("source") or source
    final_tier = existing_meta.get("tier") or tier
    project = existing_meta.get("project")

    tag_list = _flatten_tag_input(tags)
    existing_tags = existing_meta.get("tags")
    for t in _flatten_tag_input(existing_tags):
        if t.lower() not in {x.lower() for x in tag_list}:
            tag_list.append(t)

    lines = ["---"]
    lines.extend(_render_yaml_block_string_list("aliases", merged_aliases))
    lines.extend(_render_yaml_block_string_list("tags", tag_list))
    lines.append(f"created: {created}")
    lines.append(f"updated: {now_date}")
    lines.append(f"source: {final_source}")
    lines.append(f"tier: {final_tier}")
    if project:
        lines.append(f"project: {project}")
    lines.append("---")
    return "\n".join(lines)


def _build_toc(content: str) -> str:
    """Generate a table of contents from H2 headings."""
    h2s = re.findall(r"^##\s+(.+)$", content, re.MULTILINE)
    if not h2s:
        return ""
    lines = []
    for h in h2s:
        anchor = _normalize_heading(h)
        lines.append(f"- [{h}](#{anchor})")
    return "\n".join(lines)


def _ensure_section_dividers(content: str) -> str:
    """Insert --- dividers before H2 headings that don't already have them."""
    lines = content.split("\n")
    result = []
    for i, line in enumerate(lines):
        if line.startswith("## ") and i > 0:
            # Check if previous non-empty line is already ---
            prev_lines = [l.strip() for l in result[-2:] if l.strip()]
            if prev_lines and prev_lines[-1] != "---":
                result.append("")
                result.append("---")
        result.append(line)
    return "\n".join(result)


# C1: convert in-vault markdown links to Obsidian wiki-links so the saved
# pages render natively inside an Obsidian vault. Skips canon paths (canon
# is read-only and not part of the writable wiki vault), external URLs,
# anchors, mailto:, image embeds, and code-fenced regions.
_MARKDOWN_LINK_RE = re.compile(r"(!?)\[([^\]]+)\]\(([^)]+)\)")
_VAULT_LINK_PREFIXES = (
    "wiki/",
    "memory/",
    "knowledge/wiki/",
    "knowledge/memory/",
)


def _convert_markdown_links_to_wiki(content: str) -> str:
    """Rewrite ``[Text](wiki/page.md#anchor)`` as ``[[page#anchor|Text]]``.

    Only touches links that target the writable wiki/memory tiers — canon
    links (read-only ground truth) stay as plain markdown so the citation
    semantics are preserved.

    Code fences and inline code spans are skipped so example markdown in
    documentation pages survives intact.
    """
    if not content or "[" not in content:
        return content

    # Split on triple-backtick code fences so we never rewrite inside one.
    out: list[str] = []
    in_fence = False
    for line in content.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        out.append(_convert_links_in_line(line))
    return "\n".join(out)


def _convert_links_in_line(line: str) -> str:
    # Walk inline code spans manually so we don't touch links inside them.
    pieces: list[str] = []
    i = 0
    in_code = False
    chunk_start = 0
    while i < len(line):
        ch = line[i]
        if ch == "`":
            pieces.append(
                _convert_links_in_text(line[chunk_start:i])
                if not in_code
                else line[chunk_start:i]
            )
            pieces.append("`")
            in_code = not in_code
            chunk_start = i + 1
        i += 1
    tail = line[chunk_start:]
    pieces.append(_convert_links_in_text(tail) if not in_code else tail)
    return "".join(pieces)


def _convert_links_in_text(text: str) -> str:
    def repl(match: re.Match) -> str:
        bang = match.group(1)
        link_text = match.group(2)
        target = match.group(3).strip()
        # Image embed -> leave alone
        if bang:
            return match.group(0)
        # External / scheme / mailto / anchor / canon -> leave alone
        lower = target.lower()
        if (
            lower.startswith(("http://", "https://", "mailto:", "ftp://", "#"))
            or lower.startswith("canon/")
            or lower.startswith("canon:")
        ):
            return match.group(0)
        if not any(
            lower.startswith(p) for p in _VAULT_LINK_PREFIXES
        ):
            return match.group(0)

        # Strip leading 'knowledge/' so 'knowledge/wiki/x.md' and 'wiki/x.md'
        # both become 'wiki/x.md' for slug derivation.
        rel = target
        if rel.startswith("knowledge/"):
            rel = rel[len("knowledge/"):]

        anchor = ""
        if "#" in rel:
            rel, anchor = rel.split("#", 1)
        slug = Path(rel).stem
        if not slug:
            return match.group(0)
        body = slug
        if anchor:
            body = f"{slug}#{anchor}"
        # Preserve display text whenever it differs from the slug at all
        # (Obsidian wiki-link rendering is case-sensitive, so 'Cortisol' vs
        # 'cortisol' must keep the alias even though they hit the same page).
        if link_text.strip() != slug:
            return f"[[{body}|{link_text}]]"
        return f"[[{body}]]"

    return _MARKDOWN_LINK_RE.sub(repl, text)


def _build_file_content(
    filename: str,
    content: str,
    tags: str = "",
    aliases: list[str] | None = None,
    source: str = "knowledge",
    tier: str = "wiki",
) -> str:
    """Assemble a complete KB file with frontmatter, H1, TOC, and dividers.

    The agent provides raw H2+ content. This function wraps it with:
    - Obsidian-vault-valid YAML frontmatter (aliases, tags as block lists;
      ``created`` / ``updated`` as ``YYYY-MM-DD``; ``source``, ``tier``)
    - H1 heading from filename
    - Auto-generated TOC from H2 headings
    - Section dividers before H2 headings
    - In-vault ``[Text](wiki/x.md)`` links rewritten to ``[[x|Text]]``
    """
    stem = Path(filename).stem
    now = datetime.now().strftime("%Y-%m-%d")

    existing_meta = {}
    if content.startswith("---"):
        existing_meta, content = _parse_frontmatter(content)

    content = _convert_markdown_links_to_wiki(content)
    content = _ensure_section_dividers(content)

    toc = _build_toc(content)

    fm = _build_frontmatter(
        tags=tags,
        date_created=now,
        existing_meta=existing_meta,
        aliases=aliases,
        source=source,
        tier=tier,
    )

    parts = [fm, "", f"# {stem}", ""]
    if toc:
        parts.append(toc)
        parts.append("")
    parts.append(content)
    parts.append("")
    return "\n".join(parts)


def _append_log(kb_dir: Path, action: str, target: str, tags: str = "", detail: str = "") -> None:
    """Append an entry to the knowledge base mutation log."""
    log_path = kb_dir / "log.md"
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"## [{timestamp}] {action} | {target}"]
    if tags:
        parts.append(f"tags: {tags}")
    if detail:
        parts.append(detail)
    entry = " | ".join(parts)

    if not log_path.exists():
        log_path.write_text(f"# Knowledge Base Log\n\n{entry}\n", encoding="utf-8")
    else:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{entry}\n")


def _rebuild_index(kb_dir: Path, canon_dir: Path) -> None:
    """Generate index.md catalog of all KB files.

    Creates two markdown tables: canon (read-only) and knowledge (editable).
    Each table has columns: File, Tags, Summary, Modified.
    """
    import yaml

    def _scan_files(base_dir: Path, source: str) -> list[dict]:
        entries = []
        for md_file in sorted(base_dir.rglob("*.md")):
            if md_file.name in ("index.md", "log.md"):
                continue
            rel = str(md_file.relative_to(base_dir))
            try:
                text = md_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            # Parse frontmatter for tags
            meta = {}
            if text.startswith("---"):
                try:
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        meta = yaml.safe_load(parts[1]) or {}
                except Exception:
                    pass

            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]
            tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)

            # First meaningful line as summary
            summary = ""
            for line in text.split("\n"):
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("---"):
                    continue
                summary = stripped[:120]
                break

            mtime = md_file.stat().st_mtime
            from datetime import datetime
            modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

            entries.append({
                "file": rel,
                "tags": tag_str,
                "summary": summary,
                "modified": modified,
            })
        return entries

    def _make_table(entries: list[dict]) -> str:
        if not entries:
            return "*No files.*\n"
        lines = ["| File | Tags | Summary | Modified |", "|------|------|---------|----------|"]
        for e in entries:
            lines.append(f"| {e['file']} | {e['tags']} | {e['summary']} | {e['modified']} |")
        return "\n".join(lines) + "\n"

    canon_entries = _scan_files(canon_dir, "canon") if canon_dir.exists() else []
    kb_entries = _scan_files(kb_dir, "knowledge") if kb_dir.exists() else []

    index_content = "# Knowledge Base Index\n\n"
    index_content += "## Canon (read-only reference)\n\n"
    index_content += _make_table(canon_entries)
    index_content += "\n## Knowledge (editable)\n\n"
    index_content += _make_table(kb_entries)

    index_path = kb_dir / "index.md"
    index_path.write_text(index_content, encoding="utf-8")
from debug_log import get_logger, log_event

log = logging.getLogger(__name__)
dbg = get_logger("tools")

# -- Retrieval budget state (set per request by streaming loop) --
# These are module-level by design: the streaming loop in web/app.py
# resets them per request via reset_budget(), then updates them per
# tool-loop iteration via set_available_tokens() and set_context_window().
_current_available_tokens: int = 999999
_current_kb_loads: int = 0
_current_tool_tokens_used: int = 0
_current_context_window: int = 32000

# Hard ceiling on number of section loads. Real cap is the token budget
# (see _can_afford_load below); this is a safety net to prevent runaway
# loops when a model ignores refusal messages.
_KB_MAX_LOADS_PER_RESPONSE: int = 15

# Minimum context tokens that must remain free before allowing another load.
_KB_MIN_REMAINING_TOKENS: int = 8000

# Max tokens for a single section load (enforced inside read_knowledge_section).
_KB_FILE_MAX_TOKENS: int = 8000

# Fraction of model context window that tool results are allowed to occupy.
# Above this, new loads are refused so the model has room to actually answer.
_KB_MAX_TOOL_FRACTION: float = 0.5


def _tool_token_cap() -> int:
    """Token cap for accumulated tool results this request."""
    return int(_current_context_window * _KB_MAX_TOOL_FRACTION)


def _can_afford_load() -> tuple[bool, str]:
    """Check whether another section load is allowed under the adaptive budget.

    Returns (allowed, refusal_message). Refusal messages always include both
    the used and the cap so the model can reason about how much more it has.
    """
    if _current_kb_loads >= _KB_MAX_LOADS_PER_RESPONSE:
        return False, (
            f"REFUSED: section-load limit reached "
            f"({_current_kb_loads}/{_KB_MAX_LOADS_PER_RESPONSE} loads this response). "
            f"Respond with what you have OR ask the user a follow-up to continue research."
        )
    if _current_available_tokens < _KB_MIN_REMAINING_TOKENS:
        return False, (
            f"REFUSED: context budget too low to load more content "
            f"({_current_available_tokens:,} tokens remaining, "
            f"{_KB_MIN_REMAINING_TOKENS:,} required). "
            f"Respond with what you have OR ask a follow-up to continue."
        )
    cap = _tool_token_cap()
    if _current_tool_tokens_used >= cap:
        return False, (
            f"REFUSED: tool budget used {_current_tool_tokens_used:,} of "
            f"{cap:,} token cap (50% of {_current_context_window:,} ctx). "
            f"Respond with what you have OR ask a follow-up to continue research."
        )
    return True, ""


def _record_load(tokens_loaded: int) -> None:
    """Record a successful tool load against the adaptive budget."""
    global _current_kb_loads, _current_tool_tokens_used
    _current_kb_loads += 1
    _current_tool_tokens_used += max(0, int(tokens_loaded))

# KB directories
_KNOWLEDGE_DIR = Path("/app/knowledge")
_CANON_DIR = Path("/app/canon")


class ToolResult:
    """Result from a tool execution."""

    def __init__(self, success: bool, output: str, error: Optional[str] = None):
        self.success = success
        self.output = output
        self.error = error

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


class FileTools:
    """File system tools for knowledge base operations."""

    def __init__(self, kb_dir: Path, canon_dir: Path):
        self.kb_dir = kb_dir
        self.canon_dir = canon_dir

    def read_file(self, path: str) -> ToolResult:
        """Read a file from knowledge or canon directory."""
        # Try knowledge first
        file_path = self.kb_dir / path
        if not file_path.exists():
            file_path = self.canon_dir / path

        if not file_path.exists():
            return ToolResult(False, "", f"FILE NOT FOUND: '{path}' does not exist. Do not fabricate content.")

        try:
            content = file_path.read_text()
            return ToolResult(True, content)
        except Exception as e:
            return ToolResult(False, "", str(e))

    def write_file(self, path: str, content: str) -> ToolResult:
        """Write a file to knowledge directory (canon is read-only)."""
        # Prevent writing to canon
        canon_path = self.canon_dir / path
        if canon_path.exists():
            return ToolResult(False, "", f"Cannot write to canon file: {path}")

        file_path = self.kb_dir / path

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            file_path.write_text(content)
            return ToolResult(True, f"Written: {path}")
        except Exception as e:
            return ToolResult(False, "", str(e))

    def list_files(self, folder: str = "knowledge") -> ToolResult:
        """List files in a folder."""
        base_dir = self.kb_dir if folder == "knowledge" else self.canon_dir
        if not base_dir.exists():
            return ToolResult(True, "No files found.")

        files = []
        for f in base_dir.rglob("**/*.md"):
            files.append(str(f.relative_to(base_dir)))

        if not files:
            return ToolResult(True, "No files found.")

        return ToolResult(True, "\n".join(files))


class ShellTools:
    """Safe shell execution tools."""

    # Whitelist of allowed commands
    ALLOWED_COMMANDS = {
        "ls", "cat", "head", "tail", "wc", "grep", "find",
        "echo", "date", "pwd", "whoami",
    }

    def execute(self, cmd: str, timeout: int = 30) -> ToolResult:
        """Execute a shell command (whitelisted only)."""
        # Extract the base command
        base_cmd = cmd.strip().split()[0] if cmd.strip() else ""

        if base_cmd not in self.ALLOWED_COMMANDS:
            return ToolResult(
                False, "",
                f"Command '{base_cmd}' not allowed. Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS))}"
            )

        try:
            import subprocess
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout or result.stderr
            return ToolResult(result.returncode == 0, output)
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(False, "", str(e))


class KBTools:
    """Knowledge base tools with token-shopping budget enforcement.

    The agent sees a heading tree (read_knowledge) with token costs per
    section, then selectively loads only the sections it needs
    (read_knowledge_section). Budget enforcement prevents context overload:
    max 15 section loads per response, min 8000 tokens remaining.
    """

    def __init__(
        self,
        kb_index,
        kb_dir: Path = None,
        canon_dir: Path = None,
        conversation_store=None,
    ):
        self.kb_index = kb_index
        self.kb_dir = kb_dir or _KNOWLEDGE_DIR
        self.canon_dir = canon_dir or _CANON_DIR
        # B2: optional ConversationStore so the agent can query its own past
        # threads. Kept optional so existing tests/integrations don't have to
        # pass it; conversation tools degrade to a clear "not wired" message
        # when absent.
        self.conversation_store = conversation_store
        # Per-turn embeddings cached on the conversation_id key. Kept in
        # memory only; persisted by piggybacking the session JSON file.
        self._conversation_search_active_id: str | None = None

    def list_knowledge(self) -> str:
        """List all knowledge base files with medallion tier badges.

        Includes canon (gold), wiki (silver), and raw (bronze) tiers.
        Output is grouped: canon first, then wiki, then raw — so the agent
        sees the most authoritative material at the top of the list.
        """
        # Lazy import to avoid agent->knowledge package coupling at import time
        from knowledge.index import KBIndex

        rows: list[tuple[str, str, str]] = []  # (tier, rel_path, modified)
        for source, base_dir in [("knowledge", self.kb_dir), ("canon", self.canon_dir)]:
            if not base_dir.exists():
                continue
            for f in sorted(base_dir.rglob("**/*.md")):
                rel = str(f.relative_to(base_dir))
                tier = KBIndex._compute_tier(source, rel)
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d")
                except OSError:
                    mtime = "unknown"
                rows.append((tier, rel, mtime))

        if not rows:
            return "Knowledge base is empty."

        tier_order = {"canon": 0, "wiki": 1, "memory": 2, "raw": 3}
        rows.sort(key=lambda r: (tier_order.get(r[0], 9), r[1]))

        # P0.6: emit canonical <source>:<relpath> form alongside the tier
        # badge so the agent always learns the addressable identifier.
        rows4: list[tuple[str, str, str, str]] = []
        for source, base_dir in [("knowledge", self.kb_dir), ("canon", self.canon_dir)]:
            if not base_dir.exists():
                continue
            for f in sorted(base_dir.rglob("**/*.md")):
                rel = str(f.relative_to(base_dir))
                tier = KBIndex._compute_tier(source, rel)
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d")
                except OSError:
                    mtime = "unknown"
                cano = kb_paths.to_canonical(source, rel)
                rows4.append((tier, cano, rel, mtime))
        rows4.sort(key=lambda r: (tier_order.get(r[0], 9), r[1]))

        lines = []
        current_tier = None
        for tier, cano, _rel, mtime in rows4:
            if tier != current_tier:
                current_tier = tier
                lines.append("")
                lines.append(f"== [{tier}] tier ==")
            lines.append(f"  [{tier}] {cano}  modified: {mtime}")
        return "\n".join(lines).lstrip()

    def _resolve_kb_filename(
        self, filename: str
    ) -> tuple[str, str, Path] | None:
        """Resolve a possibly-loose filename to (rel_path, source, abs_path).

        Resolution waterfall (P1.1):
          1. Exact match in knowledge/
          2. Exact match in canon/
          3. Strip leading 'knowledge/' or 'canon/' prefix and retry
          4. Substring match against indexed filenames
             (basename-exact preferred, then full-path substring;
             only resolves when exactly one candidate matches)

        Fuzzy suggestions live in _suggest_filenames so that not-found
        errors can list near-misses without auto-resolving them.
        Returns None when no unambiguous resolution is possible.
        """
        if not filename or not filename.strip():
            return None

        # P0.6: canonical-form input takes priority. ``canon:foo.md`` /
        # ``knowledge:wiki/x.md`` short-circuits the heuristic waterfall.
        if ":" in filename and not filename.lstrip().startswith(("/", ".")):
            try:
                src, rel = kb_paths.from_canonical(filename.strip())
                base = self.canon_dir if src == "canon" else self.kb_dir
                p = base / rel
                if p.exists() and p.is_file():
                    return rel, src, p
                # Canonical was well-formed but the file is missing — keep
                # going so the not-found message still gets fuzzy suggestions.
            except kb_paths.KBPathError:
                pass

        name = filename.strip().lstrip("/").replace("\\", "/")
        if ".." in Path(name).parts:
            return None

        p = self.kb_dir / name
        if p.exists() and p.is_file():
            return name, "knowledge", p
        p = self.canon_dir / name
        if p.exists() and p.is_file():
            return name, "canon", p

        if "/" in name:
            head, rest = name.split("/", 1)
            if head in ("knowledge", "canon") and rest:
                return self._resolve_kb_filename(rest)

        if not self.kb_index:
            return None
        try:
            candidates = self.kb_index.list_indexed_filenames()
        except Exception:
            return None
        if not candidates:
            return None

        target_basename = Path(name).name.lower()
        same_base = [
            c for c in candidates
            if Path(c["filename"]).name.lower() == target_basename
        ]
        if len(same_base) == 1:
            c = same_base[0]
            base = self.canon_dir if c["source"] == "canon" else self.kb_dir
            return c["filename"], c["source"], base / c["filename"]

        lower_name = name.lower()
        sub = [c for c in candidates if lower_name in c["filename"].lower()]
        if len(sub) == 1:
            c = sub[0]
            base = self.canon_dir if c["source"] == "canon" else self.kb_dir
            return c["filename"], c["source"], base / c["filename"]

        return None

    @staticmethod
    def _fuzzy_section_suggestions(
        query: str, headings: list[str], limit: int = 3
    ) -> list[str]:
        """Rank section headings by fuzzy similarity to a query.

        P1.4: when the agent asks for a section that doesn't exist, surface
        the closest headings so it can self-correct without scanning the
        whole available list. Conservative cutoff (0.45) so we don't lie
        about matches when the query is wildly off.
        """
        if not query or not headings:
            return []
        q = query.strip().lower()
        scored: list[tuple[float, str]] = []
        for h in headings:
            if not h:
                continue
            full = difflib.SequenceMatcher(None, q, h.lower()).ratio()
            # Also score against the leaf segment (after " > ") so
            # short queries match deep heading paths.
            leaf = h.split(" > ")[-1].lower()
            leaf_score = difflib.SequenceMatcher(None, q, leaf).ratio()
            substr_bonus = 0.2 if q in h.lower() else 0.0
            scored.append((max(full, leaf_score) + substr_bonus, h))
        scored.sort(reverse=True)
        out: list[str] = []
        seen: set[str] = set()
        for score, h in scored:
            if score < 0.45:
                continue
            if h in seen:
                continue
            seen.add(h)
            out.append(h)
            if len(out) >= limit:
                break
        return out

    def _suggest_filenames(self, filename: str, limit: int = 5) -> list[str]:
        """Return up to `limit` near-miss filenames for not-found errors.

        Combines basename and full-path fuzzy matching against indexed
        filenames using difflib. Used by read_knowledge / read_knowledge_section
        to give the agent a corrective hint without auto-resolving (P1.4 hook).
        """
        if not self.kb_index or not filename:
            return []
        try:
            candidates = self.kb_index.list_indexed_filenames()
        except Exception:
            return []
        if not candidates:
            return []
        paths = [c["filename"] for c in candidates]
        target = filename.strip().lower().replace("\\", "/")
        scored: list[tuple[float, str]] = []
        for p in paths:
            ratio_path = difflib.SequenceMatcher(None, target, p.lower()).ratio()
            ratio_base = difflib.SequenceMatcher(
                None, Path(target).name, Path(p).name.lower()
            ).ratio()
            scored.append((max(ratio_path, ratio_base), p))
        scored.sort(reverse=True)
        out: list[str] = []
        seen: set[str] = set()
        for score, p in scored:
            if score < 0.5:
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
            if len(out) >= limit:
                break
        return out

    def read_knowledge(self, filename: str) -> str:
        """Returns the heading tree for a knowledge file -- H1-H5 structure
        with token counts per subtree and section summaries. Never loads
        full content. Use read_knowledge_section to load the specific
        sections you need. This lets you shop for content and control
        exactly how many tokens you consume."""
        resolved = self._resolve_kb_filename(filename)
        if resolved is None:
            suggestions = self._suggest_filenames(filename)
            hint = (
                f" Did you mean: {', '.join(suggestions)}?"
                if suggestions else ""
            )
            return (
                f"FILE NOT FOUND: '{filename}' does not exist in the knowledge "
                f"base.{hint} Do not fabricate content. Tell the user this "
                f"file is not available."
            )
        rel_path, source, abs_path = resolved

        try:
            content = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return (
                f"ERROR: Could not read file '{rel_path}'. "
                f"Do not fabricate content."
            )
        if not content.strip():
            return "(file exists but has no section content)"

        tok_count = estimate_tokens(content)

        if self.kb_index:
            tree_text = self.kb_index.get_heading_tree(rel_path, source)
            if tree_text:
                tree_cost = estimate_tokens(tree_text)
                lines = [
                    f"[{tok_count:,} tokens total]",
                    "",
                    "IMPORTANT: This is a STRUCTURE tree showing headings and token costs. "
                    "It does NOT contain the actual section content. "
                    "Use read_knowledge_section to load specific sections.",
                    "",
                    tree_text,
                    "",
                    f"[Tree cost: ~{tree_cost:,} tokens. "
                    f"Each read_knowledge_section call costs additional tokens -- "
                    f"check subtree sizes above before loading.]",
                ]
                return "\n".join(lines)

        stem = Path(rel_path).stem
        cano = kb_paths.to_canonical(source, rel_path)
        return (
            f"[{tok_count:,} tokens, no heading tree available]\n"
            f"Use read_knowledge_section(\"{cano}\", \"{stem}\") to load."
        )

    def read_knowledge_section(self, filename: str, section: str, offset: int = 0) -> str:
        """Load a specific section from a KB file by heading.

        Use after ``read_knowledge`` to fetch the actual content of a section
        you saw in the heading tree. H1 headings load the full concept block
        (including H2-H5 children). H2-H5 headings extract just that
        subsection from within its parent H1. The section parameter is a
        case-insensitive substring match.

        Every successful load is prefixed with a structured marker so the
        agent can tell whether content was clipped:

        - ``[SECTION: file | X/Y | heading | LOADED N of M tokens (COMPLETE)]``
        - ``[SECTION: file | X/Y | heading | LOADED N of M tokens (TRUNCATED -- offset=N for more)]``

        Refuses when the adaptive budget is exhausted (see _can_afford_load).

        Args:
            filename: Canonical or legacy KB path to the file (e.g.
                ``"knowledge:wiki/foo.md"`` or ``"wiki/foo.md"``).
            section: Heading text to locate. Case-insensitive substring match
                against H1-H5 headings in the file.
            offset: Token offset into the section. Default 0 loads from the
                start. When a section is too large for one load the response
                includes the next offset to call back with.
        """
        # -- Adaptive budget enforcement --
        allowed, refusal = _can_afford_load()
        if not allowed:
            return refusal

        # Defensive: tool args may still arrive as str when called from tests.
        try:
            offset_tokens = max(0, int(offset))
        except (TypeError, ValueError):
            offset_tokens = 0

        # P1.1: resolve filename via the same waterfall used by read_knowledge
        # (exact → strip-prefix → substring → fail with fuzzy hints).
        resolved = self._resolve_kb_filename(filename)
        if resolved is None:
            suggestions = self._suggest_filenames(filename)
            hint = (
                f" Did you mean: {', '.join(suggestions)}?"
                if suggestions else ""
            )
            return (
                f"FILE NOT FOUND: '{filename}' does not exist in the "
                f"knowledge base.{hint} Do not fabricate content. Tell the "
                f"user this file is not available."
            )
        rel_path, source, path = resolved

        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return (
                f"ERROR: Could not read file '{rel_path}'. "
                f"Do not fabricate content."
            )

        # P1.1: prefer index-backed section enumeration when available so the
        # marker's [X/Y] and the "Available sections" hint match what the
        # agent saw in read_knowledge / search_knowledge. Fall back to disk
        # chunking only when the index is unavailable or has no rows for
        # this file (e.g. just-saved page before reindex completes).
        indexed_sections: list[dict] = []
        if self.kb_index:
            try:
                indexed_sections = self.kb_index.list_sections(rel_path, source)
            except Exception:
                indexed_sections = []

        # The natural-boundary chunker (max_tokens=None) is what gives
        # offset/TRUNCATED something meaningful to navigate — section
        # content is not stored cleanly in the index, so we still need
        # this for the actual content slice.
        chunks = chunk_file(content, rel_path, max_tokens=None)
        section_lower = section.lower()
        total_sections = (
            len(indexed_sections) if indexed_sections else len(chunks)
        )

        match = self._find_section(chunks, section_lower)
        if match is None:
            if indexed_sections:
                available = [
                    s.get("heading", "") for s in indexed_sections
                    if s.get("heading", "")
                ]
            else:
                available = [c["heading"] for c in chunks]
            # P1.4: rank near-misses with difflib so the agent gets a
            # corrective suggestion instead of having to scan the whole
            # available list.
            suggestions = self._fuzzy_section_suggestions(section, available)
            hint = (
                f"Did you mean: {', '.join(suggestions)}?\n"
                if suggestions else ""
            )
            available_hint = ", ".join(available) if available else "(none)"
            return (
                f"SECTION NOT FOUND: '{section}' does not exist in {rel_path}.\n"
                f"{hint}"
                f"Available sections: {available_hint}\n"
                f"IMPORTANT: Do NOT fabricate content for this section. "
                f"Tell the user this section was not found and list what IS available."
            )

        chunk_index, heading_display, raw_content = match
        return self._render_section_payload(
            filename=kb_paths.to_canonical(source, rel_path),
            chunk_index=chunk_index,
            total_sections=total_sections,
            heading_display=heading_display,
            raw_content=raw_content,
            offset_tokens=offset_tokens,
        )

    def _find_section(
        self, chunks: list[dict], section_lower: str
    ) -> tuple[int, str, str] | None:
        """Locate a section by H1 then H2-H5 substring match.

        Returns (chunk_index, display_heading, raw_content) or None.
        """
        # Pass 1: H1 concept heading match (case-insensitive substring)
        for chunk in chunks:
            if section_lower in chunk["heading"].lower():
                heading = chunk["heading"]
                display = "preamble" if heading == "(preamble)" else heading
                return chunk.get("chunk_index", 0), display, chunk["content"]

        # Pass 2: H2-H5 subsection extraction
        heading_re = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)
        for chunk in chunks:
            lines = chunk["content"].split("\n")
            for i, line in enumerate(lines):
                m = heading_re.match(line)
                if not m or section_lower not in m.group(2).lower():
                    continue
                level = len(m.group(1))
                extracted = [line]
                for j in range(i + 1, len(lines)):
                    nm = re.match(r"^(#{1,6})\s", lines[j])
                    if nm and len(nm.group(1)) <= level:
                        break
                    extracted.append(lines[j])
                parent_heading = chunk["heading"]
                matched_heading = m.group(2).strip()
                if parent_heading and parent_heading != "(preamble)":
                    full_heading = f"{parent_heading} > {matched_heading}"
                else:
                    full_heading = matched_heading
                return chunk.get("chunk_index", 0), full_heading, "\n".join(extracted).strip()

        return None

    def _render_section_payload(
        self,
        filename: str,
        chunk_index: int,
        total_sections: int,
        heading_display: str,
        raw_content: str,
        offset_tokens: int,
    ) -> str:
        """Apply offset + truncation, attach COMPLETE/TRUNCATED marker, record budget."""
        total_raw_tokens = count_tokens(raw_content)

        if offset_tokens >= total_raw_tokens > 0:
            return (
                f"[SECTION: {filename} | {chunk_index + 1}/{total_sections} | "
                f"{heading_display} | OFFSET {offset_tokens:,} >= total {total_raw_tokens:,} tokens (END OF SECTION)]\n\n"
                f"No more content past offset {offset_tokens:,}. "
                f"Respond with what you already loaded."
            )

        # Slice by token offset
        sliced, _, has_more_after = slice_tokens(
            raw_content, offset_tokens, _KB_FILE_MAX_TOKENS
        )

        # Truncate at sentence boundary so the model never sees mid-word cuts.
        # Char budget is generous (4 chars/token avg * KB_FILE_MAX_TOKENS) so
        # this only triggers on pathological lines without word breaks.
        char_budget = _KB_FILE_MAX_TOKENS * 6
        sliced, char_truncated = truncate_at_sentence_boundary(sliced, char_budget)

        delivered_tokens = count_tokens(sliced)
        is_truncated = has_more_after or char_truncated
        next_offset = offset_tokens + delivered_tokens

        if is_truncated:
            marker = (
                f"[SECTION: {filename} | {chunk_index + 1}/{total_sections} | "
                f"{heading_display} | LOADED {delivered_tokens:,} of "
                f"{total_raw_tokens:,} tokens, offset {offset_tokens:,} (TRUNCATED -- "
                f"call again with offset={next_offset} to continue, or load a sub-heading to narrow)]"
            )
        else:
            marker = (
                f"[SECTION: {filename} | {chunk_index + 1}/{total_sections} | "
                f"{heading_display} | LOADED {delivered_tokens:,} of "
                f"{total_raw_tokens:,} tokens (COMPLETE)]"
            )

        # Record against adaptive budget using the model-facing token estimate.
        _record_load(estimate_tokens(sliced))

        return f"{marker}\n\n{sliced}"

    def search_knowledge(self, query: str) -> str:
        """Search knowledge base by topic. Returns section-level results
        with heading paths, summaries, and position info. Use
        read_knowledge_section to load a specific section's content."""
        if not self.kb_index:
            return "Knowledge base index not available."

        # Empty or whitespace-only query — list all files instead
        if not query or not query.strip():
            return self.list_knowledge()

        # Section-level semantic search
        results = self.kb_index.search(query, top_k=10)

        if not results:
            return f"No matches for '{query}' in knowledge base."

        lines = []
        current_file = None
        for r in results:
            filename = r.get("filename", r.get("path", ""))
            source = r.get("source", "knowledge")
            tier = r.get("tier", "wiki")
            tier_badge = f" [{tier}]"
            heading = r.get("heading", "")
            summary = r.get("summary", "")
            # Show both raw similarity and the tier-weighted score so the
            # agent can distinguish "high similarity, raw tier" from
            # "moderate similarity, canon tier".
            raw_score = r.get("score", 0)
            weighted = r.get("weighted_score", raw_score)
            chunk_idx = r.get("chunk_index", 0) + 1
            section_count = r.get("section_count", 0)
            pos = f" [{chunk_idx}/{section_count}]" if section_count else ""

            # P0.6: emit canonical <source>:<relpath> form so the agent learns
            # the addressable identifier — round-trips into graph_* tools and
            # read_knowledge_section without ambiguity.
            try:
                cano = kb_paths.to_canonical(str(source), str(filename))
            except kb_paths.KBPathError:
                cano = filename

            # File header when file changes
            if cano != current_file:
                current_file = cano
                lines.append(f"== {cano}{tier_badge}")

            score_part = (
                f"(score: {weighted:.2f}"
                + (f", raw sim: {raw_score:.2f}" if abs(weighted - raw_score) > 0.001 else "")
                + ")"
            )
            lines.append(f"  > {heading}{pos} {score_part}")
            if summary:
                lines.append(f"    {summary}")

        lines.append(
            f"\n[{len(results)} sections matched, ranked with canon-boost / "
            f"raw-suppress tier weights. "
            f"Use read_knowledge_section(filename, heading) to load content.]"
        )
        return "\n".join(lines)

    @staticmethod
    def _normalize_wiki_path(filename: str) -> tuple[str | None, str | None]:
        """Validate filename and normalize it to a path under wiki/.

        Returns (normalized_rel_path, error_message). Exactly one of the two
        will be non-None.

        Rules (P0.4 medallion guards):
          - Reject empty / null filenames
          - Reject absolute paths and any path-traversal segments ('..')
          - Reject `raw/` prefix — bronze tier is read-only source material
          - Reject `canon/` prefix — gold tier is read-only canonical content
          - Allow `memory/` prefix as the memory tier (between raw and silver)
            for short-lived notes / journal pages. Memory pages still get
            normal frontmatter and indexing.
          - Force every other successful write under `wiki/` (silver tier).
            Filenames without an explicit `wiki/` or `memory/` prefix are
            silently rewritten so the agent can keep using flat names
            like 'cortisol.md'.
        """
        if not filename or not isinstance(filename, str):
            return (None, "Cannot save: filename is required.")

        norm = filename.replace("\\", "/").strip()
        if not norm:
            return (None, "Cannot save: filename is required.")

        # Reject absolute paths
        if norm.startswith("/"):
            return (None, f"Cannot save: filename '{filename}' must be a relative path.")

        # Reject any '..' segment (path traversal)
        parts = [p for p in norm.split("/") if p]
        if any(p == ".." for p in parts):
            return (None, f"Cannot save: filename '{filename}' contains '..' which is not allowed.")

        # R9: strip a single leading `knowledge/` prefix. The model often
        # carries it over from canonical paths that tool output prints as
        # `knowledge:wiki/foo.md` (colon-form → writes fine) but sometimes
        # slash-form (`knowledge/wiki/foo.md`) — without this strip, the tier
        # check below prepended `wiki/` and files landed in
        # `knowledge/wiki/knowledge/wiki/`. Only one level is stripped so a
        # genuinely nested layout like `wiki/knowledge/knowledge.md` (a wiki
        # page named "knowledge") still works.
        if len(parts) > 1 and parts[0] == "knowledge":
            parts = parts[1:]

        # Tier-based refusals
        if parts[0] == "raw":
            return (None, (
                f"Cannot save to 'raw/': bronze-tier files are read-only "
                f"source material. Compile a wiki page in 'wiki/' instead."
            ))
        if parts[0] == "canon":
            return (None, (
                f"Cannot save to 'canon/': gold-tier files are read-only. "
                f"Use 'wiki/' for agent-authored pages."
            ))

        # wiki/ and memory/ are both valid writable tiers — keep as-is.
        # Anything else is silently rewritten under wiki/ so the agent can
        # keep using flat filenames like 'cortisol.md'.
        if parts[0] not in ("wiki", "memory"):
            parts = ["wiki"] + parts

        return ("/".join(parts), None)

    @staticmethod
    def _sanitize_content_escapes(content: str) -> str:
        """Decode literal backslash escape sequences in agent-authored content.

        P1.2: when content arrives with backslash-n / -t / -r literals (two
        chars each) instead of real whitespace — typically because the model
        emitted a JSON-escaped string and an upstream parser didn't decode
        it — the chunker collapses everything into one heading line. This
        helper restores the structure so saved files chunk correctly.

        Heuristic: only apply decoding when the content has literal `\\n`
        sequences AND has fewer than two real newlines. That way we never
        mangle correctly-formed content (e.g. real markdown that contains a
        backslash followed by an n).
        """
        if not content or "\\" not in content:
            return content
        if content.count("\n") >= 2:
            return content
        if "\\n" not in content and "\\t" not in content and "\\r" not in content:
            return content
        out: list[str] = []
        i = 0
        while i < len(content):
            ch = content[i]
            if ch == "\\" and i + 1 < len(content):
                nxt = content[i + 1]
                if nxt == "n":
                    out.append("\n"); i += 2; continue
                if nxt == "r":
                    out.append("\r"); i += 2; continue
                if nxt == "t":
                    out.append("\t"); i += 2; continue
                if nxt == "\\":
                    out.append("\\"); i += 2; continue
            out.append(ch)
            i += 1
        return "".join(out)

    # ------------------------------------------------------------------
    # B3 — ## Sources block citation validator
    # ------------------------------------------------------------------

    # Matches `conversation:<id>:turn:N` and `conversation:<id>:turn:A-B`.
    # Conversation IDs are UUID-shaped in our store but the regex stays
    # permissive: any non-whitespace run of url-safe chars before the next
    # ``:turn:`` segment counts.
    _CONV_CITE_RE = re.compile(
        r"conversation:([A-Za-z0-9_\-]+):turn:(\d+)(?:-(\d+))?"
    )

    def _validate_conversation_sources(self, content: str) -> str | None:
        """Return an error string when a ## Sources citation can't be verified.

        Walks the body for a ``## Sources`` (or ``# Sources``) heading and
        scans every line below it (until the next heading) for citations of
        the form ``conversation:<id>:turn:N`` or ``conversation:<id>:turn:A-B``.
        Each citation must resolve to an existing conversation and an
        in-range turn index. Anything else aborts the save.

        Returns ``None`` when:
          - there's no Sources block, or
          - all citations resolve, or
          - the conversation store isn't wired (we can't verify what we
            don't have access to; that's the chat-server's problem to fix).
        """
        if not content or not self.conversation_store:
            return None

        lines = content.splitlines()
        in_sources = False
        cite_lines: list[str] = []
        for raw_line in lines:
            stripped = raw_line.strip()
            if stripped.lower().startswith("## sources") or \
                    stripped.lower().startswith("# sources"):
                in_sources = True
                continue
            if in_sources:
                # A new heading ends the Sources block.
                if stripped.startswith("#"):
                    in_sources = False
                    continue
                cite_lines.append(raw_line)
        if not cite_lines:
            return None

        sources_text = "\n".join(cite_lines)
        matches = list(self._CONV_CITE_RE.finditer(sources_text))
        if not matches:
            return None

        # Cache turn counts to avoid hitting disk for repeated citations.
        turn_counts: dict[str, int | None] = {}

        def _turn_count(conv_id: str) -> int | None:
            if conv_id in turn_counts:
                return turn_counts[conv_id]
            try:
                session = self.conversation_store._read_session(conv_id)  # noqa: SLF001
            except Exception:
                turn_counts[conv_id] = None
                return None
            if not session:
                turn_counts[conv_id] = None
                return None
            turns = session.get("turns") or []
            count = len(turns) if turns else 0
            # An empty session file still exists as an empty thread; treat
            # that as "no turns" and let the caller decide.
            turn_counts[conv_id] = count if count > 0 else None
            return turn_counts[conv_id]

        problems: list[str] = []
        for m in matches:
            conv_id = m.group(1)
            a = int(m.group(2))
            b = int(m.group(3)) if m.group(3) is not None else a
            count = _turn_count(conv_id)
            if count is None:
                problems.append(
                    f"  - conversation:{conv_id}:turn:{a}"
                    f"{'-' + str(b) if b != a else ''} "
                    f"-> conversation not found or has no turns"
                )
                continue
            if b < a:
                problems.append(
                    f"  - conversation:{conv_id}:turn:{a}-{b} "
                    f"-> turn range A-B requires B >= A"
                )
                continue
            if a < 0 or b >= count:
                problems.append(
                    f"  - conversation:{conv_id}:turn:{a}"
                    f"{'-' + str(b) if b != a else ''} "
                    f"-> out of range (conversation has {count} turn(s), "
                    f"valid indices 0..{count - 1})"
                )

        if problems:
            return (
                "save_knowledge: refusing to write — the ## Sources block "
                "cites conversation turns that can't be verified:\n"
                + "\n".join(problems)
                + "\nFix or remove the bad citations and call save_knowledge "
                "again. Use search_conversations / read_conversation to "
                "confirm a citation before writing it."
            )
        return None

    # ------------------------------------------------------------------
    # R12 — placeholder-URL defense for saved wiki pages
    # ------------------------------------------------------------------
    #
    # The model sometimes invents URLs that LOOK legitimate
    # (`https://example.com/stoicism-study`, `https://relevant-paper.org/...`)
    # but point at nothing real. Shipping those to disk corrupts the
    # canon/wiki link graph and silently poisons every future reader.
    # We block the write with a specific message so the model can
    # retry with either a real URL, a `[[wiki-link]]`, or no link at all.
    #
    # Scope:
    #   - Syntactic only. NO live HTTP HEAD checks. Network calls
    #     inside a tool dispatch would add latency, fail in offline
    #     dev containers, and still not catch URLs that 200 but don't
    #     answer the thing cited.
    #   - Only `http://` / `https://` URLs in markdown link targets
    #     or bare URL text. `[[wiki-link]]`, relative paths
    #     (`../../canon/...`), and `#anchors` are ignored entirely —
    #     they route through the existing link-conversion layer.

    # Placeholder-domain blocklist. Kept lowercase; matching is on
    # the URL host only so `https://example.com/article/5` and
    # `https://foo.example.com/` both get rejected, but a legit site
    # that happens to contain "example" in a path (e.g. a blog post
    # about an example problem on a real domain) is not caught.
    _URL_PLACEHOLDER_HOSTS = frozenset({
        "example.com", "example.org", "example.net",
        "placeholder.com", "placeholder.org",
        "your-domain.com", "yourdomain.com",
        "sample.com", "sample.org",
        "test.com", "test.org",
        "relevant-study.org", "relevant-paper.org",
        "relevant-source.com", "relevant-source.org",
        "fake.com", "fake.org",
        "domain.com", "domain.org",
        "somewebsite.com", "somesite.com",
    })
    # Hostname substring blocklist for patterns without a fixed domain
    # (e.g. `relevant-paper-4.com`, `placeholder-url.io`). Runs AFTER
    # the exact-host check so unambiguous matches fail fast with the
    # host listed verbatim.
    _URL_PLACEHOLDER_SUBSTRINGS = (
        "placeholder", "relevant-study", "relevant-paper",
        "relevant-source", "your-domain", "yourdomain",
    )
    _URL_RE = re.compile(r"https?://[^\s<>\"\'\)\]}]+", re.IGNORECASE)
    # Strip a trailing md-link close / sentence punctuation that the
    # greedy char class grabbed. Conservative: only trim final chars we
    # know are never legal in a URL host+path.
    _URL_TRAIL_TRIM = ".,;:!?)'\""

    def _extract_url_host(self, url: str) -> str:
        """Return the lowercase hostname of a URL, or empty string on
        parse failure. Uses urllib so we handle userinfo, ports, IPv6
        literals, etc. correctly."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = (parsed.hostname or "").lower()
            return host
        except Exception:
            return ""

    def _validate_urls(self, content: str) -> str | None:
        """Return an error string if `content` contains placeholder URLs.

        Syntactic check only. Scans every `http(s)://...` token in the
        body (anywhere — link targets, bare mentions, fenced code is
        NOT exempted because placeholder URLs in examples still end up
        cited by downstream readers). Rejects:
          - URLs whose host matches `_URL_PLACEHOLDER_HOSTS` exactly
          - URLs whose host contains any substring in
            `_URL_PLACEHOLDER_SUBSTRINGS`

        Returns None on clean content, on content with zero URLs, or
        on content whose URLs are all real (or at least not in the
        blocklist). The model is expected to re-issue the save with
        a `[[wiki-link]]` or a verified URL.
        """
        if not content:
            return None

        bad: list[tuple[str, str, str]] = []  # (raw_url, host, reason)
        seen: set[str] = set()
        for m in self._URL_RE.finditer(content):
            raw = m.group(0).rstrip(self._URL_TRAIL_TRIM)
            if raw in seen:
                continue
            seen.add(raw)
            host = self._extract_url_host(raw)
            if not host:
                continue
            matched_placeholder = False
            # Exact-host match AND subdomain match: `foo.example.com`
            # should refuse the same as bare `example.com` because the
            # apex is a placeholder; the subdomain part doesn't rescue it.
            for placeholder in self._URL_PLACEHOLDER_HOSTS:
                if host == placeholder or host.endswith("." + placeholder):
                    bad.append((raw, host, "placeholder host in blocklist"))
                    matched_placeholder = True
                    break
            if matched_placeholder:
                continue
            for needle in self._URL_PLACEHOLDER_SUBSTRINGS:
                if needle in host:
                    bad.append((raw, host, f"host contains '{needle}'"))
                    break

        if not bad:
            return None

        lines = [
            "save_knowledge: refusing to write — content contains "
            "placeholder URLs that look invented. Replace each with a "
            "real URL you have verified, a [[wiki-link]] to an existing "
            "KB page, or remove the link entirely:",
        ]
        for raw, host, reason in bad:
            lines.append(f"  - {raw}  ({reason})")
        lines.append(
            "\nNever fabricate URLs. If you don't have a verified source, "
            "use a [[wiki-link]] or cite the conversation turn instead."
        )
        return "\n".join(lines)

    def save_knowledge(
        self,
        filename: str,
        content: str,
        tags: str = "",
    ) -> str:
        """Create or update a wiki page in the agent's silver-tier namespace.

        All writes land under knowledge/wiki/. Bare filenames are silently
        normalized — `save_knowledge("cortisol.md", ...)` writes to
        `knowledge/wiki/cortisol.md`. Refuses writes to:
          - canon/ (gold tier — read-only)
          - knowledge/raw/ (bronze tier — read-only source material)
          - paths containing '..' or absolute paths

        Content should be organized as ## sections with text underneath.
        To edit an existing file, read it first with read_knowledge, make
        changes, then save the full updated content. Pass tags as a
        comma-separated string (e.g. "philosophy,stoicism,practice");
        internal callers may also pass a list and it will be normalized.

        Automatically adds frontmatter (``created`` / ``updated`` as dates,
        YAML block ``tags`` / ``aliases``), H1 heading, TOC, and section
        dividers. Also updates index.md and log.md.
        """
        tag_shape_err = _tags_save_validation_error(tags)
        if tag_shape_err:
            return tag_shape_err
        tags = _normalize_tags(tags)
        normalized, err = self._normalize_wiki_path(filename)
        if err:
            return err

        # P1.2: heading-leak defense. If the content arrived with literal
        # backslash-escape sequences instead of real newlines (e.g. tool
        # parser regression, model outputting JSON-escaped strings), the
        # chunker would treat the entire body as one heading. Decode here
        # before write so the file lands clean on disk.
        content = self._sanitize_content_escapes(content)

        # B3: validate any conversation citations in the ## Sources block
        # against the live ConversationStore. Refuse to write if a citation
        # points at a non-existent thread or out-of-range turn so we never
        # ship a wiki page with hallucinated provenance.
        cite_err = self._validate_conversation_sources(content)
        if cite_err:
            return cite_err

        # R12: refuse placeholder URLs (example.com, relevant-study.org,
        # etc.) before they land on disk. Syntactic check only — no HTTP
        # HEAD — so the model can iterate inside the turn without network.
        url_err = self._validate_urls(content)
        if url_err:
            return url_err
        # Even if user explicitly passed wiki/, also block overwriting an
        # existing canon file with the same basename (defense in depth)
        if (self.canon_dir / Path(normalized).name).exists() and "/" not in filename:
            return (
                f"Cannot save '{filename}': a canon file with that name "
                f"already exists. Choose a different filename."
            )

        filename = normalized
        file_path = self.kb_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # C1: derive tier from the normalized path so the frontmatter line
        # matches the medallion location. wiki/ is silver; memory/ is the
        # memory tier; everything else under knowledge/ defaults to wiki.
        norm_lower = filename.lower().replace("\\", "/")
        if norm_lower.startswith("memory/"):
            file_tier = "memory"
        else:
            file_tier = "wiki"

        full_content = _build_file_content(
            filename,
            content,
            tags,
            source="knowledge",
            tier=file_tier,
        )

        # Suppress watcher events for the paths we're about to write — we'll
        # re-index inline below, so the watcher firing on top would just
        # duplicate the work (and on save_knowledge in a chat loop, this
        # cascade was what burned the Gemini quota in the deadlock incident).
        try:
            from agent.watcher import suppress_paths
            suppress_paths([
                file_path,
                self.kb_dir / "log.md",
                self.kb_dir / "index.md",
            ])
        except Exception:
            pass  # watcher module optional (e.g., in tests)

        try:
            file_path.write_text(full_content, encoding="utf-8")
        except Exception as e:
            return f"Error saving file: {e}"

        # Update index.md and log.md
        _append_log(self.kb_dir, action="save", target=filename, tags=tags)
        try:
            _rebuild_index(self.kb_dir, self.canon_dir)
        except Exception:
            pass  # Index rebuild failure should not block save

        # Re-index the file in the KB vector store. Pass llm_summaries=False
        # explicitly — the per-chunk write path in _index_file now preserves
        # any prior LLM summaries by heading, so this is no longer destructive.
        if self.kb_index:
            try:
                self.kb_index._index_file(file_path, filename, source="knowledge")
            except Exception:
                pass  # Index failure should not block save

            # Rebuild graph nodes and edges for the updated file
            try:
                self.kb_index._init_graph_nodes_only()
                self.kb_index._build_graph_edges()
            except Exception:
                pass  # Graph rebuild failure should not block save

        # P3.1: append a "Related" block of similarity-suggested wiki links
        # so the page joins the graph as a connected node, not an island.
        # Uses the just-built graph to find top related pages, replaces any
        # existing auto-generated block in-place (idempotent), then triggers
        # one final reindex so the new wiki-links become real REFERENCES
        # edges via _build_wiki_link_edges.
        try:
            related_block = self._compute_related_block(filename)
            if related_block:
                with_related = self._inject_related_block(full_content, related_block)
                if with_related != full_content:
                    file_path.write_text(with_related, encoding="utf-8")
                    if self.kb_index:
                        try:
                            self.kb_index._index_file(
                                file_path, filename, source="knowledge",
                            )
                            self.kb_index._init_graph_nodes_only()
                            self.kb_index._build_graph_edges()
                            self.kb_index._build_wiki_link_edges()
                        except Exception:
                            pass
        except Exception:
            pass  # Related-block compilation failure should not block save

        return f"Saved: {filename}"

    # ------------------------------------------------------------------
    # P3.1 helpers — related-pages compilation
    # ------------------------------------------------------------------

    _RELATED_BLOCK_START = "<!-- llm-wiki-agent:related-pages start -->"
    _RELATED_BLOCK_END = "<!-- llm-wiki-agent:related-pages end -->"

    def _compute_related_block(self, filename: str, top_n: int = 5) -> str:
        """Compute a markdown 'Related' block of wiki-link suggestions.

        Walks the knowledge graph: for every chunk of `filename`, collect
        cross-file edges (SIMILAR / INTER_FILE / CROSS_DOMAIN), aggregate
        by target filename (sum of weights), drop the file itself, and
        emit the top N as `[[basename]]` wiki links.

        Returns "" when no qualifying neighbors exist (single-file corpus,
        completely orphaned page, or graph not yet built).
        """
        if not self.kb_index or not self.kb_index.graph:
            return ""
        graph = self.kb_index.graph

        # Edge classes that indicate semantic / structural cross-file
        # relevance. Skip PARENT_CHILD (folder hierarchy) and REFERENCES
        # (the agent already wrote those).
        relevant_edges = {
            EdgeType.SIMILAR,
            EdgeType.INTER_FILE,
            EdgeType.CROSS_DOMAIN,
        }

        own_chunks = [
            n for n in graph.nodes.values()
            if n.node_type == NodeType.CHUNK and n.filename == filename
        ]
        if not own_chunks:
            return ""

        scores: dict[str, float] = {}
        for chunk in own_chunks:
            for neighbor, edge in graph.get_neighbors(chunk.id):
                if edge.edge_type not in relevant_edges:
                    continue
                tgt_file = neighbor.filename or ""
                if not tgt_file or tgt_file == filename:
                    continue
                scores[tgt_file] = scores.get(tgt_file, 0.0) + float(edge.weight or 0)

        if not scores:
            return ""

        # Deterministic order: highest weight first, ties broken by target
        # filename so the same graph always produces the same Related block.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        picks = ranked[:top_n]

        lines = [
            self._RELATED_BLOCK_START,
            "",
            "## Related",
            "",
        ]
        for tgt_file, _score in picks:
            slug = Path(tgt_file).stem
            lines.append(f"- [[{slug}]]")
        lines.append("")
        lines.append(self._RELATED_BLOCK_END)
        return "\n".join(lines)

    def _inject_related_block(self, body: str, related_block: str) -> str:
        """Insert or replace the auto-generated related block.

        Idempotent: re-running with the same body+block returns body
        unchanged once the block is present and current.
        """
        if not related_block:
            return body

        start = self._RELATED_BLOCK_START
        end = self._RELATED_BLOCK_END

        if start in body and end in body:
            # Replace the existing block in-place.
            pre, _, rest = body.partition(start)
            _, _, post = rest.partition(end)
            return pre + related_block + post

        # Append, preserving a trailing newline.
        sep = "" if body.endswith("\n") else "\n"
        return body + sep + "\n" + related_block + "\n"

    def graph_neighbors(
        self,
        filename: str,
        heading: str = "",
        offset: int = 0,
        edge_type: str = "",
        limit: int = 50,
        min_weight: float = 0.0,
        query: str = "",
    ) -> str:
        """Find semantically related sections for a given file/section.

        Returns neighbors grouped by edge type (similar, inter_file,
        cross_domain, parent_child, relates_to, references). If ``heading``
        is provided, neighbors of that specific section are returned;
        otherwise neighbors are gathered for every section in the file.

        Args:
            filename: Canonical KB path, e.g. ``"canon:foo.md"`` or
                ``"knowledge:wiki/x.md"``. Legacy forms still resolve.
            heading: Optional H1-H5 heading text (case-insensitive substring).
            offset: Skip the first N edges of the flattened stream.
            edge_type: One of ``similar``, ``inter_file``, ``cross_domain``,
                ``parent_child``, ``relates_to``, ``references``, or empty
                for all.
            limit: Maximum edges to render (default 50).
            min_weight: Drop edges below this weight (0.0-1.0).
            query: Optional natural-language query. When the heading is
                ambiguous and resolves to >1 candidate, candidates are
                reranked by cosine similarity against this query before
                rendering the disambiguation list.
        """
        log_event(dbg, "tool_call", tool="graph_neighbors", filename=filename,
                  heading=heading, offset=offset, edge_type=edge_type,
                  min_weight=min_weight, query=query)
        if not self.kb_index or not self.kb_index.graph:
            return "No knowledge graph available."

        graph = self.kb_index.graph

        try:
            offset_i = max(0, int(offset))
        except (TypeError, ValueError):
            offset_i = 0
        try:
            limit_i = max(1, int(limit))
        except (TypeError, ValueError):
            limit_i = 50
        try:
            min_weight_f = max(0.0, float(min_weight))
        except (TypeError, ValueError):
            min_weight_f = 0.0

        edge_type_filter = (edge_type or "").strip().lower()
        valid_edge_types = {"similar", "inter_file", "cross_domain",
                            "parent_child", "relates_to", "references"}
        if edge_type_filter and edge_type_filter not in valid_edge_types:
            return (f"Invalid edge_type '{edge_type}'. "
                    f"Valid: {', '.join(sorted(valid_edge_types))} or '' for all.")

        nodes, disambig = _resolve_chunk_nodes(
            graph, filename, heading,
            caller="graph_neighbors", query=query, kb_index=self.kb_index,
        )
        if disambig:
            return disambig

        # Cap start nodes to prevent output explosion; record overflow.
        total_matches = len(nodes)
        nodes = nodes[:5]

        if not nodes:
            return (f"No sections found for '{filename}'"
                    + (f" > {heading}" if heading else ""))

        # Build flattened (start_node, neighbor, edge) stream across all start
        # nodes, grouped by start node and then edge_type for stable ordering.
        flattened: list[tuple] = []  # (start_node, edge_type_value, neighbor, edge)
        for node in nodes:
            neighbors = graph.get_neighbors(node.id)
            by_type: dict[str, list[tuple]] = {}
            for neighbor, edge in neighbors:
                etv = edge.edge_type.value
                if edge_type_filter and etv != edge_type_filter:
                    continue
                if edge.weight < min_weight_f:
                    continue
                by_type.setdefault(etv, []).append((neighbor, edge))
            for etv in sorted(by_type.keys()):
                for neighbor, edge in by_type[etv]:
                    flattened.append((node, etv, neighbor, edge))

        total_edges = len(flattened)
        if total_edges == 0:
            if edge_type_filter:
                return (f"No '{edge_type_filter}' edges found for '{filename}'"
                        + (f" > {heading}" if heading else ""))
            return (f"No neighbors found for '{filename}'"
                    + (f" > {heading}" if heading else ""))

        page = flattened[offset_i: offset_i + limit_i]
        if not page:
            return (f"No edges in offset window {offset_i}-{offset_i + limit_i} "
                    f"of {total_edges} total. Try a smaller offset.")

        results: list[str] = []
        current_node = None
        current_etype = None
        for start_node, etv, neighbor, edge in page:
            if start_node is not current_node:
                results.append(f"## {start_node.heading}")
                current_node = start_node
                current_etype = None
            if etv != current_etype:
                results.append(f"  [{etv}]")
                current_etype = etv
            direction = "→" if edge.source_id == start_node.id else "←"
            results.append(
                f"    {direction} {neighbor.name} (weight: {edge.weight:.2f})"
                f"{_format_edge_provenance(edge)}"
            )

        # Footers
        end = offset_i + len(page)
        if end < total_edges or offset_i > 0:
            if edge_type_filter:
                results.append(
                    f"\n[showing edges {offset_i}-{end} of {total_edges} "
                    f"of type '{edge_type_filter}'; "
                    f"call with offset={end} for next page, "
                    f"or remove edge_type to see all types]"
                )
            else:
                results.append(
                    f"\n[showing edges {offset_i}-{end} of {total_edges}; "
                    f"call again with offset={end} or "
                    f"edge_type=inter_file/cross_domain/similar to narrow]"
                )
        elif edge_type_filter:
            results.append(
                f"\n[showing all {total_edges} edges of type '{edge_type_filter}'; "
                f"remove edge_type to see all types]"
            )

        if total_matches > 5:
            results.append(
                f"[showing edges from 5 of {total_matches} matching sections; "
                f"pass an explicit heading to narrow]"
            )

        return "\n".join(results)

    def graph_traverse(
        self,
        filename: str,
        heading: str = "",
        depth: int = 2,
        min_weight: float = 0.0,
        offset: int = 0,
        limit: int = 100,
        exclude_edge_types: str = "",
        query: str = "",
    ) -> str:
        """BFS traversal from a section, returning connected concepts.

        Explores outward from a starting section up to the given depth.
        Always skips PARENT_CHILD edges (structural H1->H2 hierarchy, not
        semantic — following them causes exponential explosion). Additional
        edge types can be skipped via ``exclude_edge_types``. When the graph
        has no neighbors at all, falls back to vector search using the
        section's heading as query.

        Args:
            filename: Canonical KB path (or legacy form).
            heading: Optional H1-H5 heading to start from.
            depth: BFS depth limit (default 2).
            min_weight: Drop edges below this weight (0.0-1.0).
            offset: Skip the first N visited edges (per start node) for
                pagination across truncated traversals.
            limit: Maximum visited edges to render per start node
                (default 100).
            exclude_edge_types: Comma-separated edge types to skip in
                addition to ``parent_child``. Valid values: ``similar``,
                ``inter_file``, ``cross_domain``, ``references``,
                ``relates_to``. Example: ``"similar"`` to see only
                cross-file structural edges. Passing ``parent_child``
                is accepted (and silently dropped) since it is always
                excluded.
            query: Optional natural-language query for cosine reranking
                of ambiguous heading matches.
        """
        log_event(
            dbg, "tool_call", tool="graph_traverse",
            filename=filename, heading=heading, depth=depth,
            min_weight=min_weight, offset=offset, limit=limit,
            exclude_edge_types=exclude_edge_types, query=query,
        )
        if not self.kb_index or not self.kb_index.graph:
            return "No knowledge graph available."

        graph = self.kb_index.graph
        try:
            max_depth = int(depth)
        except (TypeError, ValueError):
            max_depth = 2
        try:
            min_weight_f = max(0.0, float(min_weight))
        except (TypeError, ValueError):
            min_weight_f = 0.0
        try:
            offset_i = max(0, int(offset))
        except (TypeError, ValueError):
            offset_i = 0
        try:
            limit_i = max(1, int(limit))
        except (TypeError, ValueError):
            limit_i = 100

        # Always skip PARENT_CHILD — structural hierarchy, not semantic.
        exclude: set = {EdgeType.PARENT_CHILD}
        valid_exclude_types = {
            "similar": EdgeType.SIMILAR,
            "inter_file": EdgeType.INTER_FILE,
            "cross_domain": EdgeType.CROSS_DOMAIN,
            "references": EdgeType.REFERENCES,
            "relates_to": EdgeType.RELATES_TO,
        }
        bad_excludes: list[str] = []
        # P0-3: parent_child is always excluded by design; the legacy code
        # treated passing it as an error, which is a foot-gun. Accept it
        # silently and surface a one-line note in the result so the caller
        # can self-correct without a refusal cycle.
        dropped_pc = False
        for raw in (exclude_edge_types or "").split(","):
            tok = raw.strip().lower()
            if not tok:
                continue
            if tok == "parent_child":
                dropped_pc = True
                continue
            mapped = valid_exclude_types.get(tok)
            if mapped is None:
                bad_excludes.append(tok)
            else:
                exclude.add(mapped)
        if bad_excludes:
            return (
                f"Invalid exclude_edge_types value(s): "
                f"{', '.join(bad_excludes)}. "
                f"Valid: {', '.join(sorted(valid_exclude_types.keys()))} "
                f"(comma-separated; parent_child is always excluded)."
            )

        start_nodes, disambig = _resolve_chunk_nodes(
            graph, filename, heading,
            caller="graph_traverse", query=query, kb_index=self.kb_index,
        )
        if disambig:
            return disambig

        # Cap start nodes to prevent output explosion
        start_nodes = start_nodes[:5]

        if not start_nodes:
            return f"No sections found for '{filename}'" + (f" > {heading}" if heading else "")

        results = []
        for start in start_nodes:
            results.append(f"## Traversal from {start.name} (depth {max_depth})")
            visited = graph.traverse(
                start.id, max_depth=max_depth, exclude_edge_types=exclude
            )
            if min_weight_f > 0.0:
                visited = [(n, e, d) for (n, e, d) in visited if e.weight >= min_weight_f]
            total_visited = len(visited)
            if visited:
                page = visited[offset_i: offset_i + limit_i]
                if not page:
                    results.append(
                        f"  No edges in offset window {offset_i}-"
                        f"{offset_i + limit_i} of {total_visited} total. "
                        f"Try a smaller offset."
                    )
                else:
                    for neighbor, edge, d in page:
                        indent = "  " * d
                        results.append(
                            f"{indent}[{edge.edge_type.value}] {neighbor.name} "
                            f"(depth {d}, weight {edge.weight:.2f})"
                            f"{_format_edge_provenance(edge)}"
                        )
                    end = offset_i + len(page)
                    if end < total_visited or offset_i > 0:
                        results.append(
                            f"  [showing edges {offset_i}-{end} of "
                            f"{total_visited}; call again with offset={end} "
                            f"for next page]"
                        )
            else:
                # Graph has no neighbors — fall back to vector search
                results.append("  No graph connections. Falling back to vector search:")
                query = start.heading.split(" > ")[-1] if start.heading else start.name
                vec_results = self.kb_index.search(query, top_k=3)
                for r in vec_results:
                    rfile = r.get("filename", r.get("path", ""))
                    rsrc = r.get("source", "knowledge")
                    try:
                        rfile = kb_paths.to_canonical(str(rsrc), str(rfile))
                    except kb_paths.KBPathError:
                        pass
                    rhead = r.get("heading", "")
                    rscore = r.get("score", 0)
                    results.append(f"    > {rfile} > {rhead} (score: {rscore:.2f})")
            results.append("")

        if not results:
            base = f"No traversal results for '{filename}'"
            if dropped_pc:
                base = (
                    "note: 'parent_child' is always excluded from "
                    "graph_traverse; dropped from exclude_edge_types.\n\n"
                ) + base
            return base
        body = "\n".join(results)
        if dropped_pc:
            body = (
                "note: 'parent_child' is always excluded from "
                "graph_traverse; dropped from exclude_edge_types.\n\n"
            ) + body
        return body

    def graph_search(self, query: str, min_weight: float = 0.0) -> str:
        """Hybrid search: vector hits expanded with their graph neighborhood.

        First runs semantic search, then for each hit shows its outgoing
        graph edges so the agent sees both lexically-related content and
        topically-connected sections in one shot.

        Args:
            query: Semantic search query.
            min_weight: Drop connected edges below this weight (0.0-1.0).
                Vector hits themselves are not filtered.
        """
        if not self.kb_index or not self.kb_index.graph:
            return "No knowledge graph available."

        try:
            min_weight_f = max(0.0, float(min_weight))
        except (TypeError, ValueError):
            min_weight_f = 0.0

        # Empty query — fall back to listing graph stats
        if not query or not query.strip():
            stats = self.kb_index.graph.get_stats()
            return (
                f"Graph has {stats.get('nodes', 0)} nodes and {stats.get('edges', 0)} edges.\n"
                f"Provide a search query to find connected sections. "
                f"Example: graph_search(query='architecture')"
            )

        graph = self.kb_index.graph

        # Start with vector search
        search_results = self.kb_index.search(query, top_k=5)
        if not search_results:
            return f"No results for '{query}'"

        results = [f"# Graph Search: {query}\n"]

        for r in search_results:
            filename = r.get("filename", "")
            source = r.get("source", "knowledge")
            heading = r.get("heading", "")
            summary = r.get("summary", "")
            score = r.get("score", 0)
            try:
                cano = kb_paths.to_canonical(str(source), str(filename))
            except kb_paths.KBPathError:
                cano = filename
            results.append(f"## {cano}")
            results.append(f"  > {heading} (score: {score:.3f})")
            if summary:
                results.append(f"    {summary}")

            norm_heading = _normalize_heading(heading)
            chunk_nodes = [n for n in graph.nodes.values()
                           if n.node_type == NodeType.CHUNK
                           and n.filename == filename
                           and (n.attributes or {}).get("source") == source
                           and _normalize_heading(n.heading) == norm_heading]

            for node in chunk_nodes[:1]:  # Show neighbors for top chunk match
                neighbors = graph.get_neighbors(node.id)
                if min_weight_f > 0.0:
                    neighbors = [(n, e) for (n, e) in neighbors if e.weight >= min_weight_f]
                if neighbors:
                    results.append("Connected:")
                    for neighbor, edge in neighbors:
                        results.append(
                            f"  [{edge.edge_type.value}] {neighbor.name} "
                            f"(weight: {edge.weight:.2f})"
                            f"{_format_edge_provenance(edge)}"
                        )
                else:
                    results.append("  No graph connections.")

            results.append("")

        return "\n".join(results)

    def graph_stats(self) -> str:
        """Show knowledge graph statistics: node/edge counts, edge distribution,
        connectivity metrics, and most-connected nodes."""
        if not self.kb_index or not self.kb_index.graph:
            return "No knowledge graph available."

        stats = self.kb_index.graph.get_stats()
        lines = [f"# Graph Statistics"]
        lines.append(f"Nodes: {stats['nodes']}, Edges: {stats['edges']}")

        # P0-2: single-line edge_share summary so the model never has to
        # arithmetic these from raw counts. Stable key order (most-common
        # KB edge types first), only types with count > 0 emitted.
        edge_share = stats.get("edge_share") or {}
        if edge_share:
            ordered_keys = ["parent_child", "similar", "inter_file",
                            "cross_domain", "references", "relates_to"]
            shown_keys = [k for k in ordered_keys if k in edge_share]
            shown_keys += [k for k in edge_share if k not in ordered_keys]
            parts = [f"{k}={edge_share[k] * 100:.1f}%" for k in shown_keys]
            lines.append(f"edge_share: {' '.join(parts)}")

        # Node type breakdown
        lines.append("\n## Node Types")
        for ntype, count in stats.get("node_types", {}).items():
            lines.append(f"  {ntype}: {count}")

        # Edge type breakdown
        lines.append("\n## Edge Types")
        edge_types = stats.get("edge_types", {})
        for etype, count in edge_types.items():
            lines.append(f"  {etype}: {count}")

        # P2.4: Provenance breakdown — explicit references (wiki / markdown
        # / prose) vs algorithmic similarity vs structural hierarchy. The
        # agent uses this to gauge whether the graph is "linked" by author
        # intent or merely by embedding similarity.
        graph = self.kb_index.graph
        ref_kinds: dict[str, int] = {"wiki": 0, "markdown": 0, "prose": 0, "other": 0}
        for e in graph.edges.values():
            if e.edge_type != EdgeType.REFERENCES:
                continue
            kind = (getattr(e, "attributes", {}) or {}).get("link_kind") or "other"
            ref_kinds[kind] = ref_kinds.get(kind, 0) + 1
        ref_total = sum(ref_kinds.values())
        if ref_total:
            lines.append("\n## REFERENCES Breakdown")
            for kind in ("wiki", "markdown", "prose", "other"):
                cnt = ref_kinds.get(kind, 0)
                if cnt:
                    pct = cnt / ref_total * 100
                    lines.append(f"  {kind}: {cnt} ({pct:.0f}%)")

        # Inter/intra ratio + provenance-class summary
        inter = edge_types.get("inter_file", 0) + edge_types.get("cross_domain", 0)
        intra = edge_types.get("similar", 0) + edge_types.get("parent_child", 0)
        total_edges = stats["edges"]
        if total_edges > 0:
            lines.append(f"\n## Connectivity")
            lines.append(f"  Inter-file/cross-domain edges: {inter} ({inter/total_edges*100:.0f}%)")
            lines.append(f"  Intra-file edges: {intra} ({intra/total_edges*100:.0f}%)")
            lines.append(f"  Orphan nodes: {stats.get('orphan_nodes', 0)}")
            lines.append(f"  Avg edges per node: {stats.get('avg_edges_per_node', 0):.1f}")

            # P2.4: provenance-class roll-up so the agent can tell at a
            # glance whether links are author-intended, model-inferred, or
            # purely structural.
            explicit_refs = ref_kinds.get("wiki", 0) + ref_kinds.get("markdown", 0)
            prose_refs = ref_kinds.get("prose", 0)
            similarity = (
                edge_types.get("similar", 0)
                + edge_types.get("inter_file", 0)
                + edge_types.get("cross_domain", 0)
            )
            hierarchy = edge_types.get("parent_child", 0)
            entities = edge_types.get("relates_to", 0)
            lines.append(f"\n## Provenance Classes")
            lines.append(
                f"  explicit-reference (wiki+markdown): {explicit_refs} "
                f"({explicit_refs/total_edges*100:.0f}%)"
            )
            lines.append(
                f"  prose-bridge (heuristic): {prose_refs} "
                f"({prose_refs/total_edges*100:.0f}%)"
            )
            lines.append(
                f"  similarity (embedding cosine): {similarity} "
                f"({similarity/total_edges*100:.0f}%)"
            )
            lines.append(
                f"  hierarchy (parent_child): {hierarchy} "
                f"({hierarchy/total_edges*100:.0f}%)"
            )
            if entities:
                lines.append(
                    f"  entity (relates_to): {entities} "
                    f"({entities/total_edges*100:.0f}%)"
                )

        # Most-connected nodes (semantic + cross-file; parent_child excluded)
        top = stats.get("most_connected", [])
        if top:
            lines.append(f"\n## Most Connected (semantic + cross-file)")
            for entry in top:
                # Tolerate both new dict shape and legacy tuple shape
                if isinstance(entry, dict):
                    fn = entry.get("filename", "")
                    hd = entry.get("heading", "")
                    cnt = entry.get("count", 0)
                    nm = entry.get("name", "")
                    share = entry.get("share_non_pc", 0.0) or 0.0
                    # P0-2: per-hub share of non-parent_child edges so the
                    # model can compare hubs without re-arithmetic.
                    share_str = (
                        f", {share * 100:.1f}% of non-parent_child edges"
                        if share > 0 else ""
                    )
                    if fn:
                        lines.append(f"  {fn}")
                        lines.append(
                            f"    > {hd} ({cnt} connections{share_str})"
                        )
                    else:
                        lines.append(f"  {nm}: {cnt} connections{share_str}")
                else:
                    name, cnt = entry
                    lines.append(f"  {name}: {cnt} connections")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # B4 — describe_node: per-section provenance dump
    # ------------------------------------------------------------------

    DESCRIBE_NODE_MAX_EDGES_PER_TYPE = 12

    def describe_node(
        self,
        filename: str,
        heading: str = "",
        min_weight: float = 0.0,
        query: str = "",
    ) -> str:
        """Render everything the graph knows about a section in one block.

        Resolves ``(filename, heading)`` through the same waterfall as
        ``graph_neighbors`` / ``graph_traverse`` (canonical paths preferred,
        legacy bare paths still work, ``filename > heading`` combined form
        accepted), then emits:

          - canonical address, tier, source
          - on-disk file metadata (token cost, line count) when readable
          - structured summary if the chunker produced one
          - outgoing edges grouped by EdgeType, each line annotated with
            ``_format_edge_provenance`` so the agent sees WHY the link exists
          - incoming edges grouped by EdgeType the same way
          - quick stats (degree, distinct neighbors, ref-kind breakdown)

        This is the single entrypoint for "tell me everything about this
        node" — use it instead of stitching together graph_neighbors +
        graph_traverse + read_knowledge_section when you want to evaluate a
        single section's place in the graph.

        Args:
            filename: Canonical KB path (or legacy bare path).
            heading: Optional H1-H5 heading to focus on.
            min_weight: Drop edges below this weight (0.0-1.0) before the
                per-type cap kicks in. Use 0.83+ to suppress the noisy
                long tail on densely-curated files (quote vaults etc.).
            query: Optional natural-language query for cosine reranking
                of ambiguous heading matches.
        """
        log_event(
            dbg, "tool_call", tool="describe_node",
            filename=filename, heading=heading, min_weight=min_weight,
            query=query,
        )
        if not self.kb_index or not self.kb_index.graph:
            return "No knowledge graph available."

        graph = self.kb_index.graph
        try:
            min_weight_f = max(0.0, float(min_weight))
        except (TypeError, ValueError):
            min_weight_f = 0.0

        nodes, disambig = _resolve_chunk_nodes(
            graph, filename, heading,
            caller="describe_node", query=query, kb_index=self.kb_index,
        )
        if disambig:
            return disambig
        if not nodes:
            return (
                f"No sections found for '{filename}'"
                + (f" > {heading}" if heading else "")
                + ". Try graph_neighbors with the same args to see what is "
                "indexed under that path."
            )

        # When ambiguous (multiple chunk hits), describe up to 3 to keep
        # output bounded but still useful when the agent passed only an H1
        # and the file has same-named sections.
        nodes = nodes[:3]

        out: list[str] = []
        if len(nodes) > 1:
            out.append(
                f"# describe_node — {len(nodes)} matching sections (showing all)"
            )

        for node in nodes:
            out.extend(self._render_node_description(graph, node, min_weight_f))
            out.append("")

        return "\n".join(out).rstrip()

    def _render_node_description(self, graph, node, min_weight: float = 0.0) -> list[str]:
        """Render one section's full descriptor."""
        attrs = node.attributes or {}
        source = attrs.get("source", "knowledge")
        try:
            cano = kb_paths.to_canonical(str(source), str(node.filename))
        except Exception:
            cano = node.filename

        from knowledge.index import KBIndex
        tier = KBIndex._compute_tier(source, node.filename or "")

        out: list[str] = []
        out.append(f"# Node: {cano}")
        out.append(f"- heading: {node.heading or '(unnamed)'}")
        out.append(f"- tier: {tier}")
        out.append(f"- source: {source}")
        if node.summary:
            summary = node.summary.strip()
            if len(summary) > 400:
                summary = summary[:397] + "..."
            out.append(f"- summary: {summary}")

        token_count = attrs.get("token_count")
        if isinstance(token_count, (int, float)) and token_count:
            out.append(f"- tokens: ~{int(token_count):,}")

        # Bucket all edges incident to this node by direction + type.
        outgoing: dict[str, list[tuple]] = {}
        incoming: dict[str, list[tuple]] = {}
        ref_kinds: dict[str, int] = {}
        distinct_neighbors: set[str] = set()

        filtered_below_threshold = 0
        for edge in graph.edges.values():
            if edge.source_id == node.id:
                bucket = outgoing
                neighbor = graph.nodes.get(edge.target_id)
            elif edge.target_id == node.id:
                bucket = incoming
                neighbor = graph.nodes.get(edge.source_id)
            else:
                continue
            if not neighbor:
                continue
            if min_weight > 0.0 and edge.weight < min_weight:
                filtered_below_threshold += 1
                continue
            distinct_neighbors.add(neighbor.id)
            etv = edge.edge_type.value
            bucket.setdefault(etv, []).append((neighbor, edge))
            if edge.edge_type == EdgeType.REFERENCES:
                kind = (edge.attributes or {}).get("link_kind") or "other"
                ref_kinds[kind] = ref_kinds.get(kind, 0) + 1

        total_out = sum(len(v) for v in outgoing.values())
        total_in = sum(len(v) for v in incoming.values())
        out.append(
            f"- degree: {total_out + total_in} "
            f"(out={total_out}, in={total_in}, "
            f"distinct neighbors={len(distinct_neighbors)})"
        )
        if ref_kinds:
            kinds_str = ", ".join(
                f"{k}={v}" for k, v in sorted(ref_kinds.items(), key=lambda kv: -kv[1])
            )
            out.append(f"- references-by-kind: {kinds_str}")
        if min_weight > 0.0 and filtered_below_threshold > 0:
            out.append(
                f"- min_weight={min_weight:.2f} "
                f"(suppressed {filtered_below_threshold} edges below threshold)"
            )

        def _render_block(title: str, bucket: dict[str, list[tuple]]) -> None:
            if not bucket:
                return
            out.append("")
            out.append(f"## {title}")
            cap = self.DESCRIBE_NODE_MAX_EDGES_PER_TYPE
            for etv in sorted(bucket.keys()):
                edges = sorted(bucket[etv], key=lambda ne: -ne[1].weight)
                out.append(f"  [{etv}] ({len(edges)})")
                for neighbor, edge in edges[:cap]:
                    name = neighbor.name or neighbor.heading or neighbor.filename
                    out.append(
                        f"    - {name} (weight: {edge.weight:.2f})"
                        f"{_format_edge_provenance(edge)}"
                    )
                if len(edges) > cap:
                    out.append(
                        f"    ... and {len(edges) - cap} more "
                        f"(call graph_neighbors with edge_type='{etv}' to page through)"
                    )

        _render_block("Outgoing", outgoing)
        _render_block("Incoming", incoming)

        if not outgoing and not incoming:
            out.append("")
            if min_weight > 0.0 and filtered_below_threshold > 0:
                out.append(
                    f"## (no edges above min_weight={min_weight:.2f})"
                )
                out.append(
                    f"  All {filtered_below_threshold} edges were filtered. "
                    f"Re-call with a lower min_weight to inspect them."
                )
            else:
                out.append(
                    "## (orphan)"
                )
                out.append(
                    "  No edges. Add a wiki link from a related page or save a "
                    "compiled wiki/ entry that cites this section to integrate "
                    "it into the graph."
                )

        return out

    # ------------------------------------------------------------------
    # P3.2 — lint_knowledge: structural audit of the KB
    # ------------------------------------------------------------------

    LINT_OVERSIZED_TOKENS = 2000
    LINT_HEADING_COLLISION_THRESHOLD = 3
    LINT_MAX_ITEMS_PER_SECTION = 15
    # D5: flat-similarity-cluster — flag files whose intra-file SIMILAR
    # edges all bunch into a narrow weight band. Signals a tightly-clustered
    # source (quote vault, glossary) where the embedding floor is high
    # enough to bury real signal under noise.
    LINT_FLAT_CLUSTER_MIN_EDGES = 20      # need a meaningful sample
    LINT_FLAT_CLUSTER_BAND = 0.05         # min - max weight spread

    def lint_knowledge(self, scope: str = "") -> str:
        """Audit the KB for structural problems and surface findings the
        agent can fix with save_knowledge edits.

        Checks:
          - Orphans            chunks with no semantic edges (SIMILAR / INTER_FILE
                               / CROSS_DOMAIN / REFERENCES / RELATES_TO). Folder /
                               heading hierarchy edges are excluded — those are
                               structural, not semantic.
          - Broken Wiki Links  [[link]] and [text](file.md) targets that fail to
                               resolve to any indexed file.
          - Heading Collisions same heading appears in 3+ files (potential
                               duplicate / merge candidate).
          - Oversized Chunks   chunks above 2000 tokens — usually chunker
                               overflow on dense files; a candidate for splitting
                               into focused subsections.
          - Flat Sim Clusters  files whose intra-file SIMILAR edges all bunch
                               into a 0.05-wide weight band (>= 20 edges).
                               Signals a tightly-clustered source (quote vault,
                               glossary) where the embedding floor is high
                               enough to bury real signal under noise — a hint
                               to split into per-entity files.
          - Nested tag YAML    wiki/memory files whose frontmatter ``tags:``
                               still contains nested lists (agent mistake).
          - KB-id markdown hrefs  ``](canon:...)`` / ``](knowledge:...)`` in
                               wiki/memory bodies — not Obsidian-valid; use
                               filesystem-relative paths to canon instead.

        scope = "" lints everything. Pass a filename (e.g. "wiki/cortisol.md")
        to scope every check to that single file's chunks and links.
        """
        log_event(dbg, "tool_call", tool="lint_knowledge", scope=scope)
        if (
            not self.kb_index
            or self.kb_index.graph is None
            or self.kb_index.table is None
        ):
            return "No knowledge index available."

        from knowledge.wiki_links import parse_links, resolve_link

        graph = self.kb_index.graph

        try:
            df = self.kb_index.table.to_pandas()
        except Exception as e:
            return f"lint_knowledge: failed to read index: {e}"
        if df.empty:
            return "Knowledge base is empty — nothing to lint."

        scope_clean = (scope or "").strip()
        if scope_clean:
            df = df[df["filename"] == scope_clean].reset_index(drop=True)
            if df.empty:
                return f"lint_knowledge: no chunks indexed for '{scope_clean}'."

        chunk_ids_in_scope = set(df["id"].tolist())
        files_in_scope = sorted(set(df["filename"].tolist()))

        # ---- 1. Orphans ----------------------------------------------------
        SEMANTIC_EDGES = {
            EdgeType.SIMILAR,
            EdgeType.INTER_FILE,
            EdgeType.CROSS_DOMAIN,
            EdgeType.REFERENCES,
            EdgeType.RELATES_TO,
        }
        connected: set[str] = set()
        for e in graph.edges.values():
            if e.edge_type in SEMANTIC_EDGES:
                connected.add(e.source_id)
                connected.add(e.target_id)
        orphans: list[tuple[str, str]] = []
        for nid, n in graph.nodes.items():
            if n.node_type != NodeType.CHUNK:
                continue
            if nid not in chunk_ids_in_scope:
                continue
            if nid in connected:
                continue
            orphans.append((n.filename or "", n.heading or ""))
        orphans.sort()

        # ---- 2. Broken wiki links -----------------------------------------
        try:
            indexed_files = self.kb_index.list_indexed_filenames()
        except Exception:
            indexed_files = []
        broken_links: list[dict] = []
        for _, row in df.iterrows():
            # The chunk body is stored as `document`. It includes a YAML
            # context header (file/position/doc_summary) prepended for
            # search context — wiki links inside the original markdown
            # body are still present in the rendered chunk.
            doc = row.get("document") or ""
            if not doc:
                continue
            src_filename = row.get("filename") or ""
            src_heading = row.get("heading") or ""
            for link in parse_links(doc):
                hit = resolve_link(link, indexed_files)
                if hit is not None:
                    continue
                broken_links.append({
                    "src_file": src_filename,
                    "src_heading": src_heading,
                    "raw": link.get("raw", ""),
                    "target": link.get("target", ""),
                    "kind": link.get("kind", ""),
                })

        # ---- 3. Heading collisions ----------------------------------------
        # (skip when scoped to a single file — collision detection needs the
        # full corpus to be meaningful)
        heading_collisions: list[tuple[str, list[str]]] = []
        if not scope_clean:
            heading_to_files: dict[str, set[str]] = {}
            for nid, n in graph.nodes.items():
                if n.node_type != NodeType.CHUNK:
                    continue
                h = (n.heading or "").strip().lower()
                if not h:
                    continue
                # Use the leaf heading (last " > " segment) so structural
                # nesting like "topic > subsection" isn't double-counted.
                leaf = h.split(" > ")[-1].strip()
                if not leaf:
                    continue
                heading_to_files.setdefault(leaf, set()).add(n.filename or "")
            for h, files in heading_to_files.items():
                files_clean = {f for f in files if f}
                if len(files_clean) >= self.LINT_HEADING_COLLISION_THRESHOLD:
                    heading_collisions.append((h, sorted(files_clean)))
            heading_collisions.sort(key=lambda x: (-len(x[1]), x[0]))

        # ---- 4. Oversized chunks ------------------------------------------
        oversized: list[tuple[str, str, int]] = []
        if "token_count" in df.columns:
            for _, row in df.iterrows():
                tc = int(row.get("token_count") or 0)
                if tc > self.LINT_OVERSIZED_TOKENS:
                    oversized.append((
                        row.get("filename") or "",
                        row.get("heading") or "",
                        tc,
                    ))
            oversized.sort(key=lambda x: -x[2])

        # ---- 5. Flat-similarity clusters ----------------------------------
        # Group intra-file SIMILAR edge weights by filename and flag files
        # whose weight spread (max - min) is below the band threshold.
        # Skipped when scoped to a single file — a one-file scope can't
        # produce comparative judgement against the rest of the corpus,
        # but the per-file metric still applies, so we include it.
        per_file_weights: dict[str, list[float]] = {}
        for e in graph.edges.values():
            if e.edge_type != EdgeType.SIMILAR:
                continue
            src = graph.nodes.get(e.source_id)
            tgt = graph.nodes.get(e.target_id)
            if not src or not tgt:
                continue
            if src.filename != tgt.filename or not src.filename:
                continue  # SIMILAR is intra-file by construction; defensive
            if src.id not in chunk_ids_in_scope and tgt.id not in chunk_ids_in_scope:
                continue
            per_file_weights.setdefault(src.filename, []).append(e.weight)

        flat_clusters: list[tuple[str, int, float, float]] = []
        for fname, weights in per_file_weights.items():
            if len(weights) < self.LINT_FLAT_CLUSTER_MIN_EDGES:
                continue
            wmin = min(weights)
            wmax = max(weights)
            if (wmax - wmin) <= self.LINT_FLAT_CLUSTER_BAND:
                flat_clusters.append((fname, len(weights), wmin, wmax))
        flat_clusters.sort(key=lambda x: -x[1])

        # ---- 6. Wiki/memory frontmatter + markdown href hygiene ----------
        from knowledge.index import KBIndex

        malformed_nested_tags: list[tuple[str, str]] = []
        canon_id_md_links: list[tuple[str, str]] = []
        for fn in sorted(set(files_in_scope)):
            tier = KBIndex._compute_tier("knowledge", fn)
            if tier not in ("wiki", "memory"):
                continue
            fpath = self.kb_dir / fn
            if not fpath.is_file():
                continue
            try:
                disk_text = fpath.read_text(encoding="utf-8")
            except OSError:
                continue
            if disk_text.startswith("---"):
                meta, _ = _parse_frontmatter(disk_text)
                tg = meta.get("tags")
                if isinstance(tg, list) and any(isinstance(x, list) for x in tg):
                    malformed_nested_tags.append(
                        (fn, "frontmatter `tags` contains nested YAML lists"),
                    )
            for m in _MARKDOWN_CANONICAL_TARGET_RE.finditer(disk_text):
                tgt = m.group(1).strip()
                if " " in tgt:
                    tgt = tgt.split()[0]
                tgt = tgt.strip("\"'")
                if tgt.startswith(("canon:", "knowledge:")):
                    canon_id_md_links.append(
                        (fn, f"markdown link uses KB id in href: {tgt[:72]}"),
                    )

        # ---- Render report ------------------------------------------------
        lines = ["# Knowledge Base Lint Report"]
        if scope_clean:
            lines.append(f"Scope: {scope_clean}")
        else:
            lines.append(f"Scope: full KB ({len(files_in_scope)} files, "
                         f"{len(chunk_ids_in_scope)} chunks)")
        lines.append("")
        lines.append(
            f"Findings: {len(orphans)} orphans, {len(broken_links)} broken links, "
            f"{len(heading_collisions)} heading collisions, "
            f"{len(oversized)} oversized chunks, "
            f"{len(flat_clusters)} flat-similarity clusters, "
            f"{len(malformed_nested_tags)} nested-tag frontmatter issues, "
            f"{len(canon_id_md_links)} canon:/knowledge: markdown hrefs."
        )

        def _render_section(title: str, items: list, render_one) -> None:
            lines.append("")
            lines.append(f"## {title}")
            if not items:
                lines.append("  (none)")
                return
            shown = items[: self.LINT_MAX_ITEMS_PER_SECTION]
            for it in shown:
                lines.append(render_one(it))
            extra = len(items) - len(shown)
            if extra > 0:
                lines.append(f"  … +{extra} more")

        _render_section(
            "Orphans (chunks with no semantic edges)",
            orphans,
            lambda o: f"  - {o[0]} > {o[1]}" if o[1] else f"  - {o[0]}",
        )
        _render_section(
            "Broken Wiki Links (target not indexed / ambiguous)",
            broken_links,
            lambda b: (
                f"  - {b['src_file']} > {b['src_heading']} "
                f"→ {b['raw']} (target: {b['target']}, kind: {b['kind']})"
            ),
        )
        if not scope_clean:
            _render_section(
                f"Heading Collisions (same heading in "
                f"{self.LINT_HEADING_COLLISION_THRESHOLD}+ files)",
                heading_collisions,
                lambda hc: (
                    f"  - '{hc[0]}' in {len(hc[1])} files: "
                    f"{', '.join(hc[1][:5])}"
                    + (f" … +{len(hc[1]) - 5} more" if len(hc[1]) > 5 else "")
                ),
            )
        _render_section(
            f"Oversized Chunks (> {self.LINT_OVERSIZED_TOKENS} tokens)",
            oversized,
            lambda o: f"  - {o[0]} > {o[1]} ({o[2]} tokens)",
        )
        _render_section(
            f"Flat-Similarity Clusters "
            f"(>= {self.LINT_FLAT_CLUSTER_MIN_EDGES} intra-file SIMILAR "
            f"edges within a {self.LINT_FLAT_CLUSTER_BAND:.2f} weight band)",
            flat_clusters,
            lambda fc: (
                f"  - {fc[0]} ({fc[1]} edges, weight {fc[2]:.2f}-{fc[3]:.2f})"
            ),
        )
        _render_section(
            "Nested-tag frontmatter (wiki/memory)",
            malformed_nested_tags,
            lambda nt: f"  - {nt[0]} — {nt[1]}",
        )
        _render_section(
            "Markdown hrefs using canon: / knowledge: ids (invalid in vault)",
            canon_id_md_links,
            lambda cl: f"  - {cl[0]} — {cl[1]}",
        )

        lines.append("")
        lines.append(
            "Fix-it hints: orphans → add wiki links from related pages; "
            "broken links → save_knowledge the missing target or correct the "
            "link; collisions → merge or rename to disambiguate; oversized "
            "chunks → split into focused subsections under finer headings; "
            "flat-similarity clusters → split into per-entity files or "
            "restructure the heading hierarchy so each section carries "
            "distinct vocabulary; nested tags → re-save with a flat string "
            "array; KB-id hrefs → use ../../canon/... relative paths from "
            "the wiki file (see compile_knowledge canon examples)."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # P3.2 — compile_knowledge: stage a wiki page from a raw/ source
    # ------------------------------------------------------------------

    COMPILE_RELATED_TOP_N = 8

    def _compile_canon_citation_examples(
        self,
        canon_top: list[dict],
        wiki_output_relpath: str,
    ) -> list[str]:
        """Emit copy-paste markdown lines: relative ``../../canon/...`` hrefs."""
        if not canon_top:
            return []
        lines: list[str] = [
            "",
            "## Canon citation examples (copy into wiki body)",
            "",
            "Use **filesystem-relative** markdown links from the saved wiki "
            f"path `{wiki_output_relpath}` to canon files — not `canon:` / "
            "`knowledge:` ids inside `(...)` (those break in Obsidian).",
            "",
        ]
        for c in canon_top[: min(5, len(canon_top))]:
            fn = (c.get("filename") or "").strip()
            if not fn:
                continue
            head = (c.get("heading") or "").strip()
            leaf = head.split(" > ")[-1].strip() if head else ""
            anchor = _normalize_heading(leaf) if leaf else ""
            stem = Path(fn).stem.replace("-", " ").title()
            try:
                href = _wiki_to_canon_markdown_href(
                    self.kb_dir,
                    self.canon_dir,
                    wiki_output_relpath,
                    fn,
                    anchor,
                )
            except (OSError, ValueError):
                continue
            lines.append(f"- [{stem}]({href})")
        lines.append("")
        return lines

    def compile_knowledge(
        self,
        source: str = "",
        query: str = "",
        source_type: str = "file",
        source_ref: str = "",
    ) -> str:
        """Stage the compilation of a new wiki page from a source.

        Two source modes:

        * ``source_type="file"`` (default) — ``source`` is a KB path. The tool
          reads the file (typically a raw/ bronze-tier note), runs semantic
          search for related wiki and canon material, and emits a compilation
          plan: suggested wiki slug, top related pages, canon anchors, steps.
        * ``source_type="conversation"`` — ``source_ref`` selects turns from a
          persisted thread. Accepted shapes:

            - ``<conv_id>``                — entire thread
            - ``<conv_id>:last:N``         — most recent N turns
            - ``<conv_id>:turn:A-B``       — inclusive turn range
            - ``<conv_id>:turn:N``         — single turn

          The tool stitches the selected turns into a synthetic source body
          and runs the same compilation pipeline, so wiki pages can be
          compiled directly from a productive chat exchange instead of going
          back to raw/. Citations belong in a ``## Sources`` block of the
          eventual save_knowledge body, in the form
          ``- conversation:<conv_id>:turn:N`` so save_knowledge can verify
          they exist.

        This tool does NOT write anything.

        Args:
          source:       KB path. Required when source_type="file".
          query:        Optional search query override. When empty, the tool
                        derives a query from the source's slug + leading body.
          source_type:  Either "file" (default) or "conversation".
          source_ref:   Conversation selector. Required when
                        source_type="conversation". Ignored otherwise.
        """
        log_event(
            dbg, "tool_call", tool="compile_knowledge",
            source=source, query=query,
            source_type=source_type, source_ref=source_ref,
        )
        if not self.kb_index:
            return "No knowledge index available."

        # B3: route to the conversation-source path before doing any KB
        # filename resolution. This keeps the file-source flow byte-for-byte
        # identical to the prior behaviour for callers that don't pass the
        # new args.
        st = (source_type or "file").strip().lower()
        if st == "conversation":
            return self._compile_knowledge_from_conversation(
                source_ref=source_ref, query=query,
            )
        if st not in ("file", ""):
            return (
                f"Invalid source_type '{source_type}'. "
                f"Use 'file' (default) or 'conversation'."
            )

        resolved = self._resolve_kb_filename(source)
        if resolved is None:
            suggestions = self._suggest_filenames(source)
            hint = (
                f" Did you mean: {', '.join(suggestions)}?"
                if suggestions else ""
            )
            return (
                f"FILE NOT FOUND: '{source}' is not in the knowledge base."
                + hint
            )
        rel_path, _src, abs_path = resolved

        try:
            body = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return f"ERROR: could not read '{rel_path}'."

        from knowledge.index import KBIndex
        tier = KBIndex._compute_tier(_src, rel_path)

        # Tier guidance: compile_knowledge is designed for bronze (raw)
        # inputs, but we don't refuse other tiers — we just flag the
        # unusual case so the agent knows what it's doing.
        tier_note = ""
        if tier == "wiki":
            tier_note = (
                " (NOTE: source is already a wiki page — usually you'd "
                "edit it with save_knowledge, not re-compile from it.)"
            )
        elif tier == "canon":
            tier_note = (
                " (NOTE: source is canon — gold-tier content. Compiling "
                "TO wiki is allowed, but treat the canon text as ground "
                "truth and cite it explicitly.)"
            )

        # Derive a search query when none provided. Stem + first ~250 body
        # chars (whitespace-collapsed) gives the embedder enough lexical
        # signal without dragging in headers / TOCs / dividers.
        derived_query = ""
        if not (query or "").strip():
            stem = Path(rel_path).stem.replace("-", " ").replace("_", " ")
            body_clean = re.sub(r"\s+", " ", body).strip()
            # Skip front-matter and H1 lines if present
            body_for_query = re.sub(
                r"^---.*?---\s*", "", body_clean, count=1, flags=re.DOTALL,
            )
            body_for_query = re.sub(r"^#\s+\S[^.\n]{0,80}", "", body_for_query)
            derived_query = f"{stem} {body_for_query[:250]}".strip()
            search_query = derived_query
        else:
            search_query = query.strip()

        # Run search; fall back gracefully on failure
        try:
            results = self.kb_index.search(search_query, top_k=20)
        except Exception as e:
            results = []
            log_event(
                dbg, "tool_error", tool="compile_knowledge",
                error=f"search failed: {e}",
            )

        # Bucket results by tier; never recommend the source itself.
        wiki_hits: list[dict] = []
        canon_hits: list[dict] = []
        for r in results or []:
            r_filename = r.get("filename") or ""
            if r_filename == rel_path:
                continue
            r_tier = r.get("tier") or KBIndex._compute_tier(
                r.get("source", "knowledge"), r_filename,
            )
            entry = {
                "filename": r_filename,
                "heading": r.get("heading", "") or "",
                "summary": r.get("summary", "") or "",
                "score": float(r.get("weighted_score", r.get("score", 0)) or 0),
                "tier": r_tier,
            }
            if r_tier == "wiki":
                wiki_hits.append(entry)
            elif r_tier == "canon":
                canon_hits.append(entry)

        # Dedupe per-file (best-scoring chunk wins) and trim to top N
        def _dedupe_top(items: list[dict], top_n: int) -> list[dict]:
            best: dict[str, dict] = {}
            for it in items:
                fn = it["filename"]
                if fn not in best or it["score"] > best[fn]["score"]:
                    best[fn] = it
            return sorted(best.values(), key=lambda x: -x["score"])[:top_n]

        wiki_top = _dedupe_top(wiki_hits, self.COMPILE_RELATED_TOP_N)
        canon_top = _dedupe_top(canon_hits, self.COMPILE_RELATED_TOP_N)

        # Suggested wiki slug from the source basename, lowercased + clean
        src_stem = Path(rel_path).stem
        suggested_slug = re.sub(r"[^a-z0-9-]+", "-", src_stem.lower()).strip("-")
        if not suggested_slug:
            suggested_slug = "new-page"
        suggested_filename = f"{suggested_slug}.md"

        # Heading tree of the source (cheap, never loads full content)
        try:
            tree = self.kb_index.get_heading_tree(rel_path, _src) or ""
        except Exception:
            tree = ""

        body_tokens = estimate_tokens(body)

        # ---- Render compilation prompt ------------------------------------
        out: list[str] = []
        out.append(f"# Compile Knowledge from: {rel_path}")
        out.append("")
        out.append(f"## Source [tier={tier}]{tier_note}")
        out.append(f"- path: {rel_path}")
        out.append(f"- size: ~{body_tokens:,} tokens")
        if tree:
            out.append("- structure (heading tree):")
            for line in tree.splitlines():
                out.append(f"  {line}")

        out.append("")
        out.append(f"## Related Wiki Pages (top {len(wiki_top)}) — link with [[wiki-link]]")
        if wiki_top:
            for w in wiki_top:
                slug = Path(w["filename"]).stem
                summary_part = f" — {w['summary'][:120]}" if w["summary"] else ""
                out.append(
                    f"- [[{slug}]]  ({w['filename']}, score {w['score']:.2f})"
                    f"{summary_part}"
                )
        else:
            out.append(
                "  (no related wiki pages — this looks like an entirely new "
                "topic; consider also saving prerequisite concept pages.)"
            )

        out.append("")
        out.append(f"## Canon Anchors (top {len(canon_top)}) — cite as ground truth")
        if canon_top:
            for c in canon_top:
                summary_part = f" — {c['summary'][:120]}" if c["summary"] else ""
                out.append(
                    f"- {c['filename']} > {c['heading']}  "
                    f"(score {c['score']:.2f}){summary_part}"
                )
        else:
            out.append("  (no canon anchors found for this query)")

        wiki_out = f"wiki/{suggested_filename}"
        out.extend(self._compile_canon_citation_examples(canon_top, wiki_out))

        out.append("")
        out.append("## Suggested Output")
        out.append(f"- filename: {wiki_out}")
        if derived_query:
            out.append(f"- derived search query: {derived_query[:200]}")

        out.append("")
        out.append("## Compilation Plan")
        out.append(
            "1. read_knowledge_section(\""
            f"{rel_path}\", <heading>) — load source content piecemeal "
            "from the heading tree above."
        )
        out.append(
            "2. Synthesize a focused, self-contained wiki page. Keep it "
            "narrow — one concept per page (LLM Wiki Pattern)."
        )
        if wiki_top:
            sample = ", ".join(f"[[{Path(w['filename']).stem}]]" for w in wiki_top[:3])
            out.append(
                f"3. Link related wiki pages inline using [[wiki-link]] "
                f"syntax (e.g. {sample})."
            )
        else:
            out.append(
                "3. No existing wiki neighbors yet — record any prerequisite "
                "concepts as separate pages so the new node isn't an island."
            )
        if canon_top:
            out.append(
                "4. Cite canon using the **relative** markdown links from "
                "## Canon citation examples (``../../canon/...``). Do not put "
                "``canon:`` or ``knowledge:`` inside the link target parentheses."
            )
        else:
            out.append(
                "4. No canon anchors — flag any uncited claim as needing "
                "future canon material."
            )
        out.append(
            f"5. save_knowledge(\"{suggested_filename}\", <compiled content>, "
            "[\"tag-one\", \"tag-two\"]) — writes to wiki/, triggers reindex, "
            "and auto-injects a Related-pages block. Pass tags as a **flat** "
            "array of strings."
        )
        return "\n".join(out)

    # ------------------------------------------------------------------
    # B3 — compile_knowledge from a conversation source
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_conversation_ref(ref: str) -> tuple[str | None, str | None, str]:
        """Parse a source_ref of the form expected by compile_knowledge.

        Returns ``(conv_id, spec, error)``. ``spec`` is either ``""`` (whole
        thread), ``"last:N"`` or ``"range:A:B"`` (the same vocabulary
        ``read_conversation`` uses internally). ``error`` is non-empty when
        the reference is malformed.
        """
        if not ref or not ref.strip():
            return None, None, (
                "source_ref is required when source_type='conversation'. "
                "Use '<conv_id>', '<conv_id>:last:N', or '<conv_id>:turn:A-B'."
            )
        raw = ref.strip()
        # Whole-thread: no colon, or only conv id portion present.
        if ":" not in raw:
            return raw, "", ""
        head, _, tail = raw.partition(":")
        conv_id = head.strip()
        rest = tail.strip()
        if not conv_id:
            return None, None, "source_ref missing conversation id before ':'."
        if rest.startswith("last:"):
            try:
                n = int(rest.split(":", 1)[1])
            except (ValueError, IndexError):
                return None, None, (
                    f"Invalid last:N selector in source_ref='{ref}'."
                )
            if n <= 0:
                return None, None, "last:N requires N >= 1."
            return conv_id, f"last:{n}", ""
        if rest.startswith("turn:"):
            sel = rest.split(":", 1)[1]
            # Single index: "turn:N"
            if sel.isdigit():
                idx = int(sel)
                return conv_id, f"range:{idx}:{idx}", ""
            # Range: "turn:A-B"
            if "-" in sel:
                a_str, b_str = sel.split("-", 1)
                if a_str.isdigit() and b_str.isdigit():
                    a, b = int(a_str), int(b_str)
                    if b < a:
                        return None, None, (
                            f"turn range A-B requires B >= A (got {a}-{b})."
                        )
                    return conv_id, f"range:{a}:{b}", ""
            return None, None, (
                f"Invalid turn selector '{sel}'. "
                f"Use 'turn:N' or 'turn:A-B'."
            )
        return None, None, (
            f"Invalid source_ref '{ref}'. "
            f"Use '<conv_id>', '<conv_id>:last:N', or '<conv_id>:turn:A-B'."
        )

    def _compile_knowledge_from_conversation(
        self, source_ref: str, query: str = "",
    ) -> str:
        """Compile a wiki page draft from a slice of a persisted conversation.

        Mirrors the file-source ``compile_knowledge`` output (related wiki,
        canon anchors, suggested slug, plan), but the source body is the
        rendered conversation turns instead of a markdown file. Each turn is
        labelled ``turn[N]`` so the eventual ``## Sources`` block in the
        save_knowledge call can cite ``conversation:<conv_id>:turn:N``.
        """
        if not self.conversation_store:
            return (
                "compile_knowledge from a conversation requires a wired "
                "ConversationStore. This usually means the chat server "
                "didn't pass one in."
            )
        conv_id, spec, err = self._parse_conversation_ref(source_ref)
        if err:
            return err

        try:
            session = self.conversation_store._read_session(conv_id)  # noqa: SLF001
        except Exception as e:
            return f"ERROR: could not load conversation '{conv_id}': {e}"
        turns = (session.get("turns") or []) if session else []
        if not turns:
            return (
                f"Conversation '{conv_id}' has no turns "
                f"(unknown id or empty thread)."
            )

        total = len(turns)
        if not spec:
            sliced = list(enumerate(turns))
        elif spec.startswith("last:"):
            n = int(spec.split(":", 1)[1])
            start = max(0, total - n)
            sliced = list(enumerate(turns))[start:]
        elif spec.startswith("range:"):
            _, a_str, b_str = spec.split(":")
            a, b = int(a_str), int(b_str)
            if a >= total:
                return (
                    f"turn:{a} is out of range — conversation '{conv_id}' "
                    f"only has {total} turn(s) (indices 0..{total - 1})."
                )
            b_clamped = min(b, total - 1)
            sliced = list(enumerate(turns))[a:b_clamped + 1]
        else:
            return f"Internal: unrecognised spec '{spec}'."

        if not sliced:
            return f"Selected range produced no turns from '{conv_id}'."

        # Build the synthetic source body so the rest of the compilation
        # pipeline (related search, slug suggestion, plan) has something
        # textual to chew on. Each turn is labelled with its absolute index
        # so citation strings stay stable even when we slice.
        body_lines: list[str] = []
        for idx, turn in sliced:
            role = (turn.get("role") or "?").strip()
            content_str = (turn.get("content") or "").strip()
            body_lines.append(f"### turn[{idx}] {role}")
            body_lines.append(content_str)
            body_lines.append("")
        body = "\n".join(body_lines).strip()
        body_tokens = estimate_tokens(body)

        # Derive query from the first user turn when none provided. Falling
        # back to the whole body keeps the embedder fed even if the user
        # turns are absent.
        if (query or "").strip():
            search_query = query.strip()
            derived_query = ""
        else:
            first_user = next(
                (
                    (t.get("content") or "").strip()
                    for _, t in sliced
                    if (t.get("role") or "") == "user"
                ),
                "",
            )
            derived_query = (first_user or body)[:500]
            search_query = derived_query

        # Run search — same pipeline as the file path, just inlined.
        try:
            results = self.kb_index.search(search_query, top_k=20) if self.kb_index else []
        except Exception as e:
            results = []
            log_event(
                dbg, "tool_error", tool="compile_knowledge",
                error=f"conversation search failed: {e}",
            )

        from knowledge.index import KBIndex
        wiki_hits: list[dict] = []
        canon_hits: list[dict] = []
        for r in results or []:
            r_filename = r.get("filename") or ""
            r_tier = r.get("tier") or KBIndex._compute_tier(
                r.get("source", "knowledge"), r_filename,
            )
            entry = {
                "filename": r_filename,
                "heading": r.get("heading", "") or "",
                "summary": r.get("summary", "") or "",
                "score": float(r.get("weighted_score", r.get("score", 0)) or 0),
                "tier": r_tier,
            }
            if r_tier == "wiki":
                wiki_hits.append(entry)
            elif r_tier == "canon":
                canon_hits.append(entry)

        def _dedupe_top(items: list[dict], top_n: int) -> list[dict]:
            best: dict[str, dict] = {}
            for it in items:
                fn = it["filename"]
                if fn not in best or it["score"] > best[fn]["score"]:
                    best[fn] = it
            return sorted(best.values(), key=lambda x: -x["score"])[:top_n]

        wiki_top = _dedupe_top(wiki_hits, self.COMPILE_RELATED_TOP_N)
        canon_top = _dedupe_top(canon_hits, self.COMPILE_RELATED_TOP_N)

        # Suggested slug from the first user turn (or first content turn).
        first_text = next(
            ((t.get("content") or "").strip() for _, t in sliced if (t.get("content") or "").strip()),
            "",
        )
        head_words = re.split(r"\s+", first_text)[:6]
        slug_seed = "-".join(w for w in head_words if w) or f"thread-{conv_id[:8]}"
        suggested_slug = re.sub(r"[^a-z0-9-]+", "-", slug_seed.lower()).strip("-")
        if not suggested_slug:
            suggested_slug = f"thread-{conv_id[:8]}"
        suggested_filename = f"{suggested_slug}.md"

        # Render
        out: list[str] = []
        out.append(
            f"# Compile Knowledge from conversation: {conv_id}"
        )
        out.append("")
        out.append("## Source [tier=memory] (chat-thread excerpt)")
        out.append(f"- conversation: {conv_id}")
        first_idx = sliced[0][0]
        last_idx = sliced[-1][0]
        if first_idx == last_idx:
            out.append(f"- turn: {first_idx}")
        else:
            out.append(f"- turns: {first_idx}..{last_idx} ({len(sliced)} of {total})")
        out.append(f"- size: ~{body_tokens:,} tokens")
        out.append("- excerpt:")
        for line in body.splitlines()[:40]:
            out.append(f"  {line}")
        if len(body.splitlines()) > 40:
            out.append("  ...")

        out.append("")
        out.append(
            f"## Related Wiki Pages (top {len(wiki_top)}) — link with [[wiki-link]]"
        )
        if wiki_top:
            for w in wiki_top:
                slug = Path(w["filename"]).stem
                summary_part = f" — {w['summary'][:120]}" if w["summary"] else ""
                out.append(
                    f"- [[{slug}]]  ({w['filename']}, score {w['score']:.2f})"
                    f"{summary_part}"
                )
        else:
            out.append(
                "  (no related wiki pages — this looks like an entirely new "
                "topic; consider also saving prerequisite concept pages.)"
            )

        out.append("")
        out.append(
            f"## Canon Anchors (top {len(canon_top)}) — cite as ground truth"
        )
        if canon_top:
            for c in canon_top:
                summary_part = f" — {c['summary'][:120]}" if c["summary"] else ""
                out.append(
                    f"- {c['filename']} > {c['heading']}  "
                    f"(score {c['score']:.2f}){summary_part}"
                )
        else:
            out.append("  (no canon anchors found for this query)")

        wiki_out = f"wiki/{suggested_filename}"
        out.extend(self._compile_canon_citation_examples(canon_top, wiki_out))

        out.append("")
        out.append("## Suggested Output")
        out.append(f"- filename: {wiki_out}")
        if derived_query:
            out.append(f"- derived search query: {derived_query[:200]}")

        out.append("")
        out.append("## Compilation Plan")
        out.append(
            "1. Reuse the conversation excerpt above directly — do NOT call "
            "search_knowledge to re-derive what you already have in context."
        )
        out.append(
            "2. Synthesize a focused, self-contained wiki page. One concept "
            "per page (LLM Wiki Pattern). Drop chat scaffolding, keep the "
            "decisions/conclusions."
        )
        if wiki_top:
            sample = ", ".join(
                f"[[{Path(w['filename']).stem}]]" for w in wiki_top[:3]
            )
            out.append(
                f"3. Link related wiki pages inline using [[wiki-link]] "
                f"syntax (e.g. {sample})."
            )
        else:
            out.append(
                "3. No existing wiki neighbors yet — record any prerequisite "
                "concepts as separate pages so the new node isn't an island."
            )
        if canon_top:
            out.append(
                "4. Cite canon using the **relative** markdown links from "
                "## Canon citation examples (``../../canon/...``). Do not put "
                "``canon:`` or ``knowledge:`` inside the link target parentheses."
            )
        else:
            out.append(
                "4. No canon anchors — flag any uncited claim as needing "
                "future canon material."
            )
        # Build the canonical Sources block the agent should paste verbatim.
        cite_lines = [
            f"- conversation:{conv_id}:turn:{idx}" for idx, _ in sliced[:8]
        ]
        if len(sliced) > 8:
            cite_lines.append(
                f"  (and {len(sliced) - 8} more turns from {conv_id})"
            )
        out.append(
            "5. Include a `## Sources` block with verifiable citations:"
        )
        for line in cite_lines:
            out.append(f"   {line}")
        out.append(
            f"6. save_knowledge(\"{suggested_filename}\", <compiled content>, "
            "[\"tag-one\", \"tag-two\"]) — writes to wiki/, triggers reindex, "
            "and auto-injects a Related-pages block. Pass tags as a **flat** "
            "array of strings. save_knowledge will REFUSE the write if any "
            "conversation citation in `## Sources` can't be verified."
        )
        return "\n".join(out)

    def folder_tree(self, folder: str = "") -> str:
        """Show the folder hierarchy across the medallion tiers.

        Default (no args, or folder='all'): renders both sources concatenated —
        canon (gold) first, then knowledge (silver wiki/ + bronze raw/), each
        annotated with [tier] badges per folder.

        Drill-down forms:
          - ``folder='canon'`` or ``'knowledge'`` — render that source only.
          - ``folder='canon/mind-en-place'`` — render the subtree rooted at
            ``mind-en-place`` inside canon. Useful when a top-level folder
            collapses 100+ files into one summary line.
          - ``folder='knowledge/raw/technology'`` — drill multiple levels
            inside knowledge.

        Returns indented folder tree with file counts and folder summaries
        (read from each folder's README.md / index.md / _index.md).
        """
        log_event(dbg, "tool_call", tool="folder_tree", folder=folder)
        if not self.kb_index:
            return "No knowledge index available."
        raw = (folder or "").strip()
        choice = raw.lower()

        # Top-level cases first — keep the original cheap path intact.
        if choice in ("", "all"):
            sources = [("canon", None), ("knowledge", None)]
        elif choice in ("knowledge", "canon"):
            sources = [(choice, None)]
        else:
            # Drill-down form: <source>/<sub/path>
            normalized = raw.replace("\\", "/").strip("/")
            head, _, tail = normalized.partition("/")
            head_lower = head.lower()
            if head_lower not in ("canon", "knowledge") or not tail:
                return (
                    f"Invalid folder '{folder}'. Use '' (all tiers, default), "
                    f"'canon', 'knowledge', or a drill-down path like "
                    f"'canon/mind-en-place' / 'knowledge/raw/technology'."
                )
            sources = [(head_lower, tail)]

        rendered: list[str] = []
        for source, root_path in sources:
            try:
                tree = self.kb_index.get_folder_tree(
                    source=source, root_path=root_path
                )
            except Exception as e:
                rendered.append(f"Error rendering '{source}' folder tree: {e}")
                continue
            if not tree:
                if root_path:
                    rendered.append(
                        f"No folder '{source}/{root_path}' in the graph. "
                        f"Call folder_tree('{source}') to see what exists."
                    )
                continue
            rendered.append(tree)
        if not rendered:
            return "No folders found in any tier."
        return "\n\n".join(rendered)

    # ------------------------------------------------------------------
    # B2: Conversation memory tools
    # ------------------------------------------------------------------

    @staticmethod
    def _embed_text_for_search(kb_index, text: str) -> list[float] | None:
        """Run the KBIndex's configured embedding fn on a single text.

        Used by the conversation memory tools so they share the embedding
        model with KB search instead of standing up a parallel pipeline.
        Returns ``None`` when no embedding function is configured.
        """
        if not kb_index or not getattr(kb_index, "_embedding_fn", None):
            return None
        try:
            vecs = kb_index._embedding_fn([text[:8000]])
        except Exception:
            return None
        if not vecs:
            return None
        v = vecs[0]
        return list(v) if v is not None else None

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        import math
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def search_conversations(self, query: str, limit: int = 5) -> str:
        """Semantic search across persisted chat threads.

        Walks every conversation under ``/app/sessions``, embeds turns that
        carry user/assistant prose (tool messages and empty turns are
        skipped), computes cosine similarity against the query embedding,
        and returns the top ``limit`` matches across all threads.

        Each hit shows the conversation id, turn index, role, timestamp, a
        160-char snippet, and the similarity score. Use the
        ``conversation_id`` + ``turn_index`` with ``read_conversation`` to
        load the surrounding context.

        Counts against the explore budget. Embeddings are cached on the
        session JSON file so repeat searches stay cheap.
        """
        log_event(dbg, "tool_call", tool="search_conversations", query=query, limit=limit)
        if not self.conversation_store:
            return (
                "search_conversations is not wired (no ConversationStore). "
                "This usually means the chat server didn't pass one in."
            )
        if not self.kb_index or not getattr(self.kb_index, "_embedding_fn", None):
            return (
                "search_conversations needs an embedding function but the KB "
                "index is not initialized. Try /kb/reindex first."
            )
        if not query or not str(query).strip():
            return "search_conversations: query is required."

        try:
            limit = max(1, min(int(limit), 20))
        except (TypeError, ValueError):
            limit = 5

        q_vec = self._embed_text_for_search(self.kb_index, query)
        if q_vec is None:
            return "search_conversations: failed to embed the query."

        # ------------------------------------------------------------------
        # Build a flat list of (score, conv_id, turn_idx, role, ts, snippet).
        # ------------------------------------------------------------------
        store = self.conversation_store
        try:
            convs_meta = store.list_conversations()
        except Exception as e:  # pragma: no cover - defensive
            return f"search_conversations: failed to list conversations: {e}"

        scored: list[tuple[float, str, int, str, str, str]] = []
        embed_writes: dict[str, dict] = {}

        for meta in convs_meta:
            conv_id = meta.get("id") or ""
            if not conv_id:
                continue
            try:
                session = store._read_session(conv_id)  # noqa: SLF001
            except Exception:
                continue
            turns = session.get("turns") or []
            if not turns:
                continue
            mutated = False
            for idx, turn in enumerate(turns):
                role = turn.get("role") or ""
                content = (turn.get("content") or "").strip()
                if role not in ("user", "assistant"):
                    continue
                if not content:
                    continue
                vec = turn.get("embedding") if isinstance(turn.get("embedding"), list) else None
                if vec is None:
                    vec = self._embed_text_for_search(self.kb_index, content)
                    if vec is None:
                        continue
                    turn["embedding"] = vec
                    mutated = True
                score = self._cosine(q_vec, vec)
                snippet_src = " ".join(content.split())
                snippet = snippet_src[:160] + ("..." if len(snippet_src) > 160 else "")
                scored.append((
                    float(score),
                    conv_id,
                    idx,
                    role,
                    turn.get("timestamp") or "",
                    snippet,
                ))
            if mutated:
                # Defer write so we don't thrash the disk inside the inner loop.
                embed_writes[conv_id] = session

        # Persist any newly-computed embeddings.
        for conv_id, session in embed_writes.items():
            try:
                store._write_session(conv_id, session)  # noqa: SLF001
            except Exception:
                pass  # cache write failure is non-fatal

        if not scored:
            return "search_conversations: no matching turns across persisted threads."

        scored.sort(key=lambda r: r[0], reverse=True)
        top = scored[:limit]

        out_lines = [f"# Conversation matches for: {query!r} (top {len(top)})"]
        for score, conv_id, idx, role, ts, snippet in top:
            out_lines.append(
                f"- [{score:.3f}] conv={conv_id} turn={idx} role={role} ts={ts}\n"
                f"  > {snippet}"
            )
        return "\n".join(out_lines)

    def read_conversation(
        self,
        conversation_id: str,
        turn_range: str = "last:20",
    ) -> str:
        """Load turns from a persisted conversation thread.

        ``turn_range`` accepts:
          - ``"last:N"`` — the most recent N turns (default ``last:20``)
          - ``"range:A:B"`` — turns at indices A..B inclusive (0-based)

        Returns role-labeled markdown with timestamps. Caps the total
        rendered text at ~6000 tokens to protect the context window;
        truncated turns are marked with ``[truncated]``.

        Counts against the explore budget. Use ``search_conversations``
        first to find a relevant ``conversation_id`` + turn index.
        """
        log_event(
            dbg, "tool_call", tool="read_conversation",
            conversation_id=conversation_id, turn_range=turn_range,
        )
        if not self.conversation_store:
            return (
                "read_conversation is not wired (no ConversationStore). "
                "This usually means the chat server didn't pass one in."
            )
        if not conversation_id:
            return "read_conversation: conversation_id is required."

        store = self.conversation_store
        try:
            session = store._read_session(conversation_id)  # noqa: SLF001
        except Exception as e:
            return f"read_conversation: failed to read conversation: {e}"
        turns = session.get("turns") or []
        if not turns:
            return f"read_conversation: conversation {conversation_id} has no turns."

        spec = (turn_range or "").strip().lower()
        if spec.startswith("last:"):
            try:
                n = max(1, int(spec.split(":", 1)[1]))
            except (ValueError, IndexError):
                return "read_conversation: invalid 'last:N' spec."
            slice_ = list(enumerate(turns))[-n:]
        elif spec.startswith("range:"):
            try:
                _, start_s, end_s = spec.split(":")
                start = max(0, int(start_s))
                end = min(len(turns) - 1, int(end_s))
            except (ValueError, IndexError):
                return "read_conversation: invalid 'range:A:B' spec."
            if start > end:
                return f"read_conversation: range start {start} > end {end}."
            slice_ = list(enumerate(turns[start:end + 1], start=start))
        else:
            return (
                "read_conversation: turn_range must be 'last:N' or "
                "'range:A:B' (e.g. 'last:10', 'range:3:7')."
            )

        from agent.tokenizer import count_tokens

        title = session.get("title", "Untitled")
        out: list[str] = [
            f"# Conversation {conversation_id} — {title}",
            f"({len(turns)} total turns; showing {len(slice_)})",
        ]
        budget_tokens = 6000
        used = 0
        for idx, turn in slice_:
            role = turn.get("role", "?")
            ts = turn.get("timestamp", "")
            content = (turn.get("content") or "").strip()
            cost = count_tokens(content)
            if used + cost > budget_tokens and used > 0:
                out.append(f"\n## turn[{idx}] {role} {ts}\n[truncated: budget exhausted]")
                break
            if used + cost > budget_tokens:
                # Single oversized turn — render a head slice rather than nothing.
                head = content[:4000]
                out.append(
                    f"\n## turn[{idx}] {role} {ts}\n{head}\n[truncated: turn too large]"
                )
                used = budget_tokens
                continue
            out.append(f"\n## turn[{idx}] {role} {ts}\n{content}")
            used += cost
        return "\n".join(out)

    def _read_file_content(self, filename: str) -> str | None:
        """Read raw file content from knowledge/ or canon/."""
        path = self.kb_dir / filename
        if not path.exists():
            path = self.canon_dir / filename
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None


def reset_budget():
    """Reset budget counters for a new request. Called by streaming loop."""
    global _current_kb_loads, _current_available_tokens, _current_tool_tokens_used
    _current_kb_loads = 0
    _current_available_tokens = 999999
    _current_tool_tokens_used = 0


def set_available_tokens(tokens: int):
    """Set the available context tokens for budget enforcement."""
    global _current_available_tokens
    _current_available_tokens = max(0, int(tokens))


def set_context_window(tokens: int):
    """Set the active model's context window (used to compute tool budget cap)."""
    global _current_context_window
    _current_context_window = max(1024, int(tokens))


def get_budget_state() -> dict:
    """Return a snapshot of the current per-request budget. Useful for logging."""
    return {
        "kb_loads": _current_kb_loads,
        "kb_loads_max": _KB_MAX_LOADS_PER_RESPONSE,
        "available_tokens": _current_available_tokens,
        "context_window": _current_context_window,
        "tool_tokens_used": _current_tool_tokens_used,
        "tool_token_cap": _tool_token_cap(),
    }


# ---------------------------------------------------------------------------
# Native tool-calling registry (A1 + A3)
# ---------------------------------------------------------------------------
#
# Each KBTools method maps to a tool *class* with its own per-turn budget.
# The chat loop counts executions per class and refuses further calls in a
# class once the budget is spent — without starving other classes. This
# replaces the old monolithic MAX_TOTAL_TOOL_EXECUTIONS that let exploration
# tools eat the entire budget before the model could write anything.
#
# Class semantics:
#   - explore     : read-only / discovery (search, graph, read, list)
#   - write       : creates or stages KB content (save / compile)
#   - maintenance : audits and integrity checks (lint)

TOOL_CLASSES: dict[str, str] = {
    # P1-1: cheap orientation tools moved to their own ``orient`` class so
    # 3-4 cheap "where am I" calls don't burn the explore budget that the
    # heavy tools (read_knowledge_section, graph_traverse, etc.) need.
    "list_knowledge": "orient",
    "folder_tree": "orient",
    "graph_stats": "orient",
    "read_knowledge": "explore",
    "read_knowledge_section": "explore",
    "search_knowledge": "explore",
    "graph_neighbors": "explore",
    "graph_traverse": "explore",
    "graph_search": "explore",
    "describe_node": "explore",
    "search_conversations": "explore",
    "read_conversation": "explore",
    "save_knowledge": "write",
    "compile_knowledge": "write",
    "lint_knowledge": "maintenance",
}

CLASS_BUDGETS: dict[str, int] = {
    # P1-1: orient is a separate small budget for cheap orientation tools
    # (graph_stats, list_knowledge, folder_tree). Keeping it independent of
    # explore means a tour can call all three plus still have a full
    # explore budget for read_knowledge_section / graph_traverse / etc.
    "orient": 5,
    # R13: raised 8 -> 10 so a single deep research turn can load ~4 sections,
    # run 2 graph_traverse calls, 2 describe_node calls, a search_knowledge,
    # and a search_conversations without hitting the refusal. Real sessions
    # were stalling at 8 when the model actually needed to cross-reference.
    "explore": 10,
    "write": 2,
    "maintenance": 3,
}


def build_tool_registry(kb_tools: "KBTools") -> dict[str, callable]:
    """Build a {tool_name: callable} registry exposed to the model.

    The Ollama Python SDK reads each callable's signature + Google-style
    docstring to auto-generate a JSON schema, so we hand it the bound
    methods directly. Dispatch in the chat loop is purely by name.
    """
    registry: dict[str, callable] = {
        "list_knowledge": kb_tools.list_knowledge,
        "read_knowledge": kb_tools.read_knowledge,
        "read_knowledge_section": kb_tools.read_knowledge_section,
        "search_knowledge": kb_tools.search_knowledge,
        "save_knowledge": kb_tools.save_knowledge,
        "graph_neighbors": kb_tools.graph_neighbors,
        "graph_traverse": kb_tools.graph_traverse,
        "graph_search": kb_tools.graph_search,
        "graph_stats": kb_tools.graph_stats,
        "describe_node": kb_tools.describe_node,
        "folder_tree": kb_tools.folder_tree,
        "lint_knowledge": kb_tools.lint_knowledge,
        "compile_knowledge": kb_tools.compile_knowledge,
    }
    # B2: only expose conversation tools when the store is wired so the
    # model never sees a tool it can't actually invoke.
    if getattr(kb_tools, "conversation_store", None) is not None:
        registry["search_conversations"] = kb_tools.search_conversations
        registry["read_conversation"] = kb_tools.read_conversation
    return registry


def class_for_tool(name: str) -> str:
    """Return the budget class for a tool name, or ``"explore"`` if unknown.

    Unknown tools are treated as exploration so a typo never silently bypasses
    the write budget.
    """
    return TOOL_CLASSES.get(name, "explore")