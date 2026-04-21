"""Section-based chunking for knowledge base files.

Splits markdown files on heading boundaries for indexing and token-aware
retrieval. Detects the highest heading level present (H1-H5) and splits
on that. Handles recursive splitting for oversized sections and hard
splits when no sub-headings exist.

Also provides heading tree construction -- a hierarchical view of all
headings with cumulative token counts per subtree. Used by read_knowledge
to give the agent a structural map of oversized files.

Ported from agent-zero/knowledge/chunker.py. Self-contained, no
agent-zero imports.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.tokenizer import count_tokens


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a file into frontmatter dict and remaining content.

    Simple YAML frontmatter parser. Only handles flat key: value pairs.
    Returns (metadata_dict, body_text).
    """
    if not text.startswith("---"):
        return {}, text

    lines = text[3:].split("\n")
    metadata = {}
    end_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            end_idx = i
            break
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()

    body = "\n".join(lines[end_idx + 1:])
    return metadata, body


@dataclass
class HeadingNode:
    """A node in the heading tree. Represents one markdown heading."""
    level: int                                  # 1-5 (or 0 for root)
    heading: str                                # heading text without # prefix
    own_tokens: int = 0                         # tokens in this node's direct content
    subtree_tokens: int = 0                    # own_tokens + all children's subtree_tokens
    children: list["HeadingNode"] = field(default_factory=list)
    summary: str = ""                           # LLM-generated summary (populated from index)


def _detect_heading_level(body: str) -> int | None:
    """Find the highest (lowest number) heading level in the body.

    Scans for H1 through H5. Returns None if no headings found.
    """
    for level in range(1, 6):
        pattern = rf"^{'#' * level}\s+\S"
        if re.search(pattern, body, re.MULTILINE):
            return level
    return None


def _sanitize_heading_text(heading: str) -> tuple[str, str]:
    """Defense-in-depth: split a captured heading on literal escape sequences.

    P1.2: when malformed content makes it through to the chunker (entire
    file body collapsed onto the heading line because `\\n` was never
    decoded), don't let the whole body become the heading. Take whatever
    is before the first literal `\\n` / `\\r` / `\\t` as the heading and
    return the rest as bonus body content.

    Returns (clean_heading, leaked_body). leaked_body is "" when the
    heading was already clean.
    """
    for token in ("\\n", "\\r", "\\t"):
        idx = heading.find(token)
        if idx >= 0:
            head = heading[:idx].rstrip()
            tail = heading[idx:]
            tail = (
                tail.replace("\\n", "\n")
                    .replace("\\r", "\n")
                    .replace("\\t", "    ")
            )
            return (head or "(unknown)", tail.lstrip())
    return (heading, "")


def _split_on_level(body: str, level: int) -> list[tuple[str, str]]:
    """Split body into (heading, content) pairs at the given heading level.

    Content before the first heading at this level is captured with
    heading="(preamble)". The heading prefix (###) is stripped from
    the heading text. Divider lines (---) immediately before headings
    are excluded from the preceding section's content.
    """
    prefix = "#" * level
    pattern = rf"^{prefix}\s+(.+)$"
    sections: list[tuple[str, str]] = []
    current_heading = "(preamble)"
    current_lines: list[str] = []

    for line in body.split("\n"):
        match = re.match(pattern, line)
        if match:
            content = "\n".join(current_lines).rstrip()
            if content.endswith("---"):
                content = content[:-3].rstrip()
            if current_lines or sections:
                sections.append((current_heading, content))
            raw_heading = match.group(1).strip()
            cleaned, leaked = _sanitize_heading_text(raw_heading)
            current_heading = cleaned
            current_lines = leaked.split("\n") if leaked else []
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).rstrip()
    sections.append((current_heading, content))

    return sections


def _hard_split(text: str, max_tokens: int, overlap: int = 50) -> list[str]:
    """Split text into token-bounded chunks with overlap.

    Used as a last resort when no sub-headings exist and the section
    exceeds the token budget.
    """
    from agent.tokenizer import _get_encoder
    enc = _get_encoder()
    tokens = enc.encode(text)
    # Ensure overlap is smaller than max_tokens to avoid infinite loops
    step = max(1, max_tokens - overlap)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start += step
    return chunks


def build_heading_tree(text: str, filename: str) -> HeadingNode:
    """Build a hierarchical heading tree from a markdown file.

    Parses all H1-H5 headings and organizes them into a tree where each
    node's subtree_tokens includes its own content plus all descendants.
    The root node (level 0) represents the file itself.

    Args:
        text: Raw file content (including frontmatter).
        filename: Used as the root node heading.

    Returns:
        HeadingNode tree with computed subtree_tokens at every level.
    """
    _, body = _parse_frontmatter(text)
    body = body.strip()

    root = HeadingNode(level=0, heading=Path(filename).stem)

    if not body:
        return root

    # Parse all headings and their content spans
    heading_pattern = re.compile(r"^(#{1,5})\s+(.+)$", re.MULTILINE)
    entries: list[tuple[int, str, str]] = []  # (level, heading, content)

    matches = list(heading_pattern.finditer(body))

    if not matches:
        # No headings -- all content belongs to root
        root.own_tokens = count_tokens(body)
        root.subtree_tokens = root.own_tokens
        return root

    # Content before first heading = root's own content
    preamble = body[:matches[0].start()].strip()
    if preamble:
        root.own_tokens = count_tokens(preamble)

    for i, match in enumerate(matches):
        level = len(match.group(1))
        raw_heading = match.group(2).strip()
        # P1.2: split heading on literal escape sequences so a body that
        # leaked into the heading line doesn't become a 1,000-token title.
        heading, leaked_body = _sanitize_heading_text(raw_heading)
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[content_start:content_end].strip()
        if leaked_body:
            content = (leaked_body.strip() + ("\n\n" + content if content else "")).strip()
        if content.endswith("---"):
            content = content[:-3].rstrip()
        entries.append((level, heading, content))

    # Build tree using a stack-based approach
    stack: list[HeadingNode] = [root]

    for level, heading, content in entries:
        node = HeadingNode(
            level=level,
            heading=heading,
            own_tokens=count_tokens(content) if content else 0,
        )
        # Pop stack until we find a parent with a lower level
        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)

    # Compute subtree_tokens bottom-up
    _compute_subtree_tokens(root)

    return root


def _compute_subtree_tokens(node: HeadingNode) -> int:
    """Recursively compute subtree_tokens for a node and all descendants."""
    child_total = sum(_compute_subtree_tokens(c) for c in node.children)
    node.subtree_tokens = node.own_tokens + child_total
    return node.subtree_tokens


def format_heading_tree(root: HeadingNode, indent: int = 0) -> str:
    """Render a heading tree as indented text for LLM consumption.

    Output looks like:
        dim-modeling-guide (2,450 tokens)
          ## Overview (320 tokens) -- Introduces dimensional modeling concepts
          ## Star Schema (1,200 tokens) -- Covers star schema design patterns
            ### Fact Tables (500 tokens) -- Fact table grain and measures
            ### Dimension Tables (700 tokens) -- Conformed dimension design
          ## Snowflake Schema (930 tokens) -- Normalized variant of star schema
    """
    lines: list[str] = []
    prefix = "  " * indent

    if root.level == 0:
        # Root node -- show filename
        root_line = f"{root.heading} ({root.subtree_tokens:,} tokens)"
        if root.summary:
            root_line += f" -- {root.summary}"
        lines.append(root_line)
        for child in root.children:
            lines.append(format_heading_tree(child, indent + 1))
    else:
        hashes = "#" * root.level
        tok = root.subtree_tokens if root.children else root.own_tokens
        node_line = f"{prefix}{hashes} {root.heading} ({tok:,} tokens)"
        if root.summary:
            node_line += f" -- {root.summary}"
        lines.append(node_line)
        if root.children:
            for child in root.children:
                lines.append(format_heading_tree(child, indent + 1))

    return "\n".join(lines)


def enrich_tree_summaries(
    node: HeadingNode,
    summaries: dict[str, str],
) -> None:
    """Attach summaries from the KB index to heading tree nodes.

    Walks the tree and sets each node's summary field from the
    heading -> summary mapping. Matching is case-insensitive.
    Modifies the tree in place.
    """
    # Build lowercase lookup for case-insensitive matching
    lower_map = {k.lower(): v for k, v in summaries.items()}

    def _walk(n: HeadingNode) -> None:
        key = n.heading.lower()
        if key in lower_map:
            n.summary = lower_map[key]
        for child in n.children:
            _walk(child)

    _walk(node)


def _chunk_sections(
    sections: list[tuple[str, str]],
    level: int,
    max_tokens: int | None,
) -> list[dict]:
    """Convert (heading, content) pairs into chunk dicts.

    If max_tokens is set and a section exceeds it, try to split on the
    next heading level down. If that fails, hard-split at token boundaries.
    """
    chunks = []

    for heading, content in sections:
        # Skip empty preamble sections
        if heading == "(preamble)" and not content.strip():
            continue

        full_text = content if heading == "(preamble)" else f"{'#' * level} {heading}\n\n{content}"
        tok_count = count_tokens(full_text)

        if max_tokens is None or tok_count <= max_tokens:
            chunks.append({
                "heading": heading,
                "content": full_text,
                "token_count": tok_count,
            })
        else:
            # Try splitting on next heading level
            sub_level = _detect_heading_level(content)
            if sub_level and sub_level > level:
                sub_sections = _split_on_level(content, sub_level)
                sub_chunks = _chunk_sections(sub_sections, sub_level, max_tokens)
                # Prefix sub-chunk headings with parent heading for context
                for sc in sub_chunks:
                    if sc["heading"] != "(preamble)":
                        sc["heading"] = f"{heading} > {sc['heading']}"
                    else:
                        sc["heading"] = heading
                chunks.extend(sub_chunks)
            else:
                # No sub-headings -- hard split
                parts = _hard_split(full_text, max_tokens)
                for i, part in enumerate(parts):
                    suffix = f" (part {i + 1}/{len(parts)})" if len(parts) > 1 else ""
                    chunks.append({
                        "heading": f"{heading}{suffix}",
                        "content": part,
                        "token_count": count_tokens(part),
                    })

    return chunks


def _doc_summary(text: str, max_chars: int = 150) -> str:
    """Extract first meaningful line from file body for context headers."""
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped == "---":
            continue
        if stripped.startswith("[") and "](#" in stripped:
            continue
        if stripped.startswith("|"):
            continue
        if len(stripped) > max_chars:
            return stripped[:max_chars] + "..."
        return stripped
    return "(no summary)"


def _build_context_header(
    filename: str,
    position: int,
    total: int,
    heading: str,
    doc_summary: str,
) -> str:
    """Build a compact YAML-like context header for a chunk.

    Includes file, position, ancestors (heading path minus own heading),
    and doc summary. Prepended to chunk content before embedding so
    vector search can find parent context.
    """
    lines = ["---"]
    lines.append(f"file: {filename}")
    lines.append(f"position: {position}/{total}")

    # Extract ancestors from heading path: "H1 > H2 > H3" → ancestors="H1 > H2"
    if " > " in heading:
        parts = heading.split(" > ")
        ancestors = " > ".join(parts[:-1])
        lines.append(f"ancestors: {ancestors}")

    if doc_summary and doc_summary != "(no summary)":
        lines.append(f"doc_summary: {doc_summary}")

    lines.append("---")
    return "\n".join(lines)


def chunk_file(
    text: str,
    filename: str,
    *,
    max_tokens: int | None = None,
) -> list[dict]:
    """Chunk a knowledge file into H1-level concept sections for indexing.

    H1 headings are first-class concept boundaries -- each H1 block becomes
    one chunk.  H2-H5 headings are children of their parent H1 and remain
    inside that chunk.  When max_tokens is set and an H1 chunk exceeds it,
    the chunk is recursively split at H2 (then H3, etc.), then hard-split
    at token boundaries if no sub-headings exist.

    Files with no H1 fall back to H2 as the primary split level, then H3, etc.
    Files with no headings at all produce a single chunk.

    Args:
        text: Raw file content (including frontmatter).
        filename: The filename (used as fallback heading for headingless files).
        max_tokens: Optional per-chunk token limit.

    Returns list of dicts, each with:
        heading: str -- H1 section heading (or composite path for sub-splits)
        content: str -- section text
        chunk_index: int -- 0-based position in the file
        token_count: int -- token count of the content
    """
    _, body = _parse_frontmatter(text)
    body = body.strip()

    if not body:
        return []

    # Compute file-level doc summary for context headers
    doc_summ = _doc_summary(body)
    stem = Path(filename).stem

    level = _detect_heading_level(body)

    if level is None:
        # No headings -- single chunk
        tok_count = count_tokens(body)
        if max_tokens and tok_count > max_tokens:
            parts = _hard_split(body, max_tokens)
            total = len(parts)
            return [
                {
                    "heading": stem,
                    "content": part,
                    "chunk_index": i,
                    "token_count": count_tokens(part),
                    "filename": filename,
                    "total_chunks": total,
                    "doc_summary": doc_summ,
                }
                for i, part in enumerate(parts)
            ]
        return [{
            "heading": stem,
            "content": body,
            "chunk_index": 0,
            "token_count": tok_count,
            "filename": filename,
            "total_chunks": 1,
            "doc_summary": doc_summ,
        }]

    sections = _split_on_level(body, level)
    chunks = _chunk_sections(sections, level, max_tokens)

    # Assign chunk indices and file-level context
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i
        chunk["filename"] = filename
        chunk["total_chunks"] = total
        chunk["doc_summary"] = doc_summ

    return chunks