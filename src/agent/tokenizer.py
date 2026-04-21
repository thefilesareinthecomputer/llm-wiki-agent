"""Token counting and truncation utilities.

Uses tiktoken cl100k_base as a general-purpose tokenizer. Raw counts are
cl100k_base tokens with no adjustments. For context budget estimation
(where cl100k_base undercounts relative to Gemma/Qwen by ~1.5x), use
estimate_tokens() instead.
"""

import tiktoken

_encoder: tiktoken.Encoding | None = None

# cl100k_base undercounts relative to Gemma/Qwen tokenizers by ~1.5x.
# Only applied in estimate_tokens(), not in count_tokens().
_MODEL_MULTIPLIER = 1.5


def _get_encoder() -> tiktoken.Encoding:
    """Lazy-load the tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Return the cl100k_base token count for a string.

    Raw count, no multipliers. Used by chunker, heading trees,
    KB index metadata, and embedding size limits.
    """
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def estimate_tokens(text: str) -> int:
    """Estimate model token count from cl100k_base.

    Applies a 1.5x multiplier to approximate Gemma/Qwen tokenization.
    Use this ONLY for context budget calculations shown to the agent
    (chat streaming, tools). Everything else uses count_tokens().
    """
    if not text:
        return 0
    raw = len(_get_encoder().encode(text))
    return int(raw * _MODEL_MULTIPLIER)


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate text to fit within a token budget.

    max_tokens is in cl100k_base tokens (raw count).

    Returns (text, was_truncated). Truncation happens at token boundaries,
    not character boundaries, so no partial tokens appear in output.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    truncated = enc.decode(tokens[:max_tokens])
    return truncated, True


def slice_tokens(text: str, offset: int, max_tokens: int) -> tuple[str, int, bool]:
    """Slice a text by token offset and length.

    offset and max_tokens are in cl100k_base tokens (raw count).

    Returns (sliced_text, total_tokens_in_input, has_more_after).
    Used by read_knowledge_section to support offset-based continuation
    of large sections.
    """
    enc = _get_encoder()
    tokens = enc.encode(text)
    total = len(tokens)
    if offset >= total:
        return "", total, False
    end = min(offset + max_tokens, total)
    sliced = enc.decode(tokens[offset:end])
    return sliced, total, end < total


def truncate_at_sentence_boundary(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate text at the nearest paragraph/sentence/word boundary
    that does not exceed max_chars. Never cuts mid-word.

    Strategy (in order of preference):
      1. last paragraph break (\n\n) past 50% of max
      2. last sentence terminator (. ! ?) past 50% of max
      3. last newline past 50% of max
      4. last whitespace past 50% of max
      5. hard cut at max_chars

    Returns (truncated_text, was_truncated).
    """
    if len(text) <= max_chars:
        return text, False

    snippet = text[:max_chars]
    # Quarter floor prevents pathological cases where the only boundary is
    # very close to the start (we'd rather truncate harder than return ~5%).
    floor = max_chars // 4

    last_para = snippet.rfind("\n\n")
    if last_para >= floor:
        return snippet[:last_para].rstrip(), True

    for punct in (". ", "? ", "! ", ".\n", "?\n", "!\n"):
        idx = snippet.rfind(punct)
        if idx >= floor:
            return snippet[: idx + 1].rstrip(), True

    last_nl = snippet.rfind("\n")
    if last_nl >= floor:
        return snippet[:last_nl].rstrip(), True

    last_sp = snippet.rfind(" ")
    if last_sp >= floor:
        return snippet[:last_sp].rstrip(), True

    # No good word boundary found within the budget. Avoid the mid-word
    # cut by walking forward to the next space (may slightly exceed the
    # char budget, but the model never sees a half-word).
    next_sp = text.find(" ", max_chars)
    if next_sp != -1 and next_sp - max_chars < max_chars:  # within 2x budget
        return text[:next_sp].rstrip(), True

    return snippet, True