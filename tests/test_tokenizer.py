"""Tests for token counting and truncation utilities."""

import pytest
from agent.tokenizer import (
    count_tokens,
    estimate_tokens,
    slice_tokens,
    truncate_at_sentence_boundary,
    truncate_to_tokens,
)


class TestCountTokens:
    """Test raw cl100k_base token counting."""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_none_like_empty(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        count = count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("hi")
        long = count_tokens("This is a much longer sentence with many more words.")
        assert long > short

    def test_deterministic(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert count_tokens(text) == count_tokens(text)


class TestEstimateTokens:
    """Test model token estimation with 1.5x multiplier."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_multiplier_applied(self):
        text = "Hello, world!"
        raw = count_tokens(text)
        estimated = estimate_tokens(text)
        assert estimated == int(raw * 1.5)

    def test_estimate_always_gte_raw(self):
        text = "Some text for counting."
        assert estimate_tokens(text) >= count_tokens(text)


class TestTruncateToTokens:
    """Test token-bounded truncation."""

    def test_no_truncation_needed(self):
        text = "Short text"
        result, was_truncated = truncate_to_tokens(text, 100)
        assert result == text
        assert was_truncated is False

    def test_truncation_happens(self):
        text = "This is a sentence that should be truncated."
        result, was_truncated = truncate_to_tokens(text, 3)
        assert was_truncated is True
        assert count_tokens(result) <= 3

    def test_truncated_result_is_valid_text(self):
        text = "Hello world, this is a test of truncation behavior."
        result, _ = truncate_to_tokens(text, 5)
        # Should be a valid string, no partial tokens
        assert isinstance(result, str)
        assert len(result) > 0

    def test_exact_fit(self):
        text = "Hello world"
        token_count = count_tokens(text)
        result, was_truncated = truncate_to_tokens(text, token_count)
        assert result == text
        assert was_truncated is False


class TestSliceTokens:
    """Test offset-based token slicing for read_knowledge_section continuation."""

    def test_offset_zero_returns_prefix(self):
        text = "one two three four five six seven eight nine ten"
        sliced, total, more = slice_tokens(text, 0, 3)
        assert sliced  # non-empty
        assert total > 0
        assert more is True

    def test_offset_past_end_returns_empty(self):
        text = "short"
        total = count_tokens(text)
        sliced, returned_total, more = slice_tokens(text, total + 5, 10)
        assert sliced == ""
        assert returned_total == total
        assert more is False

    def test_full_slice_no_more(self):
        text = "one two three"
        total = count_tokens(text)
        sliced, _, more = slice_tokens(text, 0, total)
        assert more is False

    def test_round_trip_offsets(self):
        """Concatenating sequential slices reproduces the input (token-wise)."""
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        total = count_tokens(text)
        a, _, _ = slice_tokens(text, 0, 3)
        b, _, _ = slice_tokens(text, 3, total)
        assert count_tokens(a) + count_tokens(b) == total


class TestTruncateAtSentenceBoundary:
    """Test sentence-boundary truncation that never cuts mid-word."""

    def test_no_truncation_when_short(self):
        text = "Short sentence."
        out, truncated = truncate_at_sentence_boundary(text, 1000)
        assert out == text
        assert truncated is False

    def test_paragraph_boundary_preferred(self):
        """When a paragraph break is reachable in the budget, prefer it."""
        text = "First paragraph here is reasonably long.\n\nSecond paragraph runs longer than the cap."
        out, truncated = truncate_at_sentence_boundary(text, 50)
        assert truncated is True
        assert out.endswith("First paragraph here is reasonably long.")

    def test_sentence_boundary_used(self):
        text = "Hello world. This is fine. Now this would overflow our budget."
        out, truncated = truncate_at_sentence_boundary(text, 30)
        assert truncated is True
        # Must end at a sentence terminator, not mid-word.
        assert out.endswith(".") or out.endswith("!") or out.endswith("?")

    def test_never_cuts_mid_word(self):
        """When there's a reachable space within ~2x budget, prefer it over mid-word cut."""
        text = "supercalifragilisticexpialidocious is one extraordinarily long word."
        out, truncated = truncate_at_sentence_boundary(text, 20)
        assert truncated is True
        # Should end at the space after the long word, not in the middle of it.
        assert out == "supercalifragilisticexpialidocious"