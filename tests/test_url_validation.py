"""R12: save_knowledge refuses content with placeholder URLs.

The model sometimes invents URLs that syntactically look real
(`https://example.com/stoicism-study`) but resolve to nothing. This
suite pins the syntactic blocklist that aborts those saves before they
hit disk, along with the shape of the refusal message so the model can
actually recover from it.

Design notes:
- Pure function tests on `_validate_urls` — no disk, no gateway, no
  network. That keeps the suite fast and isolates from KB index setup.
- `save_knowledge` call-site tests live in the integration suite and
  should already cover the refusal-vs-success propagation. Tests here
  are focused on correctness of the validator itself.
"""

from __future__ import annotations

import pytest

from agent.tools import KBTools


@pytest.fixture
def tools() -> KBTools:
    """A KBTools instance is only needed for method dispatch; the
    validator is stateless so no index / store wiring is required."""
    return KBTools.__new__(KBTools)


class TestURLPlaceholderRejection:
    """Placeholder hosts and substring patterns must refuse."""

    def test_example_com_refused(self, tools):
        err = tools._validate_urls(
            "See [more](https://example.com/article) for details."
        )
        assert err is not None
        assert "example.com" in err
        assert "placeholder" in err.lower() or "blocklist" in err.lower()

    def test_example_org_refused(self, tools):
        err = tools._validate_urls("Visit https://example.org for info.")
        assert err is not None

    def test_relevant_study_substring_refused(self, tools):
        err = tools._validate_urls(
            "Source: https://relevant-study-42.com/paper.pdf"
        )
        assert err is not None
        assert "relevant-study" in err

    def test_placeholder_substring_refused(self, tools):
        err = tools._validate_urls(
            "A [great doc](https://placeholder-url.io/doc)"
        )
        assert err is not None
        assert "placeholder" in err

    def test_your_domain_refused(self, tools):
        err = tools._validate_urls(
            "Docs: https://your-domain.com/docs"
        )
        assert err is not None

    def test_subdomain_of_placeholder_refused(self, tools):
        """A subdomain under `example.com` is still fake — host match
        is on the effective hostname, not the trailing path."""
        err = tools._validate_urls(
            "See https://blog.example.com/post-1 for discussion."
        )
        assert err is not None


class TestURLsAcceptedUnchanged:
    """Real-looking URLs, wiki links, relative paths, anchors must pass."""

    def test_real_url_passes(self, tools):
        assert tools._validate_urls(
            "See [Stoicism](https://en.wikipedia.org/wiki/Stoicism) "
            "for the encyclopedic entry."
        ) is None

    def test_multiple_real_urls_pass(self, tools):
        text = (
            "Primary: https://plato.stanford.edu/entries/stoicism/\n"
            "Secondary: https://iep.utm.edu/stoicism/"
        )
        assert tools._validate_urls(text) is None

    def test_wiki_link_ignored(self, tools):
        """Obsidian `[[wiki-link]]` syntax isn't a URL and must not
        be scanned — otherwise the conversion layer can't do its job."""
        assert tools._validate_urls(
            "Related: [[cortisol]] and [[hpa-axis]] deepen this."
        ) is None

    def test_relative_canon_path_ignored(self, tools):
        """`../../canon/...` is the canonical in-vault link shape and
        must never be rejected by the URL validator."""
        assert tools._validate_urls(
            "See [Meditations](../../canon/mind/meditations.md) for source."
        ) is None

    def test_anchor_only_ignored(self, tools):
        """Fragment-only links (`#section`) aren't URLs at all."""
        assert tools._validate_urls(
            "Jump to [the conclusion](#conclusion) for the TL;DR."
        ) is None

    def test_empty_content_passes(self, tools):
        assert tools._validate_urls("") is None
        assert tools._validate_urls("   \n\n  ") is None

    def test_content_without_urls_passes(self, tools):
        assert tools._validate_urls(
            "A paragraph of text with no links at all."
        ) is None

    def test_trailing_punctuation_not_treated_as_host(self, tools):
        """URLs ending with a period or closing paren from surrounding
        prose are trimmed before host extraction — a legitimate URL
        followed by a sentence-ending period must not be flagged."""
        assert tools._validate_urls(
            "Visit https://en.wikipedia.org/wiki/Stoicism. It's a start."
        ) is None


class TestRefusalMessageShape:
    """The model needs a structured, actionable error so it can retry
    instead of looping. Pin the shape."""

    def test_refusal_lists_every_bad_url(self, tools):
        err = tools._validate_urls(
            "First: https://example.com/a\n"
            "Second: https://placeholder.org/b\n"
            "Third: https://relevant-paper.com/c"
        )
        assert err is not None
        assert "example.com" in err
        assert "placeholder.org" in err
        assert "relevant-paper.com" in err

    def test_refusal_tells_model_how_to_recover(self, tools):
        err = tools._validate_urls(
            "Source: https://example.com/data"
        )
        assert err is not None
        assert "[[wiki-link]]" in err or "wiki-link" in err
        assert "fabricate" in err.lower() or "invented" in err.lower()

    def test_refusal_starts_with_save_knowledge_refusing(self, tools):
        err = tools._validate_urls("See https://example.com")
        assert err is not None
        assert err.startswith("save_knowledge: refusing to write")

    def test_duplicate_bad_url_listed_once(self, tools):
        """If the same bad URL appears twice in content, the refusal
        should dedupe it — otherwise the error balloons on long pages."""
        err = tools._validate_urls(
            "See https://example.com/a and also https://example.com/a "
            "which is the same link twice."
        )
        assert err is not None
        assert err.count("https://example.com/a") == 1
