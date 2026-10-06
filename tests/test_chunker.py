"""Tests for the chunking stage."""

from __future__ import annotations

import pytest

from site_rag_analyst.chunking.chunker import Chunker
from site_rag_analyst.models import ExtractedPage


def make_page(text: str, url: str = "https://example.com/pricing") -> ExtractedPage:
    return ExtractedPage(
        url=url, title="Pricing — Example", text=text, word_count=len(text.split())
    )


class TestBasicBehavior:
    def test_small_page_single_chunk(self) -> None:
        page = make_page("## Pricing\nPlans start at 19 dollars per user monthly.")
        chunks = Chunker().chunk_page(page, page_index=0)
        assert len(chunks) == 1
        assert chunks[0].id == "p00-c0"
        assert chunks[0].word_count == chunks[0].word_count

    def test_empty_page_yields_no_chunks(self) -> None:
        assert Chunker().chunk_page(make_page(""), page_index=0) == []

    def test_chunk_ids_are_stable_and_scoped_to_page(self) -> None:
        text = "## A\n" + "alpha words here. " * 40 + "\n\n## B\n" + "beta words there. " * 40
        page_a = make_page(text, url="https://example.com/a")
        page_b = make_page(text, url="https://example.com/b")
        chunks_a = Chunker().chunk_page(page_a, page_index=0)
        chunks_b = Chunker().chunk_page(page_b, page_index=3)
        assert [c.id for c in chunks_a] == [f"p00-c{i}" for i in range(len(chunks_a))]
        assert all(c.id.startswith("p03-") for c in chunks_b)

    def test_chunks_carry_page_metadata(self) -> None:
        page = make_page("## Plans\nAll plans include support.")
        chunk = Chunker().chunk_page(page, page_index=1)[0]
        assert chunk.page_url == "https://example.com/pricing"
        assert chunk.page_title == "Pricing — Example"
        assert chunk.index == 0


class TestStructureAwareness:
    def test_heading_path_tracks_section_hierarchy(self) -> None:
        text = (
            "## Pricing\nTeams costs 49 per user.\n\n"
            "### Enterprise\nContact sales for a quote.\n\n"
            "## Support\nEmail support is included."
        )
        chunks = Chunker(target_words=5, overlap_words=1).chunk_page(make_page(text), 0)
        # each tiny section becomes its own chunk, in order
        assert [c.heading_path for c in chunks] == [
            "Pricing",
            "Pricing > Enterprise",
            "Support",
        ]

    def test_small_sections_merge_with_first_heading(self) -> None:
        text = "## Alpha\nfirst small section body text.\n\n## Beta\nsecond small section body text."
        chunks = Chunker(target_words=100, overlap_words=5).chunk_page(make_page(text), 0)
        assert len(chunks) == 1  # merged: both fit in one chunk
        assert chunks[0].heading_path == "Alpha"
        assert "second small section" in chunks[0].text


class TestBudgets:
    def test_no_chunk_exceeds_target_plus_overlap(self) -> None:
        text = "## Long\n" + "This sentence has exactly ten words in it always. " * 60
        chunks = Chunker(target_words=50, overlap_words=10).chunk_page(make_page(text), 0)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.word_count <= 50 + 10 + 5  # target + overlap + line join slack

    def test_split_section_carries_overlap_tail(self) -> None:
        text = "## Section\n" + ("sentence number one. " * 100)
        chunks = Chunker(target_words=30, overlap_words=5).chunk_page(make_page(text), 0)
        assert len(chunks) >= 2
        first_tail = chunks[0].text.split()[-5:]
        assert chunks[1].text.split()[:5] == first_tail

    def test_invalid_overlap_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            Chunker(target_words=100, overlap_words=100)

    def test_default_settings_chunk_typical_page(self) -> None:
        body = "## Features\n" + "\n\n".join(
            f"Paragraph {i} about the product and its many useful capabilities." for i in range(30)
        )
        chunks = Chunker().chunk_page(make_page(body), 0)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.word_count <= 280 + 45 + 5
