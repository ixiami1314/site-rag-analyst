"""Tests for embedding providers (TF-IDF offline; API provider mocked)."""

from __future__ import annotations

import numpy as np
import pytest

from site_rag_analyst.embedding.tfidf import TfidfEmbeddings, tokenize
from site_rag_analyst.models import Chunk


def chunk_with(text: str, cid: str = "p00-c0") -> Chunk:
    return Chunk(
        id=cid,
        page_url="https://example.com/x",
        page_title="X",
        heading_path="X",
        index=0,
        text=text,
        word_count=len(text.split()),
    )


CORPUS = [
    chunk_with(
        "The Meridian Flow Builder lets logistics teams design routing "
        "workflows with a visual drag-and-drop canvas and version control.",
        "p00-c0",
    ),
    chunk_with(
        "Flow Connect integrates with SAP, Oracle NetSuite and legacy EDI "
        "systems using prebuilt connectors and a flexible mapping language.",
        "p01-c0",
    ),
    chunk_with(
        "Pricing: the Starter plan costs 19 dollars per user per month, the "
        "Team plan 49 dollars, and Enterprise is custom-quoted annually.",
        "p02-c0",
    ),
]


class TestTokenize:
    def test_lowercases_and_splits_alphanumerics(self) -> None:
        assert tokenize("State-of-the-art dashboards!") == [
            "state", "art", "dashboards",  # 'of' is a stopword
        ]

    def test_stopwords_removed(self) -> None:
        assert "the" not in tokenize("The quick brown fox jumps over the lazy dog")
        assert "quick" in tokenize("The quick brown fox")

    def test_single_characters_dropped(self) -> None:
        assert tokenize("a big idea") == ["big", "idea"]


class TestTfidfEmbeddings:
    def test_fit_then_dim_matches_vocabulary(self) -> None:
        provider = TfidfEmbeddings()
        provider.fit(CORPUS)
        vectors = provider.embed([c.text for c in CORPUS])
        assert len(vectors) == 3
        assert all(v.shape == (provider.dim,) for v in vectors)

    def test_vectors_l2_normalized(self) -> None:
        provider = TfidfEmbeddings()
        provider.fit(CORPUS)
        for vector in provider.embed([c.text for c in CORPUS]):
            assert np.isclose(np.linalg.norm(vector), 1.0)

    def test_similar_query_scores_higher_than_unrelated(self) -> None:
        provider = TfidfEmbeddings()
        provider.fit(CORPUS)
        pricing_vec = provider.embed_query("how much does the team plan cost")
        unrelated_vec = provider.embed_query("karaoke night schedule")

        def score(vec: np.ndarray) -> float:
            return max(float(vec @ v) for v in provider.embed([c.text for c in CORPUS]))

        assert score(pricing_vec) > score(unrelated_vec)

    def test_best_match_is_the_right_chunk(self) -> None:
        provider = TfidfEmbeddings()
        provider.fit(CORPUS)
        vectors = provider.embed([c.text for c in CORPUS])
        query = provider.embed_query("pricing of the starter and team plans")
        best = int(np.argmax([query @ v for v in vectors]))
        assert CORPUS[best].id == "p02-c0"

    def test_embed_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            TfidfEmbeddings().embed(["text"])

    def test_deterministic(self) -> None:
        a, b = TfidfEmbeddings(), TfidfEmbeddings()
        a.fit(CORPUS)
        b.fit(CORPUS)
        va = a.embed_query("workflow connectors pricing")
        vb = b.embed_query("workflow connectors pricing")
        assert np.allclose(va, vb)


class TestBuildProvider:
    def test_offline_settings_get_tfidf(self, default_settings) -> None:
        from site_rag_analyst.embedding.base import build_provider

        assert build_provider(default_settings).name == "tfidf"

    def test_live_settings_get_openai(self, live_settings) -> None:
        from site_rag_analyst.embedding.base import build_provider

        assert build_provider(live_settings).name == "openai"

    def test_openai_requires_credentials(self) -> None:
        from site_rag_analyst.embedding.openai_compat import OpenAICompatEmbeddings

        with pytest.raises(ValueError, match="base_url and api_key"):
            OpenAICompatEmbeddings(base_url="", api_key="", model="m")
