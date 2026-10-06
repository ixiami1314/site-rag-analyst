"""Tests for the retrieval stage (TF-IDF + numpy store, no network)."""

from __future__ import annotations

from site_rag_analyst.chunking.chunker import Chunker
from site_rag_analyst.embedding.tfidf import TfidfEmbeddings
from site_rag_analyst.models import ExtractedPage
from site_rag_analyst.retrieval.retriever import Retriever, diversify_by_page
from site_rag_analyst.store.vecstore import NumpyVectorStore


def build_corpus(tmp_path):
    pages = [
        ExtractedPage(
            url=f"https://example.com/{slug}",
            title=title,
            text=text,
            word_count=len(text.split()),
        )
        for slug, title, text in [
            (
                "pricing",
                "Pricing",
                "## Pricing\nStarter costs 19 dollars per user per month. "
                "Team costs 49 dollars per user per month. Enterprise is custom "
                "priced with annual billing and volume discounts available.",
            ),
            (
                "products",
                "Products",
                "## Products\nThe Flow Builder is a visual workflow designer. "
                "Flow Connect links SAP and Oracle systems. Flow Insights gives "
                "analytics dashboards for logistics teams.",
            ),
            (
                "contact",
                "Contact",
                "## Contact\nEmail support@meridianflow.example or call the "
                "office in Rotterdam. Support hours are 9 to 5 Central European.",
            ),
        ]
    ]
    chunker = Chunker()
    chunks = []
    for index, page in enumerate(pages):
        chunks.extend(chunker.chunk_page(page, page_index=index))
    embeddings = TfidfEmbeddings()
    texts = [c.text for c in chunks]
    embeddings.fit(texts)
    store = NumpyVectorStore(tmp_path / "rag.sqlite3")
    store.add(chunks, embeddings.embed(texts))
    return store, embeddings


class TestRetriever:
    def test_query_returns_ranked_relevant_chunks(self, tmp_path) -> None:
        store, embeddings = build_corpus(tmp_path)
        retriever = Retriever(store, embeddings)
        results = retriever.search("how much does the team plan cost", k=3)
        assert 1 <= len(results) <= 3
        assert results[0].chunk.page_url.endswith("/pricing")
        assert all(r.query == "how much does the team plan cost" for r in results)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        store.close()

    def test_pricing_query_beats_unrelated_page(self, tmp_path) -> None:
        store, embeddings = build_corpus(tmp_path)
        retriever = Retriever(store, embeddings)
        results = retriever.search("pricing plans and monthly costs", k=1)
        assert results[0].chunk.heading_path == "Pricing"
        store.close()

    def test_max_per_page_caps_single_page_domination(self, tmp_path) -> None:
        store, embeddings = build_corpus(tmp_path)
        retriever = Retriever(store, embeddings)
        results = retriever.search("workflow logistics team support pricing", k=6, max_per_page=1)
        pages = [r.chunk.page_url for r in results]
        assert len(pages) == len(set(pages)), "no page may appear twice"
        store.close()


class TestDiversify:
    def test_caps_per_page_preserving_order(self) -> None:
        from site_rag_analyst.models import Chunk, RetrievedChunk

        def item(page: str, cid: str) -> RetrievedChunk:
            chunk = Chunk(
                id=cid, page_url=page, page_title=page, heading_path="", index=0,
                text="t", word_count=1,
            )
            return RetrievedChunk(chunk=chunk, score=0.9, query="q")

        results = [
            item("/a", "a1"), item("/a", "a2"), item("/a", "a3"),
            item("/b", "b1"), item("/c", "c1"),
        ]
        picked = diversify_by_page(results, k=4, max_per_page=2)
        assert [p.chunk.id for p in picked] == ["a1", "a2", "b1", "c1"]

    def test_returns_fewer_when_k_unreachable(self) -> None:
        from site_rag_analyst.models import Chunk, RetrievedChunk

        results = [
            RetrievedChunk(
                chunk=Chunk(
                    id="x", page_url="/x", page_title="x", heading_path="",
                    index=0, text="t", word_count=1,
                ),
                score=0.5,
                query="q",
            )
        ]
        assert len(diversify_by_page(results, k=5, max_per_page=3)) == 1
