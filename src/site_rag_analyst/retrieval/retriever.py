"""Top-k retrieval over a vector store.

The retriever owns two concerns the raw store doesn't:

- embedding queries with the *same* provider that embedded the corpus, and
- page diversity: without it, one long page (a changelog, say) can crowd out
  every other page for a query like "contact" simply by having many similar
  chunks. ``diversify_by_page`` caps results per page while preserving
  relevance order.
"""

from __future__ import annotations

import numpy as np

from site_rag_analyst.embedding.base import EmbeddingProvider
from site_rag_analyst.models import RetrievedChunk
from site_rag_analyst.store.vecstore import VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embeddings: EmbeddingProvider) -> None:
        self._store = store
        self._embeddings = embeddings

    def search(
        self, query: str, k: int = 6, max_per_page: int | None = 3
    ) -> list[RetrievedChunk]:
        """Embed the query and return up to ``k`` chunks, best first."""
        query_vector = self._embeddings.embed_query(query)
        results = self._store.search(np.asarray(query_vector), k=k * 2 if max_per_page else k)
        retrieved = [
            RetrievedChunk(chunk=chunk, score=score, query=query)
            for chunk, score in results
        ]
        if max_per_page:
            retrieved = diversify_by_page(retrieved, k=k, max_per_page=max_per_page)
        return retrieved[:k]


def diversify_by_page(
    results: list[RetrievedChunk], k: int, max_per_page: int
) -> list[RetrievedChunk]:
    """Keep at most ``max_per_page`` chunks per page, relevance order intact."""
    per_page: dict[str, int] = {}
    selected: list[RetrievedChunk] = []
    for item in results:
        if len(selected) >= k:
            break
        used = per_page.get(item.chunk.page_url, 0)
        if used >= max_per_page:
            continue
        per_page[item.chunk.page_url] = used + 1
        selected.append(item)
    return selected
