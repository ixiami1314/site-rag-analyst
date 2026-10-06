"""Embedding provider abstraction.

Two implementations ship:

- :class:`~embedding.tfidf.TfidfEmbeddings` — deterministic, fully offline
  TF-IDF vectors. This is what demo mode uses: the whole pipeline runs (and
  its retrieval quality is measured in ``evals/``) with zero network access.
- :class:`~embedding.openai_compat.OpenAICompatEmbeddings` — any
  OpenAI-compatible ``/embeddings`` endpoint (OpenAI, LiteLLM, Ollama, ...).

Note the split from the *analysis* LLM: Anthropic's Claude API does not offer
embeddings, so the embedding endpoint is configured independently (see
``EMBEDDING_*`` settings) even when analysis defaults to a Claude model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from site_rag_analyst.config import Settings
from site_rag_analyst.models import Chunk


class EmbeddingProvider(Protocol):
    """Anything that can embed corpus chunks and queries into a shared space."""

    name: str

    def fit(self, chunks: list[Chunk]) -> None:
        """Corpus-level initialization (no-op for stateless providers)."""
        ...

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of corpus texts."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a retrieval query."""
        ...

    @property
    def dim(self) -> int:
        """Vector dimensionality (known after ``fit``/first embed)."""
        ...


def build_provider(settings: Settings) -> EmbeddingProvider:
    """Factory: pick the provider per settings (offline-safe)."""
    provider = settings.effective_embedding_provider
    if provider == "tfidf":
        from site_rag_analyst.embedding.tfidf import TfidfEmbeddings

        return TfidfEmbeddings()
    if provider == "openai":
        from site_rag_analyst.embedding.openai_compat import OpenAICompatEmbeddings

        return OpenAICompatEmbeddings(
            base_url=(settings.embedding_base_url or settings.llm_base_url).rstrip("/"),
            api_key=settings.embedding_api_key or settings.llm_api_key,
            model=settings.embedding_model,
        )
    raise ValueError(f"unknown embedding provider: {provider!r}")
