"""Embedding stage: pluggable text -> vector providers."""

from site_rag_analyst.embedding.base import EmbeddingProvider, build_provider
from site_rag_analyst.embedding.openai_compat import OpenAICompatEmbeddings
from site_rag_analyst.embedding.tfidf import TfidfEmbeddings

__all__ = [
    "EmbeddingProvider",
    "OpenAICompatEmbeddings",
    "TfidfEmbeddings",
    "build_provider",
]
