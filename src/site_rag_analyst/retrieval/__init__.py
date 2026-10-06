"""Retrieval stage: query embedding + top-k search with page diversity."""

from site_rag_analyst.retrieval.retriever import Retriever, diversify_by_page

__all__ = ["Retriever", "diversify_by_page"]
