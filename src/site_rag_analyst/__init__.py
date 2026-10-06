"""site-rag-analyst: crawl a website, ground an LLM in it, produce a cited report.

A compact, self-contained RAG pipeline:

    crawl -> extract -> chunk -> embed -> retrieve -> analyze -> report

Every stage is independently testable and emits typed artifacts, so the whole
chain can be inspected (and evaluated) — see the README for the architecture.
"""

__version__ = "0.1.0"
