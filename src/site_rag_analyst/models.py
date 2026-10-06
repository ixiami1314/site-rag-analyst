"""Typed artifacts produced by each pipeline stage.

The models here are the contract between stages: the crawler emits
:class:`FetchedPage`, the extractor :class:`ExtractedPage`, the chunker
:class:`Chunk`, and so on. Because every artifact is a pydantic model, the
whole pipeline (including intermediates) can be serialized, inspected and
evaluated — which is exactly what the web UI and the eval harness do.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Timezone-aware UTC now (pydantic default_factory helper)."""
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Crawl stage
# --------------------------------------------------------------------------- #


class FetchedPage(BaseModel):
    """Raw result of fetching one URL."""

    url: str
    status: int = 0
    content_type: str = ""
    html: str = ""
    fetched_at: datetime = Field(default_factory=utcnow)
    error: str | None = None
    depth: int = 0
    source: str = "http"  # http | demo (bundled demo site)

    @property
    def ok(self) -> bool:
        return self.error is None and 200 <= self.status < 300 and bool(self.html)


class CrawlStats(BaseModel):
    """Summary of one crawl run."""

    start_url: str
    pages_fetched: int = 0
    pages_failed: int = 0
    urls_skipped_robots: int = 0
    urls_skipped_budget: int = 0
    elapsed_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Extraction stage
# --------------------------------------------------------------------------- #


class ExtractedPage(BaseModel):
    """Cleaned main content extracted from a fetched page."""

    url: str
    title: str = ""
    meta_description: str = ""
    headings: list[str] = []
    text: str = ""
    word_count: int = 0
    links_discovered: int = 0
    kept_ratio: float = 0.0  # extracted words / raw html text words
    fetch_status: int = 0


# --------------------------------------------------------------------------- #
# Chunking stage
# --------------------------------------------------------------------------- #


class Chunk(BaseModel):
    """A structure-aware text chunk — the retrieval unit of the pipeline."""

    id: str  # e.g. "p03-c1" (page 3, chunk 1)
    page_url: str
    page_title: str
    heading_path: str  # e.g. "Pricing > Teams"
    index: int  # per-page chunk index
    text: str
    word_count: int


# --------------------------------------------------------------------------- #
# Retrieval stage
# --------------------------------------------------------------------------- #


class RetrievedChunk(BaseModel):
    """A chunk returned for a query, with its similarity score."""

    chunk: Chunk
    score: float  # cosine similarity, 0..1 (higher = more similar)
    query: str


class RetrievalPreview(BaseModel):
    """Top results for one analysis query (used by the UI and the report)."""

    query: str
    results: list[RetrievedChunk] = []


# --------------------------------------------------------------------------- #
# Analysis / report
# --------------------------------------------------------------------------- #


class CitedPoint(BaseModel):
    """One claim in the report plus the chunk ids that support it."""

    text: str
    citations: list[str] = []  # Chunk.id values


class ReportSection(BaseModel):
    """A themed section of the analysis (offerings, pricing, ...)."""

    id: str
    title: str
    query: str  # the retrieval query this section was grounded on
    points: list[CitedPoint] = []


class PageSummary(BaseModel):
    """Per-page digest shown in the report and UI."""

    url: str
    title: str
    word_count: int
    chunk_count: int
    summary_line: str = ""


class StageStat(BaseModel):
    """Timing/count record for one pipeline stage."""

    stage: str
    detail: str = ""
    items: int = 0
    duration_seconds: float = 0.0


class SiteReport(BaseModel):
    """The final structured analysis of a site."""

    site_url: str
    generated_at: datetime = Field(default_factory=utcnow)
    mode: str = "demo"  # demo | live
    model: str = ""  # LLM used ("" in demo mode)
    executive_summary: str = ""
    sections: list[ReportSection] = []
    pages: list[PageSummary] = []
    stage_stats: list[StageStat] = []


class PipelineResult(BaseModel):
    """Everything one pipeline run produced, including intermediates.

    This is what the web UI renders stage-by-stage and what gets persisted to
    ``output/<run_id>/artifacts.json``.
    """

    run_id: str
    url: str
    demo: bool
    mode: str  # demo | live
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    crawl: CrawlStats | None = None
    pages: list[ExtractedPage] = []
    chunks: list[Chunk] = []
    retrieval: list[RetrievalPreview] = []
    report: SiteReport | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.report is not None
