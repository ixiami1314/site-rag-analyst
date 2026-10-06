"""Pipeline orchestration: crawl -> extract -> chunk -> embed -> store -> retrieve -> analyze.

The pipeline wires the stages together but owns no logic itself; every stage
is injectable (see the constructor), which is what keeps the layers
independently testable and lets demo mode swap fetcher/analyst without any
branching in stage code.

Each run gets its own SQLite store file and output directory, so runs are
isolated, replayable and inspectable end-to-end.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from site_rag_analyst.analysis.analyst import ANALYSIS_SECTIONS, Analyst, build_analyst
from site_rag_analyst.chunking.chunker import Chunker
from site_rag_analyst.config import Settings
from site_rag_analyst.crawler.crawler import Crawler
from site_rag_analyst.crawler.fetcher import DEMO_ORIGIN, DemoSiteFetcher, Fetcher, HttpFetcher
from site_rag_analyst.embedding.base import build_provider
from site_rag_analyst.extraction.extractor import ContentExtractor
from site_rag_analyst.models import (
    ExtractedPage,
    PageSummary,
    PipelineResult,
    RetrievalPreview,
    StageStat,
)
from site_rag_analyst.reporting.report import save_outputs
from site_rag_analyst.retrieval.retriever import Retriever
from site_rag_analyst.store.vecstore import VectorStore, open_store

logger = logging.getLogger(__name__)

# Pages below this many extracted words are usually shells (login walls,
# empty categories) and only add noise to the corpus.
MIN_PAGE_WORDS = 25


class Pipeline:
    """One configured run of the full chain."""

    def __init__(
        self,
        settings: Settings,
        fetcher: Fetcher | None = None,
        analyst: Analyst | None = None,
        store_path: Path | None = None,
        persist: bool = True,
    ) -> None:
        self._settings = settings
        self._fetcher = fetcher
        self._analyst = analyst
        self._store_path = store_path
        self._persist = persist

    # ------------------------------------------------------------------ #

    def run(self, url: str, run_id: str | None = None) -> PipelineResult:
        run_id = run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
        demo_site = url.rstrip("/").startswith(DEMO_ORIGIN)
        result = PipelineResult(
            run_id=run_id,
            url=url,
            demo=self._settings.demo_mode,
            mode="demo" if self._settings.demo_mode else "live",
        )
        stats: list[StageStat] = []
        try:
            result = self._run_inner(result, stats, demo_site)
        except Exception as exc:  # noqa: BLE001 - report errors as run results
            logger.exception("pipeline run %s failed", run_id)
            result.error = f"{type(exc).__name__}: {exc}"
        result.finished_at = datetime.now(result.started_at.tzinfo)
        if self._persist:
            try:
                save_outputs(result, self._settings.output_dir / run_id)
            except OSError as exc:
                logger.warning("could not persist outputs for %s: %s", run_id, exc)
        return result

    # ------------------------------------------------------------------ #

    def _run_inner(
        self, result: PipelineResult, stats: list[StageStat], demo_site: bool
    ) -> PipelineResult:
        settings = self._settings

        # 1. crawl -------------------------------------------------------- #
        started = time.perf_counter()
        fetcher = self._fetcher or (
            DemoSiteFetcher() if demo_site else HttpFetcher(settings.user_agent)
        )
        crawler = Crawler(fetcher=fetcher, settings=settings)
        fetched, crawl_stats = crawler.crawl(result.url)
        stats.append(
            StageStat(
                stage="crawl",
                detail=f"{crawl_stats.pages_fetched} fetched, "
                f"{crawl_stats.pages_failed} failed, "
                f"{crawl_stats.urls_skipped_robots} skipped by robots",
                items=crawl_stats.pages_fetched,
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        )
        result.crawl = crawl_stats
        ok_pages = [p for p in fetched if p.ok]
        if not ok_pages:
            result.error = "crawl produced no usable pages"
            return result

        # 2. extract ------------------------------------------------------ #
        started = time.perf_counter()
        extractor = ContentExtractor()
        extracted: list[ExtractedPage] = []
        for page in ok_pages:
            doc = extractor.extract(page, links_discovered=len(fetched))
            if doc.word_count >= MIN_PAGE_WORDS:
                extracted.append(doc)
            else:
                logger.debug("skipping thin page %s (%s words)", doc.url, doc.word_count)
        if not extracted:
            result.error = "extraction produced no pages above the minimum size"
            return result
        kept_avg = sum(d.kept_ratio for d in extracted) / len(extracted)
        stats.append(
            StageStat(
                stage="extract",
                detail=f"{len(extracted)} pages, avg kept_ratio {kept_avg:.2f}",
                items=len(extracted),
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        )

        # 3. chunk -------------------------------------------------------- #
        started = time.perf_counter()
        chunker = Chunker(
            target_words=settings.chunk_target_words,
            overlap_words=settings.chunk_overlap_words,
        )
        chunks = []
        for page_index, doc in enumerate(extracted):
            chunks.extend(chunker.chunk_page(doc, page_index=page_index))
        avg_words = sum(c.word_count for c in chunks) / max(len(chunks), 1)
        stats.append(
            StageStat(
                stage="chunk",
                detail=f"{len(chunks)} chunks, avg {avg_words:.0f} words "
                f"(target {settings.chunk_target_words})",
                items=len(chunks),
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        )
        result.pages = extracted
        result.chunks = chunks

        # 4. embed -------------------------------------------------------- #
        started = time.perf_counter()
        provider = build_provider(settings)
        provider.fit(chunks)
        vectors = provider.embed([c.text for c in chunks])
        stats.append(
            StageStat(
                stage="embed",
                detail=f"provider {provider.name}, dim {provider.dim}",
                items=len(vectors),
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        )

        # 5. store -------------------------------------------------------- #
        started = time.perf_counter()
        store_path = self._store_path or (
            settings.data_dir / f"{result.run_id}.sqlite3"
        )
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store: VectorStore = open_store(
            store_path, dim=provider.dim if provider.name == "openai" else None
        )
        store.add(chunks, vectors)
        stats.append(
            StageStat(
                stage="store",
                detail=f"{store.backend_name()} at {store_path.name}",
                items=len(chunks),
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        )

        # 6. retrieve ----------------------------------------------------- #
        started = time.perf_counter()
        retriever = Retriever(store, provider)
        previews: list[RetrievalPreview] = []
        for sid, _title, query in ANALYSIS_SECTIONS:
            previews.append(
                RetrievalPreview(
                    section_id=sid,
                    query=query,
                    results=retriever.search(query, k=settings.retrieval_top_k),
                )
            )
        hits = sum(len(p.results) for p in previews)
        stats.append(
            StageStat(
                stage="retrieve",
                detail=f"{len(previews)} queries x top-{settings.retrieval_top_k}, {hits} hits",
                items=hits,
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        )
        result.retrieval = previews

        # 7. analyze ------------------------------------------------------ #
        started = time.perf_counter()
        analyst = self._analyst or build_analyst(settings)
        report = analyst.analyze(result.url, previews, extracted)
        report.pages = self._page_summaries(extracted, chunks)
        stats.append(
            StageStat(
                stage="analyze",
                detail=report.model or "unknown",
                items=sum(len(s.points) for s in report.sections),
                duration_seconds=round(time.perf_counter() - started, 3),
            )
        )
        report.stage_stats = stats
        result.report = report
        store.close()
        return result

    # ------------------------------------------------------------------ #

    @staticmethod
    def _page_summaries(
        extracted: list[ExtractedPage], chunks: list
    ) -> list[PageSummary]:
        per_page: Counter[str] = Counter(c.page_url for c in chunks)
        return [
            PageSummary(
                url=d.url,
                title=d.title or d.url.rsplit("/", 1)[-1] or d.url,
                word_count=d.word_count,
                chunk_count=per_page[d.url],
                summary_line=d.meta_description
                or (d.text[:140].rsplit(" ", 1)[0] + "…" if d.text else ""),
            )
            for d in extracted
        ]
