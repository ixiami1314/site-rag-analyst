"""Deterministic, retrieval-grounded mock analyst.

Demo mode must produce something honest: not lorem ipsum, but a real
*extractive* analysis — key sentences are picked from retrieved chunks by a
tiny relevance heuristic (numbers, proper nouns, position) and cited to their
chunk ids, exactly like the LLM path. The demo therefore demonstrates the
whole pipeline; only the final paraphrase step is simulated.

This is also the fallback when an LLM reply fails to parse, so a malformed
API response degrades the report instead of the run.
"""

from __future__ import annotations

import re
from collections import Counter

from site_rag_analyst.embedding.tfidf import tokenize
from site_rag_analyst.models import (
    CitedPoint,
    ExtractedPage,
    ReportSection,
    RetrievalPreview,
    SiteReport,
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"\d")
_PROPER_NOUN_RE = re.compile(r"(?<![.!?]\s)\b[A-Z][a-z]{2,}")


class MockAnalyst:
    """Extractive analyst: no network, fully deterministic."""

    def analyze(
        self,
        site_url: str,
        retrieval: list[RetrievalPreview],
        pages: list[ExtractedPage],
        model: str = "extractive-demo",
    ) -> SiteReport:
        sections: list[ReportSection] = []
        for preview in retrieval:
            points = self._points(preview)
            sections.append(
                ReportSection(
                    id=self._section_id(preview.query),
                    title=preview.query,
                    query=preview.query,
                    points=points,
                )
            )
        return SiteReport(
            site_url=site_url,
            mode="demo",
            model=model,
            executive_summary=self._summary(pages),
            sections=sections,
            pages=self._page_summaries(pages),
        )

    # ------------------------------------------------------------------ #

    def _points(self, preview: RetrievalPreview, per_chunk_limit: int = 1) -> list[CitedPoint]:
        points: list[CitedPoint] = []
        for item in preview.results[:4]:
            sentences = [
                s.strip() for s in _SENTENCE_RE.split(item.chunk.text) if len(s.strip()) > 30
            ]
            if not sentences:
                continue
            ranked = sorted(
                enumerate(sentences),
                key=lambda pair: (-self._sentence_score(pair[1]), pair[0]),
            )
            chosen = [text for _pos, text in ranked[:per_chunk_limit]]
            points.append(CitedPoint(text=" ".join(chosen), citations=[item.chunk.id]))
        return points

    @staticmethod
    def _sentence_score(sentence: str) -> int:
        score = 0
        if _NUMBER_RE.search(sentence):
            score += 3  # concrete facts (prices, versions, counts) first
        if _PROPER_NOUN_RE.search(sentence):
            score += 2  # named entities beat filler
        score += min(len(sentence.split()) // 25, 2)
        return score

    def _summary(self, pages: list[ExtractedPage]) -> str:
        total_words = sum(p.word_count for p in pages)
        terms = Counter()
        for page in pages:
            terms.update(tokenize(f"{page.title} {page.meta_description} {page.text[:800]}"))
        top_terms = [t for t, _count in terms.most_common(6)]
        titles = ", ".join(p.title or p.url.rsplit("/", 1)[-1] for p in pages[:5])
        return (
            f"Demo-mode extractive analysis of {len(pages)} pages (~{total_words} words). "
            f"Prominent topics: {', '.join(top_terms) if top_terms else 'n/a'}. "
            f"Sample pages: {titles}. Each point below is a key sentence extracted "
            "from retrieved context — no LLM was called."
        )

    @staticmethod
    def _page_summaries(pages: list[ExtractedPage]) -> list:
        from site_rag_analyst.models import PageSummary

        return [
            PageSummary(
                url=p.url,
                title=p.title,
                word_count=p.word_count,
                chunk_count=0,  # filled in by the pipeline (chunk counts per page)
                summary_line=p.meta_description or (p.text[:140] + "…" if p.text else ""),
            )
            for p in pages
        ]

    @staticmethod
    def _section_id(query: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
        return slug[:40] or "section"
