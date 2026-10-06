"""Tests for the pipeline artifact models."""

from __future__ import annotations

from site_rag_analyst.models import (
    Chunk,
    CitedPoint,
    ExtractedPage,
    FetchedPage,
    PipelineResult,
    ReportSection,
    RetrievedChunk,
    SiteReport,
)


def _chunk(cid: str = "p00-c0") -> Chunk:
    return Chunk(
        id=cid,
        page_url="https://example.com/pricing",
        page_title="Pricing",
        heading_path="Pricing",
        index=0,
        text="Plans start at 19 dollars per user per month.",
        word_count=9,
    )


class TestFetchedPage:
    def test_ok_requires_2xx_html_without_error(self) -> None:
        good = FetchedPage(url="u", status=200, html="<html></html>")
        assert good.ok is True

        bad_status = good.model_copy(update={"status": 404})
        assert bad_status.ok is False

        with_error = good.model_copy(update={"error": "timeout"})
        assert with_error.ok is False

        empty = good.model_copy(update={"html": ""})
        assert empty.ok is False


class TestChunkIds:
    def test_ids_are_human_scannable(self) -> None:
        c = _chunk("p03-c1")
        assert c.id == "p03-c1"
        assert c.word_count == len(c.text.split())


class TestRetrieval:
    def test_retrieved_chunk_keeps_query_context(self) -> None:
        rc = RetrievedChunk(chunk=_chunk(), score=0.83, query="how much does it cost")
        assert rc.chunk.id == "p00-c0"
        assert 0.0 <= rc.score <= 1.0


class TestPipelineResult:
    def test_not_ok_before_report_set(self) -> None:
        r = PipelineResult(run_id="r1", url="https://example.com", demo=True, mode="demo")
        assert r.ok is False
        assert r.finished_at is None

    def test_not_ok_when_error_recorded(self) -> None:
        r = PipelineResult(
            run_id="r2",
            url="https://example.com",
            demo=True,
            mode="demo",
            report=SiteReport(site_url="https://example.com"),
            error="boom",
        )
        assert r.ok is False

    def test_full_roundtrip_serialization(self) -> None:
        report = SiteReport(
            site_url="https://example.com",
            mode="demo",
            executive_summary="A demo site.",
            sections=[
                ReportSection(
                    id="pricing",
                    title="Pricing",
                    query="pricing plans",
                    points=[CitedPoint(text="Starts at $19", citations=["p00-c0"])],
                )
            ],
            pages=[],
        )
        r = PipelineResult(
            run_id="r3",
            url="https://example.com",
            demo=True,
            mode="demo",
            finished_at=None,
            pages=[ExtractedPage(url="https://example.com", title="Home", word_count=10)],
            chunks=[_chunk()],
            retrieval=[],
            report=report,
        )
        assert r.ok is True

        data = r.model_dump(mode="json")
        assert data["chunks"][0]["id"] == "p00-c0"
        restored = PipelineResult.model_validate(data)
        assert restored.report is not None
        assert restored.report.sections[0].points[0].citations == ["p00-c0"]
