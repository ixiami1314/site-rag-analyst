"""End-to-end pipeline tests over the bundled demo site — fully offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from site_rag_analyst.analysis.analyst import ANALYSIS_SECTIONS
from site_rag_analyst.config import Settings
from site_rag_analyst.crawler.fetcher import DEMO_ORIGIN
from site_rag_analyst.pipeline import Pipeline
from site_rag_analyst.reporting.report import render_markdown, save_outputs


@pytest.fixture
def demo_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="",
        llm_api_key="",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        crawl_delay_seconds=0.0,
    )


class TestPipelineDemoRun:
    def test_full_demo_run_succeeds(self, demo_settings: Settings) -> None:
        result = Pipeline(demo_settings).run(DEMO_ORIGIN + "/", run_id="test-run")
        assert result.error is None
        assert result.ok is True
        assert result.demo is True
        assert result.mode == "demo"
        assert result.finished_at is not None

    def test_demo_site_yields_expected_corpus(self, demo_settings: Settings) -> None:
        result = Pipeline(demo_settings).run(DEMO_ORIGIN + "/")
        assert len(result.pages) == 12
        assert len(result.chunks) >= 18  # 300-460 word pages split ~2 chunks each
        assert result.crawl is not None and result.crawl.pages_failed == 0

    def test_report_has_all_analysis_sections(self, demo_settings: Settings) -> None:
        result = Pipeline(demo_settings).run(DEMO_ORIGIN + "/")
        assert result.report is not None
        got = [s.id for s in result.report.sections]
        assert got == [sid for sid, _t, _q in ANALYSIS_SECTIONS]

    def test_all_citations_resolve_to_real_chunks(self, demo_settings: Settings) -> None:
        result = Pipeline(demo_settings).run(DEMO_ORIGIN + "/")
        assert result.report is not None
        chunk_ids = {c.id for c in result.chunks}
        cited = [
            citation
            for section in result.report.sections
            for point in section.points
            for citation in point.citations
        ]
        assert cited, "demo report should cite evidence"
        assert set(cited) <= chunk_ids, "citations must reference real chunks"

    def test_retrieval_previews_present_for_each_query(self, demo_settings: Settings) -> None:
        result = Pipeline(demo_settings).run(DEMO_ORIGIN + "/")
        queries = [p.query for p in result.retrieval]
        assert queries == [q for _sid, _t, q in ANALYSIS_SECTIONS]
        assert all(p.results for p in result.retrieval)

    def test_stage_stats_recorded_with_timings(self, demo_settings: Settings) -> None:
        result = Pipeline(demo_settings).run(DEMO_ORIGIN + "/")
        assert result.report is not None
        stages = [s.stage for s in result.report.stage_stats]
        assert stages == [
            "crawl", "extract", "chunk", "embed", "store", "retrieve", "analyze",
        ]
        for stat in result.report.stage_stats:
            assert stat.duration_seconds >= 0.0
            assert stat.detail

    def test_sqlite_store_created_per_run(self, demo_settings: Settings) -> None:
        Pipeline(demo_settings).run(DEMO_ORIGIN + "/", run_id="store-run")
        assert (demo_settings.data_dir / "store-run.sqlite3").exists()

    def test_outputs_persisted(self, demo_settings: Settings) -> None:
        result = Pipeline(demo_settings).run(DEMO_ORIGIN + "/", run_id="persist-run")
        out = demo_settings.output_dir / "persist-run"
        assert (out / "report.md").exists()
        assert (out / "report.json").exists()
        assert (out / "artifacts.json").exists()
        markdown = (out / "report.md").read_text(encoding="utf-8")
        assert "Meridian Flow" in markdown
        assert "Executive summary" in markdown
        assert result.report is not None and result.report.pages


class TestPipelineFailurePaths:
    def test_crawl_failure_reported_not_raised(self, demo_settings: Settings, tmp_path) -> None:
        from site_rag_analyst.crawler.fetcher import DemoSiteFetcher

        empty_root = tmp_path / "empty-site"
        empty_root.mkdir()
        result = Pipeline(
            demo_settings, fetcher=DemoSiteFetcher(root=empty_root)
        ).run(DEMO_ORIGIN + "/")
        assert result.ok is False
        assert result.error and "no usable pages" in result.error


class TestMarkdownRendering:
    def test_markdown_contains_citations_and_index(self, demo_settings: Settings) -> None:
        result = Pipeline(demo_settings).run(DEMO_ORIGIN + "/")
        markdown = render_markdown(result)
        assert "## Site Overview" in markdown or "##" in markdown
        assert "Citation index" in markdown
        assert "[p" in markdown  # citation markers like [p00-c1]
        assert "demo mode" in markdown

    def test_markdown_for_failed_run(self, demo_settings: Settings) -> None:
        from site_rag_analyst.models import PipelineResult

        result = PipelineResult(
            run_id="failed-run",
            url="https://unreachable.example.com/",
            demo=True,
            mode="demo",
            error="ConnectionError: DNS failure",
        )
        markdown = render_markdown(result)
        assert "failed" in markdown
        assert "ConnectionError" in markdown

    def test_save_outputs_writes_three_files(self, demo_settings: Settings, tmp_path) -> None:
        result = Pipeline(demo_settings, persist=False).run(DEMO_ORIGIN + "/")
        out = save_outputs(result, tmp_path / "custom-out")
        names = sorted(p.name for p in out.iterdir())
        assert names == ["artifacts.json", "report.json", "report.md"]
