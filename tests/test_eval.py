"""Tests for the eval harness — including a retrieval quality gate.

The gate pins the bundled eval set's hit@5/MRR above a floor. If someone
changes chunking, extraction or embedding in a way that quietly degrades
retrieval, this test fails the build instead of letting it ship. The floor
sits below the current numbers (hit@5 = 0.97, MRR = 0.83, TF-IDF offline) to
absorb legitimate variance without losing the safety net.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from site_rag_analyst.config import Settings
from site_rag_analyst.eval.runner import (
    build_demo_corpus,
    evaluate_cases,
    load_eval_set,
)
from site_rag_analyst.retrieval.retriever import Retriever


@pytest.fixture
def eval_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="",
        llm_api_key="",
        data_dir=tmp_path / "data",
        crawl_delay_seconds=0.0,
    )


class TestEvalSet:
    def test_bundled_set_loads_with_expected_size(self) -> None:
        cases = load_eval_set()
        assert len(cases) >= 25
        assert all(case.query and case.expected_url for case in cases)

    def test_expected_urls_cover_the_demo_site(self) -> None:
        cases = load_eval_set()
        pages = {case.expected_url for case in cases}
        assert "/pricing" in pages
        assert "/products/flow-guard" in pages
        assert "/docs/api" in pages
        assert "/about" in pages
        assert len(pages) >= 9


class TestMetrics:
    def test_perfect_and_miss_cases(self, eval_settings, tmp_path) -> None:
        store, provider = build_demo_corpus(eval_settings, tmp_path / "e.sqlite3")
        retriever = Retriever(store, provider)
        try:
            from site_rag_analyst.eval.runner import EvalCase

            cases = [
                EvalCase(query="how much does the team plan cost", expected_url="/pricing"),
                EvalCase(query="quantum karaoke machine", expected_url="/pricing"),
            ]
            results, summary = evaluate_cases(retriever, cases)
            assert results[0].hit_at_3 and results[0].first_rank is not None
            assert not results[1].hit_at_5 and results[1].first_rank is None
            assert summary.cases == 2
            assert summary.hit_at_5 == 0.5
            assert 0 < summary.mrr <= 0.5
        finally:
            store.close()


class TestQualityGate:
    def test_offline_retrieval_meets_floor(self, eval_settings, tmp_path) -> None:
        from site_rag_analyst.eval.runner import run_evaluation

        _results, summary = run_evaluation(eval_settings, tmp_path / "gate.sqlite3")
        assert summary.cases >= 25
        assert summary.hit_at_3 >= 0.85, f"hit@3 regressed: {summary.hit_at_3:.2f}"
        assert summary.hit_at_5 >= 0.90, f"hit@5 regressed: {summary.hit_at_5:.2f}"
        assert summary.mrr >= 0.75, f"MRR regressed: {summary.mrr:.2f}"
