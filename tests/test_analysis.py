"""Tests for the analysis stage: JSON extraction, mock analyst, LLM analyst.

The LLM analyst is tested against a scripted ChatClient — no network.
"""

from __future__ import annotations

import json

import pytest

from site_rag_analyst.analysis.analyst import ANALYSIS_SECTIONS, LLMAnalyst, build_analyst
from site_rag_analyst.analysis.llm import extract_json
from site_rag_analyst.analysis.mock import MockAnalyst
from site_rag_analyst.models import Chunk, ExtractedPage, RetrievalPreview, RetrievedChunk


def chunk(cid: str, text: str, page: str = "/pricing") -> Chunk:
    index = int(cid.split("-c")[1])
    return Chunk(
        id=cid,
        page_url=f"https://example.com{page}",
        page_title="Pricing",
        heading_path="Pricing",
        index=index,
        text=text,
        word_count=len(text.split()),
    )


CHUNKS = [
    chunk(
        "p00-c0",
        "Starter costs 19 dollars per user per month. Team costs 49 dollars "
        "per user per month with SSO included.",
    ),
    chunk(
        "p01-c0",
        "Flow Builder is a visual workflow designer for logistics teams.",
        page="/products",
    ),
]


def preview(query: str, chunks: list[Chunk]) -> RetrievalPreview:
    return RetrievalPreview(
        query=query,
        results=[RetrievedChunk(chunk=c, score=0.7, query=query) for c in chunks],
    )


PAGES = [
    ExtractedPage(
        url="https://example.com/pricing",
        title="Pricing",
        meta_description="Plans and pricing",
        text="Starter costs 19 dollars. Team costs 49 dollars.",
        word_count=9,
    )
]


class TestExtractJson:
    def test_plain_json(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_surrounding_prose(self) -> None:
        assert extract_json('Sure! Here is the report: {"a": 1} hope it helps') == {"a": 1}

    def test_garbage_raises(self) -> None:
        with pytest.raises(ValueError, match="JSON"):
            extract_json("no object here at all")


class TestMockAnalyst:
    def test_report_structure_and_citations(self) -> None:
        retrieval = [preview("pricing plans", [CHUNKS[0]]), preview("products", [CHUNKS[1]])]
        report = MockAnalyst().analyze("https://example.com", retrieval, PAGES)
        assert report.mode == "demo"
        assert len(report.sections) == 2
        for section in report.sections:
            assert section.points, "each section should extract at least one point"
            for point in section.points:
                assert point.citations
                assert point.citations[0] in {"p00-c0", "p01-c0"}

    def test_prefers_sentences_with_numbers(self) -> None:
        retrieval = [preview("pricing plans", [CHUNKS[0]])]
        report = MockAnalyst().analyze("https://example.com", retrieval, PAGES)
        point_text = report.sections[0].points[0].text
        assert "19 dollars" in point_text

    def test_deterministic(self) -> None:
        retrieval = [preview("pricing plans", [CHUNKS[0]])]
        r1 = MockAnalyst().analyze("https://example.com", retrieval, PAGES)
        r2 = MockAnalyst().analyze("https://example.com", retrieval, PAGES)
        d1, d2 = r1.model_dump(), r2.model_dump()
        d1.pop("generated_at"), d2.pop("generated_at")  # timestamps differ by design
        assert d1 == d2

    def test_executive_summary_mentions_page_count(self) -> None:
        retrieval = [preview("pricing plans", [CHUNKS[0]])]
        report = MockAnalyst().analyze("https://example.com", retrieval, PAGES)
        assert "1 pages" in report.executive_summary


class _ScriptedClient:
    """ChatClient stand-in returning queued replies."""

    model = "test-model"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[str] = []

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        self.calls.append(user)
        if not self.replies:
            raise AssertionError("scripted client ran out of replies")
        return self.replies.pop(0)


VALID_REPLY = json.dumps(
    {
        "executive_summary": "Meridian Flow sells logistics workflow software.",
        "sections": [
            {
                "id": "pricing",
                "title": "Pricing",
                "points": [
                    {
                        "text": "Starter costs $19/user/month",
                        "citations": ["p00-c0"],
                    },
                    {
                        "text": "Hallucinated claim with fake citation",
                        "citations": ["p99-c9"],
                    },
                ],
            }
        ],
    }
)


class TestLLMAnalyst:
    def test_valid_reply_parses_with_grounding_guard(self) -> None:
        client = _ScriptedClient([VALID_REPLY])
        retrieval = [preview("pricing plans", [CHUNKS[0]])]
        report = LLMAnalyst(client).analyze("https://example.com", retrieval, PAGES)
        assert report.mode == "live"
        assert report.model == "test-model"
        section = report.sections[0]
        assert section.id == "pricing"
        assert len(section.points) == 1  # fake-citation point discarded
        assert section.points[0].citations == ["p00-c0"]

    def test_invalid_then_valid_retries(self) -> None:
        client = _ScriptedClient(["utterly not json", VALID_REPLY])
        retrieval = [preview("pricing plans", [CHUNKS[0]])]
        report = LLMAnalyst(client).analyze("https://example.com", retrieval, PAGES)
        assert report.mode == "live"
        assert len(client.calls) == 2
        assert "not valid JSON" in client.calls[1]

    def test_persistent_failure_falls_back_to_extractive(self) -> None:
        client = _ScriptedClient(["garbage", "still garbage"])
        retrieval = [preview("pricing plans", [CHUNKS[0]])]
        report = LLMAnalyst(client).analyze("https://example.com", retrieval, PAGES)
        assert report.mode == "demo"  # extractive fallback engaged
        assert report.sections

    def test_unknown_section_ids_dropped(self) -> None:
        reply = json.dumps(
            {
                "executive_summary": "summary",
                "sections": [
                    {
                        "id": "nonexistent",
                        "title": "X",
                        "points": [{"text": "t", "citations": ["p00-c0"]}],
                    }
                ],
            }
        )
        client = _ScriptedClient([reply, VALID_REPLY])
        retrieval = [preview("pricing plans", [CHUNKS[0]])]
        report = LLMAnalyst(client).analyze("https://example.com", retrieval, PAGES)
        # first reply yields no known sections -> retry -> valid
        assert report.mode == "live"


class TestBuildAnalyst:
    def test_demo_settings_get_mock(self, default_settings) -> None:
        assert isinstance(build_analyst(default_settings), MockAnalyst)

    def test_live_settings_get_llm(self, live_settings) -> None:
        assert isinstance(build_analyst(live_settings), LLMAnalyst)


class TestAnalysisSections:
    def test_sections_have_unique_ids(self) -> None:
        ids = [sid for sid, _t, _q in ANALYSIS_SECTIONS]
        assert len(ids) == len(set(ids))
        assert "pricing" in ids
