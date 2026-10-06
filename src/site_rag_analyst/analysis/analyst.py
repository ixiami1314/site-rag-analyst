"""Analyst layer: turn retrieval results into a cited SiteReport.

Two interchangeable implementations:

- :class:`LLMAnalyst` — sends the retrieved context (with chunk ids) to an
  OpenAI-compatible chat endpoint and demands a JSON report in which every
  point cites chunk ids. Unknown citation ids are stripped — the report can
  only reference context that was actually retrieved.
- :class:`MockAnalyst` — deterministic extractive analysis used in demo mode
  and as the fallback when an LLM reply cannot be parsed.

Both consume the same inputs (per-section retrieval previews) and emit the
same model, so swapping them is a configuration concern, not a code change.
"""

from __future__ import annotations

import logging
from typing import Protocol

from site_rag_analyst.analysis.llm import ChatClient, extract_json
from site_rag_analyst.analysis.mock import MockAnalyst
from site_rag_analyst.config import Settings
from site_rag_analyst.models import (
    CitedPoint,
    ExtractedPage,
    ReportSection,
    RetrievalPreview,
    SiteReport,
)

logger = logging.getLogger(__name__)

# The analysis dimensions. Each drives one retrieval query; the union of
# results is the LLM's (or the extractor's) grounding context.
ANALYSIS_SECTIONS: list[tuple[str, str, str]] = [
    (
        "overview",
        "Site Overview",
        "what is this website about, its purpose and its main offerings",
    ),
    (
        "products",
        "Products & Features",
        "products, features, capabilities and technical details described on the site",
    ),
    (
        "pricing",
        "Pricing",
        "pricing plans, costs, tiers, trials, discounts and billing terms",
    ),
    (
        "audience",
        "Audience & Use Cases",
        "who the products are for, target customers and real-world use cases",
    ),
    (
        "company",
        "Company & Contact",
        "company background, team, contact channels and support options",
    ),
]

_SYSTEM_PROMPT = """\
You are a precise website analyst. You work strictly from the context chunks
provided — never invent facts, numbers or page names. Every claim you make
must cite the chunk id(s) that support it, in the format of the JSON schema
below. Respond with a single JSON object and nothing else.
"""

_USER_TEMPLATE = """\
Analyze the website <site>{site_url}</site> using only the context chunks below.

<context>
{context}
</context>

Return JSON with exactly this shape:
{{
  "executive_summary": "2-4 sentence summary of what this site is and offers",
  "sections": [
    {{
      "id": "{section_ids}",
      "title": "human title",
      "points": [
        {{"text": "specific, factual claim", "citations": ["p00-c1"]}}
      ]
    }}
  ]
}}

Rules:
- Produce one section for each of these ids, in order: {section_ids}
  ({section_legend}).
- 2-4 points per section; skip a section (empty points) only if the context
  truly contains nothing relevant.
- Every point cites at least one chunk id from <context>. Unciteable claims
  will be discarded.
- Prefer concrete facts: numbers, product names, plan tiers, integrations.
"""


class Analyst(Protocol):
    def analyze(
        self,
        site_url: str,
        retrieval: list[RetrievalPreview],
        pages: list[ExtractedPage],
    ) -> SiteReport: ...


class LLMAnalyst:
    """Retrieval-grounded analysis via an OpenAI-compatible chat endpoint."""

    def __init__(self, client: ChatClient) -> None:
        self._client = client
        self._fallback = MockAnalyst()

    def analyze(
        self,
        site_url: str,
        retrieval: list[RetrievalPreview],
        pages: list[ExtractedPage],
    ) -> SiteReport:
        context, known_ids = self._build_context(retrieval)
        if not context:
            logger.warning("no retrieved context; returning extractive fallback")
            return self._fallback.analyze(site_url, retrieval, pages)

        user_prompt = self._render_prompt(site_url, context)
        for attempt in (1, 2):
            failure = ""
            try:
                reply = self._client.complete(_SYSTEM_PROMPT, user_prompt)
                report = self._parse_report(site_url, reply, retrieval, known_ids)
                if report is not None:
                    return report
                failure = "reply did not validate against the required shape"
            except (RuntimeError, ValueError) as exc:
                failure = str(exc)
            logger.warning("LLM analysis attempt %s failed (%s)", attempt, failure)
            user_prompt = (
                "Your previous reply was not valid JSON with the required shape. "
                "Return ONLY the JSON object, no prose, no code fences.\n\n" + user_prompt
            )

        return self._fallback.analyze(site_url, retrieval, pages)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_context(
        retrieval: list[RetrievalPreview],
    ) -> tuple[str, set[str]]:
        lines: list[str] = []
        seen: set[str] = set()
        for preview in retrieval:
            for item in preview.results:
                chunk = item.chunk
                if chunk.id in seen:
                    continue
                seen.add(chunk.id)
                heading = chunk.heading_path or chunk.page_title or chunk.page_url
                lines.append(
                    f"[{chunk.id}] ({heading}, score {item.score:.2f}) {chunk.text}"
                )
        return "\n".join(lines), seen

    def _render_prompt(self, site_url: str, context: str) -> str:
        section_ids = ", ".join(f'"{sid}"' for sid, _title, _q in ANALYSIS_SECTIONS)
        legend = "; ".join(f"{sid} = {title}" for sid, title, _q in ANALYSIS_SECTIONS)
        return _USER_TEMPLATE.format(
            site_url=site_url,
            context=context,
            section_ids=section_ids,
            section_legend=legend,
        )

    def _parse_report(
        self,
        site_url: str,
        reply: str,
        retrieval: list[RetrievalPreview],
        known_ids: set[str],
    ) -> SiteReport | None:
        data = extract_json(reply)
        summary = data.get("executive_summary")
        if not isinstance(summary, str) or not summary.strip():
            return None
        sections: list[ReportSection] = []
        section_lookup = {sid: (title, query) for sid, title, query in ANALYSIS_SECTIONS}
        for raw in data.get("sections", []):
            sid = str(raw.get("id", "")).strip()
            if sid not in section_lookup:
                continue
            title, query = section_lookup[sid]
            points: list[CitedPoint] = []
            for raw_point in raw.get("points", []):
                text = str(raw_point.get("text", "")).strip()
                citations = [
                    str(c)
                    for c in raw_point.get("citations", [])
                    if str(c) in known_ids  # grounding guard: known ids only
                ]
                if text and citations:
                    points.append(CitedPoint(text=text, citations=citations))
            sections.append(ReportSection(id=sid, title=title, query=query, points=points))
        if not sections:
            return None
        return SiteReport(
            site_url=site_url,
            mode="live",
            model=self._client.model,
            executive_summary=summary.strip(),
            sections=sections,
            pages=[],  # the pipeline attaches page summaries with chunk counts
        )


def build_analyst(settings: Settings) -> Analyst:
    """Demo mode -> MockAnalyst; configured endpoint -> LLMAnalyst."""
    if settings.demo_mode:
        return MockAnalyst()
    client = ChatClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    return LLMAnalyst(client)
