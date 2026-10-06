"""Eval harness implementation.

A case is (query, expected_url, expected_snippet?):

- **hit@k** — any top-k chunk comes from the expected page.
- **snippet@k** — any top-k chunk from the expected page also contains the
  expected snippet text (strict: the right *part* of the right page).
- **MRR** — reciprocal rank of the first hit.

The default set (``eval_set.jsonl``) targets concrete facts planted in the
demo corpus: prices, retention periods, rate limits, founder names. Those
are exactly the questions a downstream analyst gets asked, so the numbers
translate.
"""

from __future__ import annotations

import json
import sys
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, Field

from site_rag_analyst.chunking.chunker import Chunker
from site_rag_analyst.config import Settings
from site_rag_analyst.crawler.crawler import Crawler
from site_rag_analyst.crawler.fetcher import DEMO_ORIGIN, DemoSiteFetcher
from site_rag_analyst.embedding.base import build_provider, embedding_text
from site_rag_analyst.extraction.extractor import ContentExtractor
from site_rag_analyst.models import ExtractedPage
from site_rag_analyst.retrieval.retriever import Retriever
from site_rag_analyst.store.vecstore import VectorStore, open_store


class EvalCase(BaseModel):
    query: str
    expected_url: str  # substring of the chunk's page_url
    expected_snippet: str = ""


class CaseResult(BaseModel):
    query: str
    expected_url: str
    first_rank: int | None = Field(description="1-based rank of first page hit")
    hit_at_3: bool
    hit_at_5: bool
    snippet_hit_at_5: bool
    top_pages: list[str] = Field(default_factory=list)


class EvalSummary(BaseModel):
    cases: int
    hit_at_3: float
    hit_at_5: float
    snippet_hit_at_5: float
    mrr: float


def load_eval_set(path: Path | None = None) -> list[EvalCase]:
    """Load eval cases (defaults to the bundled set shipped with the package)."""
    if path is None:
        text = resources.files("site_rag_analyst").joinpath(
            "eval/eval_set.jsonl"
        ).read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return [
        EvalCase.model_validate(json.loads(line))
        for line in text.splitlines()
        if line.strip()
    ]


def build_demo_corpus(settings: Settings, store_path: Path) -> tuple[VectorStore, object]:
    """Crawl the bundled demo site into a fresh store; return (store, provider)."""
    crawl_settings = settings.model_copy(
        update={"crawl_max_pages": 50, "crawl_max_depth": 3, "crawl_delay_seconds": 0.0}
    )
    pages, _stats = Crawler(
        fetcher=DemoSiteFetcher(), settings=crawl_settings
    ).crawl(DEMO_ORIGIN + "/")

    extractor = ContentExtractor()
    chunker = Chunker(
        target_words=settings.chunk_target_words,
        overlap_words=settings.chunk_overlap_words,
    )
    extracted: list[ExtractedPage] = []
    chunks = []
    for page in pages:  # type: FetchedPage
        if not page.ok:
            continue
        doc = extractor.extract(page)
        if doc.word_count < 25:
            continue
        extracted.append(doc)
    for page_index, doc in enumerate(extracted):
        chunks.extend(chunker.chunk_page(doc, page_index=page_index))

    provider = build_provider(settings)
    texts = [embedding_text(c) for c in chunks]
    provider.fit(texts)
    store = open_store(store_path, dim=None)
    store.add(chunks, provider.embed(texts))
    return store, provider


def evaluate_cases(
    retriever: Retriever, cases: list[EvalCase], k_max: int = 5
) -> tuple[list[CaseResult], EvalSummary]:
    results: list[CaseResult] = []
    for case in cases:
        retrieved = retriever.search(case.query, k=k_max, max_per_page=None)
        top_pages = [r.chunk.page_url for r in retrieved]
        first_rank: int | None = None
        for rank, url in enumerate(top_pages, start=1):
            if case.expected_url in url:
                first_rank = rank
                break
        snippet_hit = False
        if case.expected_snippet:
            for r in retrieved:
                if (
                    case.expected_url in r.chunk.page_url
                    and case.expected_snippet.lower() in r.chunk.text.lower()
                ):
                    snippet_hit = True
                    break
        results.append(
            CaseResult(
                query=case.query,
                expected_url=case.expected_url,
                first_rank=first_rank,
                hit_at_3=bool(first_rank and first_rank <= 3),
                hit_at_5=bool(first_rank and first_rank <= 5),
                snippet_hit_at_5=snippet_hit,
                top_pages=[u.replace(DEMO_ORIGIN, "") for u in top_pages[:3]],
            )
        )
    n = len(results)
    summary = EvalSummary(
        cases=n,
        hit_at_3=sum(r.hit_at_3 for r in results) / n,
        hit_at_5=sum(r.hit_at_5 for r in results) / n,
        snippet_hit_at_5=sum(r.snippet_hit_at_5 for r in results) / n,
        mrr=sum((1.0 / r.first_rank) if r.first_rank else 0.0 for r in results) / n,
    )
    return results, summary


def run_evaluation(settings: Settings, store_path: Path) -> tuple[list[CaseResult], EvalSummary]:
    store, provider = build_demo_corpus(settings, store_path)
    try:
        retriever = Retriever(store, provider)
        return evaluate_cases(retriever, load_eval_set())
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    import tempfile

    if Path(".env").exists():
        from site_rag_analyst.config import get_settings

        settings = get_settings()
    else:
        settings = Settings(_env_file=None)
    with tempfile.TemporaryDirectory() as tmp:
        results, summary = run_evaluation(settings, Path(tmp) / "eval.sqlite3")

    print(f"{'query':52s} {'rank':>4s}  hit@3 hit@5 snip@5  top page")
    for r in results:
        rank = str(r.first_rank) if r.first_rank else "-"
        flags = (
            f"{'Y' if r.hit_at_3 else '.'}"
            f"{'Y' if r.hit_at_5 else '.'}"
            f"{'Y' if r.snippet_hit_at_5 else '.'}"
        )
        top = r.top_pages[0] if r.top_pages else ""
        print(f"{r.query[:52]:52s} {rank:>4s}  {flags[0]:5s} {flags[1]:5s} {flags[2]:6s}  {top}")
    print()
    print(
        f"cases={summary.cases}  hit@3={summary.hit_at_3:.2f}  hit@5={summary.hit_at_5:.2f} "
        f" snippet@5={summary.snippet_hit_at_5:.2f}  MRR={summary.mrr:.2f}"
    )
    print("(provider:", "tfidf offline — demo mode)" if settings.demo_mode else "live)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
