"""Command-line entry point: ``site-rag [URL | demo]``.

Examples::

    site-rag demo                       # bundled site, fully offline
    site-rag https://example.com        # live crawl (demo mode without keys)
    site-rag demo --json                # also dump report JSON to stdout
"""

from __future__ import annotations

import argparse
import logging
import sys

from site_rag_analyst.crawler.fetcher import DEMO_ORIGIN
from site_rag_analyst.pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="site-rag",
        description="Crawl a website, ground an LLM in it, produce a cited report.",
    )
    parser.add_argument(
        "target",
        help="site URL to analyze, or the literal 'demo' for the bundled demo site",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None, help="override CRAWL_MAX_PAGES"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the report JSON to stdout"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from site_rag_analyst.config import get_settings

    settings = get_settings()
    if args.max_pages is not None:
        settings.crawl_max_pages = args.max_pages

    target = DEMO_ORIGIN + "/" if args.target.lower() == "demo" else args.target
    print(f"Analyzing {target} (mode: {'demo' if settings.demo_mode else 'live'})",
          file=sys.stderr)
    result = Pipeline(settings).run(target)
    if not result.ok:
        print(f"Run failed: {result.error}", file=sys.stderr)
        return 1

    if args.json and result.report is not None:
        print(result.report.model_dump_json(indent=2))

    print(
        f"\nReport ready: {settings.output_dir / result.run_id / 'report.md'}\n"
        f"  pages: {len(result.pages)}  chunks: {len(result.chunks)}  "
        f"sections: {len(result.report.sections if result.report else [])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
