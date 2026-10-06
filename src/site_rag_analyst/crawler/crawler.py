"""Same-site breadth-first crawler.

Boundaries and politeness are enforced here:

- **Same site only** — links are normalized, de-fragmented and filtered to the
  starting host, so one run never leaks onto third-party sites.
- **Budgets** — a page cap and a depth cap keep runs predictable.
- **robots.txt** — checked before every fetch (see :mod:`.robots`).
- **Rate limit** — a configurable delay between requests.

The crawler is fetcher-agnostic; tests inject a scripted fetcher, demo mode
serves pages from disk, and production uses :class:`~crawler.fetcher.HttpFetcher`.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from os.path import splitext
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from site_rag_analyst.config import Settings
from site_rag_analyst.crawler.robots import RobotsGate
from site_rag_analyst.models import CrawlStats, FetchedPage

logger = logging.getLogger(__name__)

# Link targets we never follow as content pages.
SKIP_EXTENSIONS = {
    ".css", ".gif", ".gz", ".ico", ".jpeg", ".jpg", ".js", ".json", ".mp3",
    ".mp4", ".pdf", ".png", ".rss", ".svg", ".tar", ".webp", ".xml", ".zip",
}


def normalize_url(url: str) -> str:
    """Canonical URL form: lowercase scheme/host, fragment stripped.

    Idempotent, so it is safe to call on already-normalized URLs.
    """
    url = url.strip()
    parts = urlsplit(url)
    if not parts.scheme:
        return url
    path_and_query = parts.path or "/"
    if parts.query:
        path_and_query += f"?{parts.query}"
    return urljoin(f"{parts.scheme.lower()}://{parts.netloc.lower()}", path_and_query)


def extract_links(html: str, base_url: str) -> list[str]:
    """Same-host, normalized links from a page, in document order, deduped."""
    soup = BeautifulSoup(html, "lxml")
    base_host = urlsplit(normalize_url(base_url)).netloc.lower()
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href.startswith("#"):
            continue  # same-page anchor; the page itself is always crawled
        url = normalize_url(urljoin(base_url, href))
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            continue
        if parts.netloc.lower() != base_host:
            continue
        if splitext(parts.path)[1].lower() in SKIP_EXTENSIONS:
            continue
        if url not in seen:
            seen.add(url)
            links.append(url)
    return links


class Crawler:
    """BFS crawler bounded by the settings' page/depth budgets."""

    def __init__(
        self,
        fetcher: object,
        settings: Settings,
        robots_fetcher: Callable[[str], str | None] | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._settings = settings
        # Prefer an explicit robots fetcher; fall back to the fetcher's own
        # text method when it has one (HttpFetcher does; demo sources don't).
        if robots_fetcher is None:
            robots_fetcher = getattr(fetcher, "fetch_text", None)
        self._robots = RobotsGate(
            user_agent=settings.user_agent,
            respect_robots=settings.respect_robots,
            fetch_robots=robots_fetcher,
        )

    def crawl(self, start_url: str) -> tuple[list[FetchedPage], CrawlStats]:
        """Crawl from ``start_url``; returns fetched pages (incl. failures)."""
        start = normalize_url(start_url)
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        queued: set[str] = {start}
        pages: list[FetchedPage] = []
        stats = CrawlStats(start_url=start)
        started = time.perf_counter()
        last_request_at = 0.0

        while queue:
            if len(pages) >= self._settings.crawl_max_pages:
                stats.urls_skipped_budget += len(queue)
                logger.info("page budget reached (%s)", self._settings.crawl_max_pages)
                break

            url, depth = queue.popleft()
            if not self._robots.allowed(url):
                stats.urls_skipped_robots += 1
                logger.debug("robots disallows %s", url)
                continue

            delay = self._settings.crawl_delay_seconds
            if delay > 0 and last_request_at:
                remaining = delay - (time.perf_counter() - last_request_at)
                if remaining > 0:
                    time.sleep(remaining)
            last_request_at = time.perf_counter()

            page = self._fetcher.fetch(url, depth=depth)
            pages.append(page)
            if page.ok:
                stats.pages_fetched += 1
                if depth < self._settings.crawl_max_depth:
                    for link in extract_links(page.html, url):
                        if link not in queued:
                            queued.add(link)
                            queue.append((link, depth + 1))
            else:
                stats.pages_failed += 1
                logger.info("fetch failed (%s): %s", url, page.error or page.status)

        stats.elapsed_seconds = time.perf_counter() - started
        return pages, stats
