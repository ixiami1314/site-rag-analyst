"""Pluggable page fetchers.

``Fetcher`` is a small protocol, so the network layer can be swapped without
touching the crawler:

- :class:`HttpFetcher` — httpx-based production fetcher (retries, timeouts,
  self-identifying User-Agent).
- :class:`DemoSiteFetcher` — serves the bundled fictional website from disk so
  demo mode runs with zero network access.
- a Playwright-based fetcher for JS-heavy sites would implement the same
  protocol (kept out of the default install to keep the Docker image lean).
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path
from typing import Protocol

import httpx

from site_rag_analyst.models import FetchedPage

logger = logging.getLogger(__name__)

DEMO_ORIGIN = "http://demo.meridian.test"


class Fetcher(Protocol):
    """Anything that can turn a URL into a FetchedPage."""

    def fetch(self, url: str, depth: int = 0) -> FetchedPage: ...


class HttpFetcher:
    """Fetches pages over HTTP(S) with retries and a descriptive UA."""

    def __init__(self, user_agent: str, timeout_seconds: float = 15.0) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept": "text/html,*/*"},
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
        )

    def fetch(self, url: str, depth: int = 0) -> FetchedPage:
        last_error: Exception | None = None
        for attempt in (1, 2):  # one retry on transport errors
            try:
                response = self._client.get(url)
                content_type = response.headers.get("content-type", "")
                html = response.text if "html" in content_type.lower() else ""
                return FetchedPage(
                    url=str(response.url),
                    status=response.status_code,
                    content_type=content_type,
                    html=html,
                    depth=depth,
                    source="http",
                )
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("fetch attempt %s failed for %s: %s", attempt, url, exc)
        return FetchedPage(
            url=url, depth=depth, source="http", error=str(last_error or "fetch failed")
        )

    def close(self) -> None:
        self._client.close()

    def fetch_text(self, url: str) -> str | None:
        """Fetch a small text resource (robots.txt); None on any failure."""
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            logger.debug("robots fetch failed for %s: %s", url, exc)
            return None
        if response.status_code != 200:
            return None
        return response.text


class DemoSiteFetcher:
    """Serves the bundled demo website (``demo_data/site``) as if it were live.

    Demo pages reference each other with absolute ``DEMO_ORIGIN`` URLs, so the
    crawler, robots gate and link extraction run unchanged in demo mode.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    def fetch(self, url: str, depth: int = 0) -> FetchedPage:
        path = self._map(url)
        if path is None:
            return FetchedPage(
                url=url,
                depth=depth,
                source="demo",
                status=404,
                error=f"no demo page for {url}",
            )
        return FetchedPage(
            url=url,
            status=200,
            content_type="text/html",
            html=path.read_text(encoding="utf-8"),
            depth=depth,
            source="demo",
        )

    # ------------------------------------------------------------------ #

    def _map(self, url: str) -> Path | None:
        """Map a demo URL to a file under the demo site root."""
        from urllib.parse import urlsplit

        parts = urlsplit(url)
        if parts.netloc and parts.netloc != DEMO_ORIGIN.split("://", 1)[1]:
            return None
        rel = parts.path.lstrip("/")
        if not rel or rel.endswith("/"):
            rel += "index.html"
        if not rel.endswith(".html"):
            rel += ".html" if "." not in rel.rsplit("/", 1)[-1] else ""

        if self._root is not None:
            candidate = self._root / rel
            return candidate if candidate.is_file() else None
        resource = resources.files("site_rag_analyst").joinpath(f"demo_data/site/{rel}")
        if resource.is_file():
            return Path(str(resource))
        return None
