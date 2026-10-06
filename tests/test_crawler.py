"""Tests for the crawling stage — no network access required.

A FakeFetcher stands in for HTTP: tests cover same-site filtering, depth
limits, page budgets, robots gating and link normalization.
"""

from __future__ import annotations

from site_rag_analyst.config import Settings
from site_rag_analyst.crawler.crawler import Crawler, extract_links, normalize_url
from site_rag_analyst.crawler.fetcher import DEMO_ORIGIN, DemoSiteFetcher
from site_rag_analyst.crawler.robots import RobotsGate
from site_rag_analyst.models import FetchedPage


class FakeFetcher:
    """Serves a scripted map of URL -> (status, html)."""

    def __init__(self, pages: dict[str, str], robots: str | None = None) -> None:
        self.pages = pages
        self.robots = robots
        self.fetched: list[str] = []

    def fetch(self, url: str, depth: int = 0) -> FetchedPage:
        self.fetched.append(url)
        if url.endswith("/robots.txt"):
            return FetchedPage(url=url, status=200, html=self.robots or "")
        status, html = self.pages.get(url, (404, ""))
        return FetchedPage(url=url, status=status, html=html, source="http")

    def fetch_robots(self, url: str) -> str | None:
        if self.robots is None:
            return None
        return self.robots


def _settings(**overrides: object) -> Settings:
    base = dict(
        _env_file=None,
        crawl_max_pages=10,
        crawl_max_depth=2,
        crawl_delay_seconds=0.0,
        respect_robots=True,
        user_agent="site-rag-analyst/0.1 (+test)",
    )
    base.update(overrides)
    return Settings(**base)


def _page(body: str = "<html><body><p>hello</p></body></html>") -> tuple[int, str]:
    return 200, body


HOME = "https://example.com/"
ABOUT = "https://example.com/about"
PRICING = "https://example.com/pricing"
BLOG_1 = "https://example.com/blog/post-1"
BLOG_2 = "https://example.com/blog/post-2"


class TestExtractLinks:
    def test_relative_links_resolved_against_base(self) -> None:
        html = '<a href="about">About</a> <a href="/pricing">Pricing</a>'
        assert set(extract_links(html, HOME)) == {ABOUT, PRICING}

    def test_fragments_and_non_http_dropped(self) -> None:
        html = (
            '<a href="#top">Top</a><a href="mailto:x@y.z">mail</a>'
            '<a href="about#team">team</a>'
        )
        assert set(extract_links(html, HOME)) == {ABOUT}

    def test_other_hosts_dropped(self) -> None:
        html = '<a href="https://other.com/x">ext</a><a href="/local">ok</a>'
        assert set(extract_links(html, HOME)) == {"https://example.com/local"}

    def test_binary_extensions_dropped(self) -> None:
        html = '<a href="/doc.pdf">pdf</a><a href="/img.png">img</a><a href="/page">ok</a>'
        assert set(extract_links(html, HOME)) == {"https://example.com/page"}


class TestNormalizeUrl:
    def test_fragment_stripped_and_host_lowercased(self) -> None:
        assert normalize_url("HTTPS://Example.COM/a?b=1#frag") == "https://example.com/a?b=1"

    def test_idempotent(self) -> None:
        once = normalize_url("https://example.com/x#y")
        assert normalize_url(once) == once


class TestRobotsGate:
    def test_disallow_is_respected(self) -> None:
        robots = "User-agent: *\nDisallow: /private/\n"
        gate = RobotsGate("site-rag-analyst/0.1", fetch_robots=lambda _url: robots)
        assert gate.allowed("https://example.com/ok") is True
        assert gate.allowed("https://example.com/private/x") is False

    def test_missing_robots_means_allowed(self) -> None:
        gate = RobotsGate("site-rag-analyst/0.1", fetch_robots=lambda _url: None)
        assert gate.allowed("https://example.com/anything") is True

    def test_respect_robots_false_disables_gate(self) -> None:
        robots = "User-agent: *\nDisallow: /\n"
        gate = RobotsGate(
            "site-rag-analyst/0.1", respect_robots=False, fetch_robots=lambda _u: robots
        )
        assert gate.allowed("https://example.com/") is True

    def test_non_http_always_allowed(self) -> None:
        gate = RobotsGate(
            "site-rag-analyst/0.1",
            fetch_robots=lambda _u: "User-agent: *\nDisallow: /\n",
        )
        # demo/file sources have no robots.txt concept
        assert gate.allowed("file:///tmp/x.html") is True


class TestCrawler:
    def _run(self, fetcher: FakeFetcher, **overrides: object):
        crawler = Crawler(
            fetcher=fetcher,
            settings=_settings(**overrides),
            robots_fetcher=fetcher.fetch_robots,
        )
        return crawler.crawl(HOME)

    def test_bfs_visits_linked_pages_within_depth(self) -> None:
        fetcher = FakeFetcher(
            {
                HOME: _page('<a href="/about">a</a>'),
                ABOUT: _page('<a href="/blog/post-1">b</a>'),
                BLOG_1: _page("<p>post</p>"),
            }
        )
        pages, stats = self._run(fetcher, crawl_max_depth=2)
        fetched = {p.url for p in pages if p.ok}
        assert fetched == {HOME, ABOUT, BLOG_1}
        assert stats.pages_fetched == 3

    def test_depth_limit_prunes_deep_pages(self) -> None:
        fetcher = FakeFetcher(
            {
                HOME: _page('<a href="/about">a</a>'),
                ABOUT: _page('<a href="/blog/post-1">b</a>'),
                BLOG_1: _page('<a href="/blog/post-2">c</a>'),
            }
        )
        # depth 0 -> 1 allowed; 1 -> 2 allowed; BLOG_2 would be depth 2->3? No:
        # HOME=0, ABOUT=1, BLOG_1=2, BLOG_2=3 -> pruned at max_depth=2
        pages, _stats = self._run(fetcher, crawl_max_depth=2)
        fetched = {p.url for p in pages}
        assert BLOG_2 not in fetched

    def test_page_budget_is_respected(self) -> None:
        fetcher = FakeFetcher({url: _page() for url in (HOME, ABOUT, PRICING, BLOG_1)})
        fetcher.pages[HOME] = _page('<a href="/about">a</a><a href="/pricing">p</a>')
        pages, stats = self._run(fetcher, crawl_max_pages=2)
        assert len(pages) == 2
        assert stats.pages_fetched == 2
        assert stats.urls_skipped_budget >= 1

    def test_failed_pages_are_recorded_not_fatal(self) -> None:
        fetcher = FakeFetcher({HOME: (500, ""), ABOUT: _page()})
        fetcher.pages[HOME] = (500, "")
        pages, stats = self._run(fetcher)
        assert stats.pages_failed == 1
        assert all(p.url == HOME for p in pages)

    def test_robots_blocks_disallowed_pages(self) -> None:
        fetcher = FakeFetcher(
            {
                HOME: _page('<a href="/pricing">p</a>'),
                PRICING: _page(),
            },
            robots="User-agent: *\nDisallow: /pricing\n",
        )
        pages, stats = self._run(fetcher)
        fetched = {p.url for p in pages}
        assert PRICING not in fetched
        assert stats.urls_skipped_robots == 1

    def test_crawling_is_bounded_to_one_host(self) -> None:
        fetcher = FakeFetcher(
            {HOME: _page('<a href="https://evil.com/x">e</a><a href="/ok">ok</a>')}
        )
        pages, _ = self._run(fetcher)
        fetched = {p.url for p in pages}
        assert all(u.startswith("https://example.com") for u in fetched)

    def test_duplicate_links_fetched_once(self) -> None:
        fetcher = FakeFetcher(
            {HOME: _page('<a href="/about">a</a><a href="about">a2</a><a href="/about#x">a3</a>'),
             ABOUT: _page()}
        )
        pages, _ = self._run(fetcher)
        assert len(pages) == 2
        assert fetcher.fetched.count(ABOUT) == 1


class TestDemoSiteFetcher:
    def test_maps_index_and_nested_paths(self, tmp_path) -> None:
        site = tmp_path / "site"
        site.mkdir()
        (site / "index.html").write_text("<h1>home</h1>", encoding="utf-8")
        (site / "pricing.html").write_text("<h1>pricing</h1>", encoding="utf-8")

        fetcher = DemoSiteFetcher(root=site)
        assert fetcher.fetch(DEMO_ORIGIN + "/").status == 200
        assert fetcher.fetch(DEMO_ORIGIN + "/pricing").status == 200
        assert fetcher.fetch(DEMO_ORIGIN + "/missing").status == 404

    def test_rejects_other_hosts(self, tmp_path) -> None:
        fetcher = DemoSiteFetcher(root=tmp_path)
        assert fetcher.fetch("https://not-demo.example.com/").status == 404
