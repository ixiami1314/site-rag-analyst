"""Tests for the extraction stage (no network; inline HTML fixtures)."""

from __future__ import annotations

from site_rag_analyst.extraction.extractor import ContentExtractor
from site_rag_analyst.models import FetchedPage


def make_page(html: str, url: str = "https://example.com/") -> FetchedPage:
    return FetchedPage(url=url, status=200, content_type="text/html", html=html)


NAV = """
<nav><a href="/">Home</a> <a href="/pricing">Pricing</a>
<a href="/docs">Docs</a> <a href="/blog">Blog</a> <a href="/about">About</a></nav>
"""

BOILERPLATE = """
<div class="cookie-banner"><button>Accept all cookies</button></div>
<footer>© 2026 Example Corp. All rights reserved. <a href="/tos">Terms</a></footer>
<script>console.log('tracking junk that must never reach a chunk');</script>
"""


class TestBoilerplateRemoval:
    def test_script_style_nav_footer_removed(self) -> None:
        html = f"""<html><body>{NAV}{BOILERPLATE}
        <main><h1>Real headline</h1><p>Real paragraph about widgets.</p></main>
        </body></html>"""
        result = ContentExtractor().extract(make_page(html))
        assert "tracking junk" not in result.text
        assert "cookie" not in result.text.lower()
        assert "Real paragraph about widgets." in result.text

    def test_nav_links_do_not_leak_into_text(self) -> None:
        html = f"<html><body>{NAV}<main><p>Content only.</p></main></body></html>"
        result = ContentExtractor().extract(make_page(html))
        assert "Pricing" not in result.text  # nav anchor text is boilerplate


class TestContainerSelection:
    def test_main_tag_preferred(self) -> None:
        html = f"""<html><body><div>filler filler filler filler filler</div>
        {NAV}<main><h1>Head</h1><p>{'word ' * 60}</p></main></body></html>"""
        result = ContentExtractor().extract(make_page(html))
        assert "Head" in result.text
        assert result.word_count >= 55

    def test_article_tag_used_when_no_main(self) -> None:
        html = f"""<html><body>{NAV}
        <article><h2>Post title</h2><p>{'content ' * 80}</p></article>
        </body></html>"""
        result = ContentExtractor().extract(make_page(html))
        assert "Post title" in result.text
        assert result.word_count > 50

    def test_dense_div_wins_without_semantic_markup(self) -> None:
        html = f"""<html><body>
        <div class="menu"><a href="/a">alpha</a> <a href="/b">beta</a>
        <a href="/c">gamma</a> <a href="/d">delta</a> <a href="/e">epsilon</a></div>
        <div class="content"><p>{'genuine content words here ' * 20}</p></div>
        </body></html>"""
        result = ContentExtractor().extract(make_page(html))
        assert "genuine content words here" in result.text
        assert "alpha" not in result.text


class TestMetadata:
    def test_title_and_meta_description_extracted(self) -> None:
        html = """<html><head><title>Example — Widgets</title>
        <meta name="description" content="We build widgets for teams.">
        </head><body><main><h1>Widgets</h1><p>word word word</p></main></body></html>"""
        result = ContentExtractor().extract(make_page(html))
        assert result.title == "Example — Widgets"
        assert result.meta_description == "We build widgets for teams."

    def test_headings_collected_in_order(self) -> None:
        html = """<html><body><main>
        <h1>Top</h1><h2>Second</h2><h3>Third</h3><p>text</p>
        </main></body></html>"""
        result = ContentExtractor().extract(make_page(html))
        assert result.headings == ["Top", "Second", "Third"]

    def test_heading_markers_preserved_in_text(self) -> None:
        html = """<html><body><main>
        <h2>Pricing</h2><p>Plans from 19 dollars.</p>
        </main></body></html>"""
        result = ContentExtractor().extract(make_page(html))
        assert "## Pricing" in result.text
        assert "Plans from 19 dollars." in result.text


class TestStatsAndEdges:
    def test_kept_ratio_between_zero_and_one(self) -> None:
        html = f"""<html><body>{NAV}<main><p>{'body words ' * 100}</p></main>
        </body></html>"""
        result = ContentExtractor().extract(make_page(html))
        assert 0.0 < result.kept_ratio <= 1.0

    def test_empty_html_yields_empty_page_not_crash(self) -> None:
        result = ContentExtractor().extract(make_page(""))
        assert result.word_count == 0
        assert result.text == ""
        assert result.kept_ratio == 0.0

    def test_list_items_render_as_lines(self) -> None:
        html = """<html><body><main><h2>Features</h2>
        <ul><li>Fast ingestion</li><li>Cited answers</li></ul>
        </main></body></html>"""
        result = ContentExtractor().extract(make_page(html))
        assert "Fast ingestion" in result.text
        assert "Cited answers" in result.text
