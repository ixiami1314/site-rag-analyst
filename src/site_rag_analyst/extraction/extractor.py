"""Main-content extraction: turn messy HTML into clean, structure-aware text.

A RAG pipeline lives or dies on its noise floor. Navigation chrome, cookie
banners and footers that survive into chunks poison retrieval — a query about
"Pricing" should not rank a nav link above the pricing page's content. This
is a deliberately small, dependency-light extractor (BeautifulSoup only):

1. strip ``<script>/<style>/<nav>/<header>/<footer>/<aside>/<form>`` and friends
2. prefer a semantic container (``<main>``, ``<article>``, ``role=main``)
3. otherwise score direct children of ``<body>`` by text density and keep the
   top region (classic readability intuition, ~40 lines)
4. render text with heading markers (``##``-prefixed) so the chunker can
   rebuild the heading path for every chunk

For most marketing/docs sites this removes 60-80% of raw text words — see
``ExtractedPage.kept_ratio``. Heavyweight JS SPAs are out of scope for the
httpx fetcher; a Playwright fetcher would plug in upstream of this stage.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from site_rag_analyst.models import ExtractedPage, FetchedPage

# Tags whose entire subtree is boilerplate for our purposes.
DROP_TAGS = (
    "script", "style", "noscript", "nav", "header", "footer", "aside",
    "form", "iframe", "svg", "button", "select", "template",
)

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_RE = re.compile(r"\n\s*\n+")


def _clean_text(fragment: Tag | BeautifulSoup) -> str:
    """Render a fragment to plain text; headings become '##'-prefixed lines."""
    lines: list[str] = []
    content_tags = ["h1", "h2", "h3", "h4", "p", "li", "td", "pre", "blockquote", "figcaption"]
    for element in fragment.find_all(content_tags):
        if isinstance(element, Tag):
            text = _WS_RE.sub(" ", element.get_text(" ")).strip()
            if not text:
                continue
            level = _HEADING_TAGS.get(element.name, 0)
            lines.append(f"{'#' * (level + 1) if level else ''} {text}".strip())
    return _BLANK_RE.sub("\n\n", "\n".join(lines)).strip()


def _text_words(fragment: Tag | BeautifulSoup) -> int:
    return len(fragment.get_text(" ", strip=True).split())


def _pick_main(soup: BeautifulSoup) -> Tag | None:
    """Choose the DOM region most likely to hold the main content."""
    for candidate in soup.find_all("main") or []:
        if isinstance(candidate, Tag) and _text_words(candidate) > 50:
            return candidate
    article = soup.find("article")
    if isinstance(article, Tag) and _text_words(article) > 50:
        return article
    role_main = soup.find(attrs={"role": "main"})
    if isinstance(role_main, Tag) and _text_words(role_main) > 50:
        return role_main

    body = soup.body
    if body is None:
        return None
    # No semantic container: score body children by text density
    # (link-heavy blocks like navbars score low even if not <nav>).
    best: Tag | None = None
    best_score = 0.0
    for child in body.find_all(recursive=False):
        if not isinstance(child, Tag):
            continue
        words = _text_words(child)
        if words < 40:
            continue
        link_words = _text_words_of_links(child)
        density = words / (1.0 + link_words)
        score = words * density
        if score > best_score:
            best, best_score = child, score
    return best or body


def _text_words_of_links(fragment: Tag) -> int:
    total = 0
    for anchor in fragment.find_all("a"):
        total += len(anchor.get_text(" ", strip=True).split())
    return total


class ContentExtractor:
    """Extract :class:`ExtractedPage` artifacts from fetched pages."""

    def extract(self, page: FetchedPage, links_discovered: int = 0) -> ExtractedPage:
        soup = BeautifulSoup(page.html or "", "lxml")
        # Baseline word count BEFORE boilerplate removal, so kept_ratio
        # reflects how much of the raw page survived into the corpus.
        raw_words = len(soup.get_text(" ", strip=True).split()) if soup.body is not None else 0
        for tag in soup.find_all(DROP_TAGS):
            tag.decompose()

        title = ""
        if soup.title is not None and soup.title.string:
            title = soup.title.string.strip()
        meta_description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if isinstance(meta_desc, Tag) and meta_desc.get("content"):
            meta_description = str(meta_desc["content"]).strip()

        main = _pick_main(soup)
        if main is None:
            return ExtractedPage(
                url=page.url,
                title=title,
                meta_description=meta_description,
                text="",
                word_count=0,
                links_discovered=links_discovered,
                kept_ratio=0.0,
                fetch_status=page.status,
            )

        text = _clean_text(main)
        headings = [
            _WS_RE.sub(" ", h.get_text(" ")).strip()
            for h in main.find_all(["h1", "h2", "h3"])
            if h.get_text(strip=True)
        ]
        word_count = len(text.split())
        return ExtractedPage(
            url=page.url,
            title=title,
            meta_description=meta_description,
            headings=headings,
            text=text,
            word_count=word_count,
            links_discovered=links_discovered,
            kept_ratio=round(word_count / raw_words, 3) if raw_words else 0.0,
            fetch_status=page.status,
        )
