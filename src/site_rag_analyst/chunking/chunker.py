"""Structure-aware chunking with bounded overlap.

Why not fixed-size windows: a chunk that cuts mid-section loses the heading
context that makes retrieval work — "Teams plan costs $49/user" is only
answerable if the chunk also carries the "Pricing" heading. So chunks are cut
on structure first, then packed to a word budget:

1. split extracted text into sections along heading markers (``##`` from the
   extractor) and track each section's heading path
2. oversized sections are split further at sentence boundaries
3. consecutive pieces are packed into chunks up to ``target`` words; when a
   single section spans a chunk boundary, the next chunk opens with the last
   ``overlap`` words of the previous chunk so no fact is lost at a seam
4. small sections merge into one chunk (token efficiency) — the chunk's
   ``heading_path`` is the heading of its first content

Word counts approximate tokens at ~0.75 words/token for English, so the
default 280-word target lands near 350-400 token chunks, which is in the
sweet spot we found for website corpora (see README "Design decisions").
"""

from __future__ import annotations

import re

from site_rag_analyst.models import Chunk, ExtractedPage

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING_RE = re.compile(r"^(#{2,5})\s+(.*)$")


class _Section:
    __slots__ = ("heading_path", "lines")

    def __init__(self, heading_path: str) -> None:
        self.heading_path = heading_path
        self.lines: list[str] = []


def _parse_sections(text: str, fallback_heading: str) -> list[_Section]:
    """Split extractor output into sections along heading markers."""
    sections: list[_Section] = []
    levels: dict[int, str] = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _HEADING_RE.match(line)
        if match:
            depth = len(match.group(1)) - 1  # '##' -> level 1
            levels[depth] = match.group(2).strip()
            # deeper levels from an earlier sibling no longer apply
            levels = {d: t for d, t in levels.items() if d <= depth}
            heading_path = " > ".join(levels[d] for d in sorted(levels))
            current = _Section(heading_path)
            sections.append(current)
        else:
            if current is None:
                current = _Section(fallback_heading)
                sections.append(current)
            current.lines.append(line)
    return sections


def _split_pieces(section: _Section, target_words: int) -> list[str]:
    """Sentence-aware pieces, each <= target words."""
    text = "\n".join(section.lines)
    if len(text.split()) <= target_words:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    pieces: list[str] = []
    buffer: list[str] = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) > target_words:  # pathological run-on sentence: hard split
            if buffer:
                pieces.append(" ".join(buffer))
                buffer, count = [], 0
            for start in range(0, len(words), target_words):
                pieces.append(" ".join(words[start : start + target_words]))
            continue
        if count + len(words) > target_words and buffer:
            pieces.append(" ".join(buffer))
            buffer, count = [], 0
        buffer.extend(words)
        count += len(words)
    if buffer:
        pieces.append(" ".join(buffer))
    return pieces


class Chunker:
    """Turns extracted pages into citation-ready chunks."""

    def __init__(self, target_words: int = 280, overlap_words: int = 45) -> None:
        if overlap_words >= target_words:
            raise ValueError("overlap_words must be smaller than target_words")
        self._target = target_words
        self._overlap = overlap_words

    def chunk_page(self, page: ExtractedPage, page_index: int) -> list[Chunk]:
        chunks: list[Chunk] = []

        def emit(heading: str, text: str) -> None:
            cleaned = text.strip()
            if not cleaned:
                return
            chunks.append(
                Chunk(
                    id=f"p{page_index:02d}-c{len(chunks)}",
                    page_url=page.url,
                    page_title=page.title,
                    heading_path=heading,
                    index=len(chunks),
                    text=cleaned,
                    word_count=len(cleaned.split()),
                )
            )

        sections = _parse_sections(page.text, fallback_heading="")
        current: list[str] = []
        current_words = 0
        current_heading = ""

        for section in sections:
            pieces = _split_pieces(section, self._target)
            for piece_index, piece in enumerate(pieces):
                piece_words = len(piece.split())
                if current and current_words + piece_words > self._target:
                    emit(current_heading, "\n".join(current))
                    current, current_words = [], 0
                    if piece_index > 0:
                        # The section is split mid-way: carry an overlap tail
                        # so facts straddling the seam stay retrievable.
                        tail = " ".join(pieces[piece_index - 1].split()[-self._overlap :])
                        if tail:
                            current.append(tail)
                            current_words += len(tail.split())
                            current_heading = section.heading_path
                if not current:
                    current_heading = section.heading_path
                current.append(piece)
                current_words += piece_words

        if current:
            emit(current_heading, "\n".join(current))
        return chunks
