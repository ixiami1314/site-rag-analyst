"""Offline TF-IDF embeddings.

Demo mode (and the bundled retrieval evals) run with zero network access, so
we ship a small deterministic lexical embedding: sublinear TF x IDF, L2
normalized, computed over the corpus vocabulary. On small website corpora
this is genuinely competitive with API embeddings for the kind of factual
queries an analyst asks ("what does the team plan cost") — the eval harness
exists precisely to keep that claim honest.

The vector dimension equals the corpus vocabulary size, which is only known
after ``fit``; that is why TF-IDF pairs with the numpy brute-force vector
store rather than the fixed-width sqlite-vec table.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Compact English stopword list — enough to keep nav-ish filler from
# dominating small corpora without butchering real product terms.
# (Kept as a split string for readability; SIM905 wants a one-line list.)
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from get got
    had has have how i if in into is it its just like may me might more most
    must my no not of on or our out over own said say says she should so some
    such than that the their them then there these they this to too us use
    used using very was we were what when where which while who why will with
    would you your yours
    """.split()  # noqa: SIM905 - kept readable as a word cloud
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class TfidfEmbeddings:
    """Deterministic TF-IDF vectors over the corpus vocabulary."""

    name = "tfidf"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None

    # ------------------------------------------------------------------ #

    def fit(self, texts: list[str]) -> None:
        """Build vocabulary and IDF over the corpus embedding texts."""
        doc_counts: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for text in texts:
            counts = Counter(tokenize(text))
            doc_counts.append(counts)
            df.update(counts.keys())
        if not doc_counts:
            doc_counts = [Counter()]
            df = Counter()

        self._vocab = {term: i for i, term in enumerate(sorted(df))}
        n_docs = len(doc_counts)
        self._idf = np.array(
            [math.log((n_docs + 1) / (df[term] + 1)) + 1.0 for term in sorted(df)],
            dtype=np.float64,
        )

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if self._idf is None:
            raise RuntimeError("fit() must run before embed()")
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        if self._idf is None:
            raise RuntimeError("fit() must run before embed_query()")
        return self._vector(text)

    @property
    def dim(self) -> int:
        if self._idf is None:
            raise RuntimeError("fit() must run before dim is known")
        return len(self._vocab)

    # ------------------------------------------------------------------ #

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        counts = Counter(tokenize(text))
        for term, count in counts.items():
            col = self._vocab.get(term)
            if col is None:
                continue  # out-of-vocabulary query terms contribute nothing
            sublinear_tf = 1.0 + math.log(count)
            vec[col] = sublinear_tf * self._idf[col]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
