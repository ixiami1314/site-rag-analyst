"""Tests for both vector-store backends (sqlite-vec and numpy cosine)."""

from __future__ import annotations

import numpy as np
import pytest

from site_rag_analyst.models import Chunk
from site_rag_analyst.store.vecstore import (
    NumpyVectorStore,
    SqliteVecStore,
    _sqlite_vec_available,
    open_store,
)


def chunk(cid: str, text: str) -> Chunk:
    n = int(cid.split("-c")[1])
    return Chunk(
        id=cid,
        page_url=f"https://example.com/page{n}",
        page_title=f"Page {n}",
        heading_path="Docs",
        index=n,
        text=text,
        word_count=len(text.split()),
    )


AXIS_CORPUS = [
    (chunk("p00-c0", "alpha alpha alpha about workflow builders"), [1.0, 0.0, 0.0]),
    (chunk("p01-c0", "beta beta beta about pricing plans"), [0.0, 1.0, 0.0]),
    (chunk("p02-c0", "gamma gamma gamma about support contacts"), [0.0, 0.0, 1.0]),
]
DIM = 3


def assert_search_order(store) -> None:
    store.add([c for c, _ in AXIS_CORPUS], [np.array(v) for _, v in AXIS_CORPUS])
    results = store.search(np.array([0.9, 0.1, 0.0]), k=2)
    assert [chunk.id for chunk, _ in results] == ["p00-c0", "p01-c0"]
    assert results[0][1] > results[1][1] > 0.0


class TestNumpyStore:
    def test_roundtrip_and_ranking(self, tmp_path) -> None:
        store = NumpyVectorStore(tmp_path / "rag.sqlite3")
        assert_search_order(store)
        store.close()

    def test_k_respected_and_empty_store(self, tmp_path) -> None:
        store = NumpyVectorStore(tmp_path / "rag.sqlite3")
        assert store.search(np.array([1.0, 0.0, 0.0]), k=3) == []
        store.add([c for c, _ in AXIS_CORPUS], [np.array(v) for _, v in AXIS_CORPUS])
        # k is an upper bound: zero-similarity chunks are not returned
        assert len(store.search(np.array([0.9, 0.1, 0.0]), k=2)) == 2
        assert len(store.search(np.array([1.0, 0.0, 0.0]), k=3)) == 1
        store.close()

    def test_persists_across_reopen(self, tmp_path) -> None:
        path = tmp_path / "rag.sqlite3"
        store = NumpyVectorStore(path)
        store.add([c for c, _ in AXIS_CORPUS], [np.array(v) for _, v in AXIS_CORPUS])
        store.close()
        reopened = NumpyVectorStore(path)
        results = reopened.search(np.array([0.0, 0.0, 1.0]), k=1)
        assert results[0][0].id == "p02-c0"
        assert results[0][0].text == "gamma gamma gamma about support contacts"
        reopened.close()

    def test_chunk_metadata_roundtrip(self, tmp_path) -> None:
        store = NumpyVectorStore(tmp_path / "rag.sqlite3")
        store.add([AXIS_CORPUS[0][0]], [np.array(AXIS_CORPUS[0][1])])
        chunk_found, _score = store.search(np.array([1.0, 0.0, 0.0]), k=1)[0]
        assert chunk_found.page_title == "Page 0"
        assert chunk_found.heading_path == "Docs"
        assert chunk_found.word_count == AXIS_CORPUS[0][0].word_count
        store.close()

    def test_mismatched_add_rejected(self, tmp_path) -> None:
        store = NumpyVectorStore(tmp_path / "rag.sqlite3")
        with pytest.raises(ValueError, match="parallel"):
            store.add([AXIS_CORPUS[0][0]], [])
        store.close()


@pytest.mark.skipif(not _sqlite_vec_available(), reason="sqlite-vec not installed")
class TestSqliteVecStore:
    def test_roundtrip_and_ranking(self, tmp_path) -> None:
        store = SqliteVecStore(tmp_path / "rag.sqlite3", dim=DIM)
        assert "sqlite-vec" in store.backend_name()
        assert_search_order(store)
        store.close()

    def test_persists_across_reopen(self, tmp_path) -> None:
        path = tmp_path / "rag.sqlite3"
        store = SqliteVecStore(path, dim=DIM)
        store.add([c for c, _ in AXIS_CORPUS], [np.array(v) for _, v in AXIS_CORPUS])
        store.close()
        reopened = SqliteVecStore(path, dim=DIM)
        results = reopened.search(np.array([0.0, 1.0, 0.0]), k=1)
        assert results[0][0].id == "p01-c0"
        reopened.close()

    def test_wrong_query_dim_rejected(self, tmp_path) -> None:
        store = SqliteVecStore(tmp_path / "rag.sqlite3", dim=DIM)
        with pytest.raises(ValueError, match="does not match"):
            store.search(np.array([1.0, 0.0]), k=1)
        store.close()


class TestOpenStoreFactory:
    def test_fixed_dim_prefers_sqlite_vec(self, tmp_path) -> None:
        store = open_store(tmp_path / "rag.sqlite3", dim=DIM)
        expected = "sqlite-vec" if _sqlite_vec_available() else "numpy-cosine"
        assert store.backend_name().startswith(expected)
        store.close()

    def test_variable_dim_gets_numpy_store(self, tmp_path) -> None:
        store = open_store(tmp_path / "rag.sqlite3", dim=None)
        assert store.backend_name() == "numpy-cosine"
        store.close()
