"""Vector stores backed by a single SQLite file.

Two interchangeable backends:

- :class:`SqliteVecStore` — a ``vec0`` virtual table from the `sqlite-vec
  <https://github.com/asg017/sqlite-vec>`_ extension (ANN-capable, fixed
  vector width). Used when embeddings have a known dimension, i.e. API
  embeddings.
- :class:`NumpyVectorStore` — embeddings as float32 blobs, exact cosine
  search in numpy. Used for TF-IDF vectors (dimension = vocabulary size,
  only known after fitting) and wherever the extension cannot load.

Exact brute-force search is optimal well past 100k chunks on a single core,
which comfortably covers website-sized corpora; ``vec0`` future-proofs the
same interface for much larger ingest jobs. Both stores persist to the same
portable single-file database — no server, no daemon, no migrations.

``open_store`` picks a backend: sqlite-vec when available and the dimension
is fixed, numpy otherwise — so the pipeline never fails because a wheel was
missing on some platform.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Protocol

import numpy as np

from site_rag_analyst.models import Chunk

_SCHEMA_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    rowid_opt INTEGER PRIMARY KEY AUTOINCREMENT,
    id        TEXT UNIQUE NOT NULL,
    page_url  TEXT NOT NULL,
    page_title TEXT NOT NULL DEFAULT '',
    heading_path TEXT NOT NULL DEFAULT '',
    idx       INTEGER NOT NULL DEFAULT 0,
    text      TEXT NOT NULL,
    word_count INTEGER NOT NULL DEFAULT 0
)
"""


class VectorStore(Protocol):
    """Chunk metadata + vector index over one SQLite file."""

    def add(self, chunks: list[Chunk], vectors: list[np.ndarray]) -> None: ...
    def search(
        self, query_vector: np.ndarray, k: int = 6
    ) -> list[tuple[Chunk, float]]: ...
    def close(self) -> None: ...
    def backend_name(self) -> str: ...


class _SqliteBase:
    """Shared chunk-metadata persistence."""

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_CHUNKS)

    # -- reads/writes shared by both backends -------------------------- #

    def _insert_chunk(self, rowid: int, chunk: Chunk) -> None:
        self._conn.execute(
            "INSERT INTO chunks(rowid_opt, id, page_url, page_title, heading_path,"
            " idx, text, word_count) VALUES (?,?,?,?,?,?,?,?)",
            (
                rowid,
                chunk.id,
                chunk.page_url,
                chunk.page_title,
                chunk.heading_path,
                chunk.index,
                chunk.text,
                chunk.word_count,
            ),
        )

    def _chunk_by_rowid(self, rowid: int) -> Chunk | None:
        row = self._conn.execute(
            "SELECT id, page_url, page_title, heading_path, idx, text, word_count"
            " FROM chunks WHERE rowid_opt = ?",
            (rowid,),
        ).fetchone()
        if row is None:
            return None
        return Chunk(
            id=row[0],
            page_url=row[1],
            page_title=row[2],
            heading_path=row[3],
            index=row[4],
            text=row[5],
            word_count=row[6],
        )

    def _all_rowids(self) -> list[int]:
        return [
            r[0]
            for r in self._conn.execute("SELECT rowid_opt FROM chunks").fetchall()
        ]

    def close(self) -> None:
        self._conn.close()

    # -- to be provided by vector backends ------------------------------ #

    def add(self, chunks: list[Chunk], vectors: list[np.ndarray]) -> None: ...
    def search(self, query_vector: np.ndarray, k: int = 6) -> list[tuple[Chunk, float]]: ...
    def backend_name(self) -> str: ...


def serialize_f32(vector: np.ndarray) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector.astype(np.float32).tolist())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SqliteVecStore(_SqliteBase):
    """ANN-capable store via the sqlite-vec ``vec0`` virtual table."""

    def __init__(self, path: str | Path, dim: int) -> None:
        super().__init__(path)
        import sqlite_vec  # imported late so numpy-only environments still work

        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._dim = dim
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[{dim}] distance_metric=cosine
            )
            """
        )

    def backend_name(self) -> str:
        return f"sqlite-vec (dim={self._dim})"

    def add(self, chunks: list[Chunk], vectors: list[np.ndarray]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be parallel lists")
        with self._conn:
            for chunk, vector in zip(chunks, vectors, strict=True):
                cursor = self._conn.execute(
                    "INSERT INTO chunks(id, page_url, page_title, heading_path,"
                    " idx, text, word_count) VALUES (?,?,?,?,?,?,?)",
                    (
                        chunk.id,
                        chunk.page_url,
                        chunk.page_title,
                        chunk.heading_path,
                        chunk.index,
                        chunk.text,
                        chunk.word_count,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (cursor.lastrowid, serialize_f32(vector)),
                )

    def search(self, query_vector: np.ndarray, k: int = 6) -> list[tuple[Chunk, float]]:
        if len(query_vector) != self._dim:
            raise ValueError(
                f"query dim {len(query_vector)} does not match index dim {self._dim}"
            )
        rows = self._conn.execute(
            """
            SELECT v.rowid, v.distance
            FROM vec_chunks v
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (serialize_f32(query_vector), k),
        ).fetchall()
        results: list[tuple[Chunk, float]] = []
        for rowid, distance in rows:
            chunk = self._chunk_by_rowid(rowid)
            if chunk is not None:
                results.append((chunk, 1.0 - float(distance)))
        return results


class NumpyVectorStore(_SqliteBase):
    """Exact cosine search over float32 blobs — works for any dimension."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vec_chunks (
                rowid_opt INTEGER PRIMARY KEY,
                dim       INTEGER NOT NULL,
                vec       BLOB NOT NULL
            )
            """
        )
        self._matrix: np.ndarray | None = None
        self._rowids: list[int] | None = None

    def backend_name(self) -> str:
        return "numpy-cosine"

    def add(self, chunks: list[Chunk], vectors: list[np.ndarray]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be parallel lists")
        with self._conn:
            for chunk, vector in zip(chunks, vectors, strict=True):
                cursor = self._conn.execute(
                    "INSERT INTO chunks(id, page_url, page_title, heading_path,"
                    " idx, text, word_count) VALUES (?,?,?,?,?,?,?)",
                    (
                        chunk.id,
                        chunk.page_url,
                        chunk.page_title,
                        chunk.heading_path,
                        chunk.index,
                        chunk.text,
                        chunk.word_count,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO vec_chunks(rowid_opt, dim, vec) VALUES (?,?,?)",
                    (cursor.lastrowid, len(vector), serialize_f32(vector)),
                )
        self._matrix = None  # invalidate cached search matrix
        self._rowids = None

    def search(self, query_vector: np.ndarray, k: int = 6) -> list[tuple[Chunk, float]]:
        matrix, rowids = self._load_matrix()
        if matrix is None or matrix.size == 0:
            return []
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
        norms[norms == 0] = 1.0
        scores = (matrix @ query_vector) / norms
        order = np.argsort(-scores)[:k]
        results: list[tuple[Chunk, float]] = []
        for position in order:
            if scores[position] <= 0:
                continue
            chunk = self._chunk_by_rowid(int(rowids[position]))
            if chunk is not None:
                results.append((chunk, float(scores[position])))
        return results

    def _load_matrix(self) -> tuple[np.ndarray | None, list[int]]:
        if self._matrix is not None:
            return self._matrix, self._rowids or []
        rows = self._conn.execute(
            "SELECT rowid_opt, dim, vec FROM vec_chunks"
        ).fetchall()
        if not rows:
            return None, []
        dim = rows[0][1]
        matrix = np.array(
            [struct.unpack(f"{dim}f", row[2]) for row in rows], dtype=np.float64
        )
        rowids = [row[0] for row in rows]
        self._matrix = matrix
        self._rowids = rowids
        return matrix, rowids


def _sqlite_vec_available() -> bool:
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        return False
    return True


def open_store(path: str | Path, dim: int | None) -> VectorStore:
    """Pick a backend: sqlite-vec for fixed-dim vectors, numpy otherwise.

    If the extension is present but fails to load (unsupported platform),
    fall back to the numpy store so ingest never hard-fails.
    """
    if dim is not None and _sqlite_vec_available():
        try:
            return SqliteVecStore(path, dim)
        except (sqlite3.OperationalError, AttributeError) as exc:
            import logging

            logging.getLogger(__name__).warning(
                "sqlite-vec unavailable (%s); using numpy cosine store", exc
            )
    return NumpyVectorStore(path)
