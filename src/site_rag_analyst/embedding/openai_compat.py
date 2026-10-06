"""Embeddings from any OpenAI-compatible ``/embeddings`` endpoint."""

from __future__ import annotations

import logging

import httpx
import numpy as np

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64


class OpenAICompatEmbeddings:
    """Thin client for ``POST {base_url}/embeddings``."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        if not base_url or not api_key:
            raise ValueError("openai embeddings require base_url and api_key")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60.0),
        )
        self._dim: int | None = None

    def fit(self, texts: list[str]) -> None:  # noqa: ARG002 - stateless provider
        return None

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("dim is known after the first embed call")
        return self._dim

    # ------------------------------------------------------------------ #

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        try:
            response = self._client.post(
                f"{self._base_url}/embeddings",
                json={"model": self._model, "input": texts},
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"embedding request failed: {exc}") from exc
        if response.status_code != 200:
            raise RuntimeError(
                f"embedding endpoint returned {response.status_code}: "
                f"{response.text[:300]}"
            )
        payload = response.json()
        items = sorted(payload["data"], key=lambda item: item["index"])
        vectors = [np.asarray(item["embedding"], dtype=np.float64) for item in items]
        self._dim = len(vectors[0])
        return vectors
