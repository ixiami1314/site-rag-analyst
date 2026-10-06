"""Minimal OpenAI-compatible chat-completions client.

The pipeline speaks the de-facto standard ``POST {base}/chat/completions``
schema, so it works with OpenAI, OpenRouter, LiteLLM, Ollama's compat layer,
or any gateway that fronts Claude models with an OpenAI-compatible API —
which is how AnythingLLM-style deployments usually expose them. Endpoint and
model are pure configuration (``LLM_BASE_URL`` / ``LLM_MODEL``); the default
model name is Claude, per the deployment this pipeline was built for.
"""

from __future__ import annotations

import json
import logging
import re
import time

import httpx

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class ChatClient:
    """One method, one job: ``complete(system, user) -> text``."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        timeout_seconds: float = 120.0,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("ChatClient requires base_url and api_key")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_seconds),
        )

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        last_error: Exception | None = None
        for attempt in (1, 2, 3):
            try:
                response = self._client.post(
                    f"{self._base_url}/chat/completions", json=payload
                )
                if response.status_code == 200:
                    return _first_content(response.json())
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(
                        f"chat endpoint returned {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                last_error = RuntimeError(
                    f"chat endpoint returned {response.status_code}"
                )
            except httpx.HTTPError as exc:
                last_error = exc
            if attempt < 3:
                delay = 2.0 * attempt
                logger.warning(
                    "chat completion failed (%s), retrying in %ss", last_error, delay
                )
                time.sleep(delay)
        raise RuntimeError(f"chat completion failed after retries: {last_error}")


def _first_content(payload: dict) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("chat response contained no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("chat response contained no content")
    return content


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from an LLM reply.

    Tolerates the two classic failure modes: markdown code fences and
    stray prose around the object. Raises ValueError when nothing parses.
    """
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in LLM reply")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM reply is not valid JSON: {exc}") from exc
