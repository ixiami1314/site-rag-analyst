"""Typed application settings, loaded from the environment (or a ``.env`` file).

Everything an operator may want to tweak is environment-driven — no code
changes are needed to point the pipeline at a different LLM gateway, embedding
endpoint, or to run fully offline in demo mode.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline configuration.

    Demo mode is *auto-detected*: when no LLM credentials are configured the
    pipeline still runs end-to-end using offline TF-IDF embeddings and a
    deterministic, retrieval-grounded mock analyst. Add credentials and the
    same code path calls your real endpoint.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Analysis LLM (any OpenAI-compatible /chat/completions endpoint) ---
    llm_base_url: str = ""  # e.g. https://your-gateway.example.com/v1
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-5-5"  # any model name your endpoint serves
    llm_temperature: float = 0.2
    llm_timeout_seconds: float = 120.0

    # --- Embeddings ---
    embedding_provider: str = "auto"  # auto | tfidf | openai
    embedding_base_url: str = ""  # falls back to llm_base_url
    embedding_api_key: str = ""  # falls back to llm_api_key
    embedding_model: str = "text-embedding-3-small"

    # --- Crawling ---
    crawl_max_pages: int = Field(default=25, ge=1, le=500)
    crawl_max_depth: int = Field(default=2, ge=0, le=10)
    crawl_delay_seconds: float = Field(default=1.0, ge=0.0)
    respect_robots: bool = True
    user_agent: str = (
        "site-rag-analyst/0.1 (+https://github.com/ixiami1314/site-rag-analyst)"
    )

    # --- Chunking ---
    chunk_target_words: int = Field(default=280, ge=80, le=1200)
    chunk_overlap_words: int = Field(default=45, ge=0, le=400)

    # --- Retrieval ---
    retrieval_top_k: int = Field(default=6, ge=1, le=50)

    # --- Storage ---
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")

    # --- Service ---
    # FastAPI's /docs and /openapi.json are handy locally but are surface
    # area on a public deployment — disable them there with DOCS_ENABLED=false.
    docs_enabled: bool = True

    @property
    def demo_mode(self) -> bool:
        """True when no LLM endpoint/key is configured — run offline instead."""
        return not (self.llm_base_url.strip() and self.llm_api_key.strip())

    @property
    def effective_embedding_provider(self) -> str:
        """Resolve ``auto`` to a concrete provider.

        Real embeddings require an endpoint + key; anything less and we fall
        back to offline TF-IDF vectors so the pipeline always works.
        """
        if self.embedding_provider != "auto":
            return self.embedding_provider
        has_key = bool(self.embedding_api_key.strip() or self.llm_api_key.strip())
        has_url = bool(self.embedding_base_url.strip() or self.llm_base_url.strip())
        return "openai" if (has_key and has_url) else "tfidf"


@lru_cache
def get_settings() -> Settings:
    """Process-wide cached settings (env + optional .env file)."""
    return Settings()
