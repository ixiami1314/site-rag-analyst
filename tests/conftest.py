"""Shared pytest fixtures.

Tests construct Settings explicitly (never reading ambient env or a stray
.env file), so behavior is deterministic on any machine.
"""

from __future__ import annotations

import pytest

from site_rag_analyst.config import Settings


@pytest.fixture
def default_settings() -> Settings:
    """Offline/demo-mode settings with no credentials."""
    return Settings(
        _env_file=None,  # ignore any .env next to the working directory
        llm_base_url="",
        llm_api_key="",
        embedding_api_key="",
    )


@pytest.fixture
def live_settings() -> Settings:
    """Settings that look like a configured live deployment."""
    return Settings(
        _env_file=None,
        llm_base_url="https://gateway.example.com/v1",
        llm_api_key="sk-test",
    )
