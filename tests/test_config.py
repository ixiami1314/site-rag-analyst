"""Tests for Settings: demo-mode auto-detection and provider resolution."""

from __future__ import annotations

from site_rag_analyst.config import Settings


class TestDemoMode:
    def test_no_credentials_means_demo(self, default_settings: Settings) -> None:
        assert default_settings.demo_mode is True

    def test_credentials_mean_live(self, live_settings: Settings) -> None:
        assert live_settings.demo_mode is False

    def test_key_without_base_url_is_still_demo(self) -> None:
        s = Settings(_env_file=None, llm_api_key="sk-x", llm_base_url="")
        assert s.demo_mode is True

    def test_whitespace_credentials_mean_demo(self) -> None:
        s = Settings(_env_file=None, llm_api_key="   ", llm_base_url="  ")
        assert s.demo_mode is True


class TestEmbeddingProvider:
    def test_auto_resolves_to_tfidf_offline(self, default_settings: Settings) -> None:
        assert default_settings.effective_embedding_provider == "tfidf"

    def test_auto_resolves_to_openai_with_credentials(
        self, live_settings: Settings
    ) -> None:
        assert live_settings.effective_embedding_provider == "openai"

    def test_explicit_provider_wins_over_auto_detection(self) -> None:
        s = Settings(
            _env_file=None,
            embedding_provider="tfidf",
            llm_base_url="https://gw.example.com/v1",
            llm_api_key="sk-test",
        )
        assert s.effective_embedding_provider == "tfidf"


class TestDefaults:
    def test_sane_crawl_defaults(self, default_settings: Settings) -> None:
        assert default_settings.crawl_max_pages >= 1
        assert default_settings.crawl_delay_seconds >= 0
        assert default_settings.respect_robots is True

    def test_chunking_bounds_are_coherent(self, default_settings: Settings) -> None:
        # overlap must be smaller than the target to make progress
        assert default_settings.chunk_overlap_words < default_settings.chunk_target_words

    def test_user_agent_identifies_itself(self, default_settings: Settings) -> None:
        assert default_settings.user_agent.startswith("site-rag-analyst/")
