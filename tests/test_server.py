"""API tests with FastAPI's TestClient — offline (demo site corpus).

The SSRF guard is tested against real reserved ranges (localhost, RFC1918,
link-local) which resolve without network access.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from site_rag_analyst.config import Settings
from site_rag_analyst.server.app import _validate_target, create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="",
        llm_api_key="",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        crawl_delay_seconds=0.0,
    )


@pytest.fixture
def client(settings) -> TestClient:
    # Context manager keeps one event loop alive, so background run tasks
    # progress between polling requests.
    with TestClient(create_app(settings)) as test_client:
        yield test_client


class TestConfigEndpoint:
    def test_config_reports_demo_mode(self, client: TestClient) -> None:
        payload = client.get("/api/config").json()
        assert payload["demo_mode"] is True
        assert payload["embedding_provider"] == "tfidf"
        assert payload["version"]


class TestSSRFGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:8000/x",
            "http://127.0.0.1/",
            "https://10.1.2.3/secret",
            "https://192.168.0.9/admin",
            "http://169.254.169.254/latest/meta-data",
            "https://127.0.0.1.nip.io/",
        ],
    )
    def test_private_targets_rejected(self, url: str) -> None:
        with pytest.raises(Exception) as excinfo:
            _validate_target(url)
        assert "public" in str(excinfo.value).lower()

    def test_file_scheme_rejected(self) -> None:
        with pytest.raises(Exception, match="http"):
            _validate_target("file:///etc/passwd")

    def test_empty_url_means_demo_site(self) -> None:
        assert _validate_target("").endswith("demo.meridian.test/")

    def test_demo_origin_always_allowed(self) -> None:
        assert "demo.meridian.test" in _validate_target("http://demo.meridian.test/")

    def test_garbage_host_rejected(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException, match="public"):
            _validate_target("https://this-host-does-not-exist-qqq123/")

    def test_api_rejects_private_url(self, client: TestClient) -> None:
        response = client.post("/api/runs", json={"url": "http://169.254.169.254/"})
        assert response.status_code == 400
        assert "public" in response.json()["detail"].lower()


class TestRunLifecycle:
    def test_demo_run_end_to_end(self, client: TestClient) -> None:
        started = client.post("/api/runs", json={"url": ""})
        assert started.status_code == 200
        run_id = started.json()["run_id"]

        payload = None
        for _ in range(120):  # ~30s budget; demo run takes ~2-4s
            payload = client.get(f"/api/runs/{run_id}").json()
            if payload["status"] in ("done", "error"):
                break
            time.sleep(0.25)

        assert payload["status"] == "done", payload
        assert len(payload["stages"]) == 7
        result = payload["result"]
        assert result["pages"] and result["chunks"]
        assert result["report"]["sections"]

    def test_report_md_served(self, client: TestClient) -> None:
        run_id = client.post("/api/runs", json={"url": ""}).json()["run_id"]
        for _ in range(120):
            if client.get(f"/api/runs/{run_id}").json()["status"] in ("done", "error"):
                break
            time.sleep(0.25)
        response = client.get(f"/api/runs/{run_id}/report.md")
        assert response.status_code == 200
        assert "Executive summary" in response.text
        assert "[p" in response.text  # citations present

    def test_unknown_run_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/nope").status_code == 404

    def test_index_served(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "site-rag-analyst" in response.text
