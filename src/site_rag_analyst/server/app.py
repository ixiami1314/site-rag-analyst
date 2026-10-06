"""FastAPI application.

Endpoints (all JSON unless noted):

- ``GET  /api/config``          — mode badge info (demo vs live, model, version)
- ``POST /api/runs {url}``      — start a run (empty url = bundled demo site);
                                  returns ``{"run_id": ...}``
- ``GET  /api/runs/{id}``       — status + stage progress + full result when done
- ``GET  /api/runs/{id}/report.md`` — the Markdown report (text/markdown)

Security posture for public deployments: submitted URLs must be public
HTTP(S) hosts — private, loopback, link-local and reserved ranges are
rejected before any request is made (SSRF guard), and runs are bounded by a
small concurrency semaphore. The bundled demo site is always allowed.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from site_rag_analyst import __version__
from site_rag_analyst.config import Settings, get_settings
from site_rag_analyst.crawler.fetcher import DEMO_ORIGIN
from site_rag_analyst.models import PipelineResult, StageStat
from site_rag_analyst.pipeline import Pipeline
from site_rag_analyst.reporting.report import render_markdown

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"
_MAX_URL_LENGTH = 2048
_MAX_TRACKED_RUNS = 30


def _is_public_hostname(hostname: str) -> bool:
    """True when the host resolves only to public, unicast addresses."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return False
        if not ip.is_global:
            return False
    return True


def _validate_target(url: str) -> str:
    """Normalize and validate a user-submitted target URL (SSRF guard)."""
    url = url.strip()
    if not url:
        return DEMO_ORIGIN + "/"
    if len(url) > _MAX_URL_LENGTH:
        raise HTTPException(status_code=400, detail="URL too long")
    if url.rstrip("/").startswith(DEMO_ORIGIN):
        return url
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname or ""
    if not host:
        raise HTTPException(status_code=400, detail="URL has no host")
    if not _is_public_hostname(host):
        raise HTTPException(
            status_code=400,
            detail="Only public hosts are accepted (private/loopback addresses are blocked).",
        )
    return url


class RunRecord(BaseModel):
    id: str
    url: str
    status: str = "queued"  # queued | running | done | error
    stages: list[StageStat] = []
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    result: PipelineResult | None = None


class RunRequest(BaseModel):
    url: str = ""


class RunRegistry:
    """In-memory run tracking (single-process; uvicorn workers=1)."""

    def __init__(self, max_concurrent: int = 2) -> None:
        self._runs: OrderedDict[str, RunRecord] = OrderedDict()
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def get(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    def latest(self) -> RunRecord | None:
        return next(reversed(self._runs.values())) if self._runs else None

    def register(self, url: str) -> RunRecord:
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        record = RunRecord(id=run_id, url=url)
        self._runs[run_id] = record
        while len(self._runs) > _MAX_TRACKED_RUNS:
            self._runs.popitem(last=False)
        return record

    async def execute(self, record: RunRecord, settings: Settings) -> None:
        async with self._semaphore:
            record.status = "running"
            loop = asyncio.get_running_loop()

            def on_stage(stat: StageStat) -> None:
                # StageStat is immutable-ish; copy to detach from pipeline state
                record.stages.append(stat.model_copy())

            pipeline = Pipeline(settings)
            try:
                result = await loop.run_in_executor(
                    None, lambda: pipeline.run(record.url, run_id=record.id, on_stage=on_stage)
                )
            except Exception as exc:  # noqa: BLE001 - surfaced as an error run
                logger.exception("run %s crashed", record.id)
                record.status = "error"
                record.result = PipelineResult(
                    run_id=record.id,
                    url=record.url,
                    demo=settings.demo_mode,
                    mode="demo" if settings.demo_mode else "live",
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            record.result = result
            record.status = "done" if result.ok else "error"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    registry = RunRegistry()
    app = FastAPI(title="site-rag-analyst", version=__version__)

    @app.on_event("startup")
    def _prepare() -> None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {
            "version": __version__,
            "demo_mode": settings.demo_mode,
            "model": "" if settings.demo_mode else settings.llm_model,
            "embedding_provider": settings.effective_embedding_provider,
            "llm_configured": not settings.demo_mode,
        }

    @app.post("/api/runs")
    async def start_run(request: RunRequest) -> dict[str, str]:
        target = _validate_target(request.url)
        record = registry.register(target)
        asyncio.create_task(registry.execute(record, settings))
        return {"run_id": record.id}

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str) -> dict[str, Any]:
        record = registry.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        payload: dict[str, Any] = {
            "id": record.id,
            "url": record.url,
            "status": record.status,
            "stages": [stage.model_dump(mode="json") for stage in record.stages],
        }
        if record.result is not None:
            payload["result"] = record.result.model_dump(mode="json")
        return payload

    @app.get("/api/runs/{run_id}/report.md", response_class=PlainTextResponse)
    def run_report_md(run_id: str) -> str:
        record = registry.get(run_id)
        if record is None or record.result is None:
            raise HTTPException(status_code=404, detail="run not found or not finished")
        return render_markdown(record.result)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
    return app


app = create_app()
