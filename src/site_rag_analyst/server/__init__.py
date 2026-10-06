"""FastAPI service: run the pipeline, stream progress, serve the web UI."""

from site_rag_analyst.server.app import create_app

__all__ = ["create_app"]
