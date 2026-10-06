# site-rag-analyst — single-image service.
# Runs in demo mode with zero configuration (no API keys needed);
# add LLM_* env vars to point the analysis layer at a real endpoint.
FROM python:3.12-slim

WORKDIR /app

# Install first for layer caching, then copy the package (wheel includes
# the bundled demo site + eval set from src/site_rag_analyst/).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

# Run as an unprivileged user; data/ and output/ are writable volumes.
RUN useradd --create-home appuser \
    && mkdir -p /app/data /app/output \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# workers=1 on purpose: the run registry is in-process by design.
CMD ["uvicorn", "site_rag_analyst.server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
