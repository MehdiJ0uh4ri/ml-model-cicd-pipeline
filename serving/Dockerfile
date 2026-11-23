# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

# The image carries the *code*, never a model. Models arrive from the registry
# at startup, which is what makes a rollback an alias move instead of a
# rebuild-and-redeploy.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY params.yaml thresholds.yaml ./
COPY src ./src
COPY serving ./serving

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/artifacts /app/registry \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# readyz, not healthz: a process that is up but has no champion loaded must
# not receive traffic.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8080/readyz', timeout=2).status_code==200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
