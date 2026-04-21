# syntax=docker/dockerfile:1.7
# =============================================================================
# Machine Learning Model — multi-stage Dockerfile
#
# Stages
#   deps  – install Python dependencies (shared cache layer)
#   test  – run the full pytest suite (CI target)
#   app   – lean production image (CLI / batch ML)
# =============================================================================

# ── deps ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt -r requirements-dev.txt

# ── test ──────────────────────────────────────────────────────────────────────
FROM deps AS test

COPY . .
RUN pip install -e . --no-deps

ENV PYTHONPATH=/app/src

# Run tests; exit code propagated to docker build / CI
CMD ["python", "-m", "pytest", "tests/", \
     "--ignore=tests/gui", \
     "-q", "--no-cov", "--tb=short"]

# ── app ───────────────────────────────────────────────────────────────────────
FROM deps AS app

COPY . .
RUN pip install -e . --no-deps \
    && groupadd -r appuser \
    && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser
ENV PYTHONPATH=/app/src

CMD ["machine-learning-model"]
