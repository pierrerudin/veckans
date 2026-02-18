# Use an official Python slim image (Debian-based)
FROM python:3.12-slim AS veckans
# Install OS packages if you need git, ffmpeg, etc. Uncomment if needed:
# install system build-deps for packages with C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libopenblas-dev \
      liblapack-dev \
      curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Poetry ----
ARG POETRY_VERSION=2.1.4
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# No nested venvs; use system site-packages
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# First stage for dependency caching
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install dependencies ONLY (skip installing the project)
# Use POETRY_WITH_DEV=true at build time if you want dev deps in this image
ARG POETRY_WITH_DEV=false
RUN if [ "$POETRY_WITH_DEV" = "true" ]; then \
      poetry install --with dev --no-root --no-interaction --no-ansi ; \
    else \
      poetry install --only main --no-root --no-interaction --no-ansi ; \
    fi

# Copy runtime files
COPY README.md ./README.md
COPY ./src ./src

# Make src the working directory so `python main.py` works
WORKDIR /app/src

# (Optional) also expose src for imports even if cwd changes
ENV PYTHONPATH=/app/src

# Non-root user
RUN useradd -m appuser
USER appuser

ENTRYPOINT [ "/bin/bash" ]