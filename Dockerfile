# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Zainstaluj uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Skopiuj pliki projektu potrzebne do instalacji zależności
COPY pyproject.toml uv.lock ./

# Zainstaluj zależności do dedykowanego venv (bez dev deps, bez editable)
RUN uv sync --frozen --no-dev --no-install-project

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Biblioteki systemowe wymagane przez OpenCV i rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Skopiuj venv z buildera
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Skopiuj kod źródłowy
COPY src/ ./src/
COPY app/ ./app/

# Model: montowany z GCS przez env lub lokalnie przy budowaniu obrazu.
# Ustaw MODEL_GCS_URI=gs://twoj-bucket/models/best_model.pt
# LUB skopiuj lokalny model:
# COPY models/best_model.pt ./models/best_model.pt

# Cloud Run wymaga portu 8080
ENV PORT=8080

# Użytkownik nieprivilegowany
RUN useradd -m appuser
USER appuser

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
