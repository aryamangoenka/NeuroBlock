# NeuroBlock — single-container image: Flask serves the API, WebSockets,
# AND the built React app (same origin, no CORS, one URL).
#
# Build:  docker build -t neuroblock .
# Run:    docker run -p 8080:8080 -e SECRET_KEY=change-me neuroblock

# ---------- stage 1: build the frontend ----------
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
# No VITE_BACKEND_URL: the app defaults to same-origin in production
RUN npm run build

# ---------- stage 2: python runtime ----------
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    FLASK_CONFIG=production \
    FLASK_ENV=production \
    PYTHONPATH=.

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY --from=frontend /fe/dist ./frontend/dist

RUN mkdir -p exports datasets/sessions sessions

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Single eventlet worker: the app keeps training state in-process
CMD exec gunicorn --bind :$PORT --worker-class eventlet --workers 1 --timeout 0 backend.main:app
