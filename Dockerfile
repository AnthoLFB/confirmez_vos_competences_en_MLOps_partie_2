# ---------- Étape de build ----------
FROM python:3.13-slim AS builder

WORKDIR /app

# Installer uv pour résoudre les dépendances
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ---------- Étape finale ----------
FROM python:3.13-slim

WORKDIR /app

# Installer les dépendances système nécessaires (libgomp1 pour LightGBM)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier l'environnement virtuel créé par uv
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copier le code source et le modèle
COPY app/ app/
COPY models/ models/

# Variables d'environnement par défaut
ENV MODEL_PATH=models/model.joblib
ENV API_KEY=change-me

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
