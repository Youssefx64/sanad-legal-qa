FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml backend/README.md /app/backend/
COPY backend/src /app/backend/src
COPY config /app/config
COPY data /app/data

RUN pip install --no-cache-dir -e "/app/backend[dev]"

WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "sanad.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
