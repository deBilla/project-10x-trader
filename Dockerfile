# Agent daemon image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Config is mounted/added at runtime; copy defaults into the image too.
COPY config ./config

# Default config dir inside the container.
ENV TRADER_CONFIG_DIR=/app/config

ENTRYPOINT ["trader"]
