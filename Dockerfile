FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    TZ=UTC

# Dependencies first so a source edit does not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY scripts ./scripts
RUN sh scripts/fetch-yamnet.sh && mv /app/models /models

COPY babymon ./babymon
RUN uv sync --frozen --no-dev

VOLUME /data
EXPOSE 8080
ENTRYPOINT ["babymon"]
CMD ["run"]
