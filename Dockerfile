FROM node:22-bookworm-slim AS frontend
WORKDIR /app/dashboard
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group dashboard
COPY catan_rl/ ./catan_rl/
COPY examples/ ./examples/
COPY --from=frontend /app/dashboard/dist ./dashboard/dist
RUN useradd --create-home app && mkdir runs && chown app:app runs
USER app
EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "catan_rl.server", "--host", "0.0.0.0"]
