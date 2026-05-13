FROM python:3.11-slim AS build
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e ".[dev]"

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=build /usr/local /usr/local
COPY --from=build /app /app
CMD ["python", "-m", "transformer_from_scratch.train", "--help"]
