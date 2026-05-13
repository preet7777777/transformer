"""FastAPI inference server.

Endpoints
---------
GET  /health           liveness + readiness
GET  /models           list registered versions
POST /generate         autoregressive token generation
POST /score            per-token log-probabilities
GET  /metrics          Prometheus-style plain-text counters
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, AsyncGenerator

import torch

from .config import ModelConfig
from .generation import generate_tokens
from .model import TransformerLM
from .registry import CheckpointRegistry, ModelVersion
from .utils import resolve_device

try:
    import fastapi
    from fastapi import Depends, FastAPI, HTTPException, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field, field_validator
    import uvicorn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "API dependencies missing. Install with: pip install 'transformer-from-scratch[api]'"
    ) from exc

log = logging.getLogger("tfs.api")

# ---------------------------------------------------------------------------
# Global app state
# ---------------------------------------------------------------------------

_REGISTRY: CheckpointRegistry | None = None
_LOADED: dict[str, TransformerLM] = {}   # version name → model
_DEVICE: torch.device = resolve_device()

# ---------------------------------------------------------------------------
# Metrics (simple in-process counters; no external dependency)
# ---------------------------------------------------------------------------

@dataclass
class _Metrics:
    requests_total: int = 0
    requests_ok: int = 0
    requests_err: int = 0
    tokens_generated: int = 0
    # rolling latency window (last 1000 requests)
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, latency_ms: float, ok: bool, tokens: int = 0) -> None:
        self.requests_total += 1
        if ok:
            self.requests_ok += 1
        else:
            self.requests_err += 1
        self.tokens_generated += tokens
        self.latencies_ms.append(latency_ms)

    def p50(self) -> float:
        data = sorted(self.latencies_ms)
        if not data:
            return 0.0
        return data[len(data) // 2]

    def p95(self) -> float:
        data = sorted(self.latencies_ms)
        if not data:
            return 0.0
        return data[int(len(data) * 0.95)]

    def error_rate(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self.requests_err / self.requests_total


METRICS = _Metrics()

# ---------------------------------------------------------------------------
# Rate limiter (token-bucket, no external dep)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Per-IP sliding-window rate limiter."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._windows: dict[str, deque] = {}

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        if key not in self._windows:
            self._windows[key] = deque()
        q = self._windows[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.max_requests:
            return False
        q.append(now)
        return True


_LIMITER = _RateLimiter(max_requests=60, window_seconds=60.0)

# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _get_registry() -> CheckpointRegistry:
    if _REGISTRY is None:
        raise HTTPException(status_code=503, detail="Registry not initialised")
    return _REGISTRY


def _rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if not _LIMITER.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Max 60 requests/minute.",
        )


def _load_model(version_name: str) -> TransformerLM:
    """Return a cached model, loading it on first access."""
    if version_name not in _LOADED:
        if _REGISTRY is None:
            raise HTTPException(status_code=503, detail="Registry not initialised")
        try:
            model, _ = _REGISTRY.load_model(version_name, device=str(_DEVICE))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc
        _LOADED[version_name] = model
    return _LOADED[version_name]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

MAX_TOKENS_LIMIT = int(os.getenv("TFS_MAX_TOKENS", "512"))
MAX_PROMPT_LEN = int(os.getenv("TFS_MAX_PROMPT_LEN", "256"))
REQUEST_TIMEOUT = float(os.getenv("TFS_REQUEST_TIMEOUT", "30.0"))


class GenerateRequest(BaseModel):
    model_version: str = Field(..., description="Registered version name")
    prompt: list[int] = Field(..., description="List of token IDs (prompt)")
    max_new_tokens: int = Field(16, ge=1, le=MAX_TOKENS_LIMIT)
    temperature: float = Field(1.0, gt=0.0, le=10.0)
    top_k: int | None = Field(None, ge=1)

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("prompt must not be empty")
        if len(v) > MAX_PROMPT_LEN:
            raise ValueError(f"prompt exceeds max length {MAX_PROMPT_LEN}")
        return v


class GenerateResponse(BaseModel):
    model_version: str
    prompt: list[int]
    generated: list[int]
    full_sequence: list[int]
    tokens_generated: int
    latency_ms: float


class ScoreRequest(BaseModel):
    model_version: str = Field(..., description="Registered version name")
    tokens: list[int] = Field(..., description="Full token sequence to score")

    @field_validator("tokens")
    @classmethod
    def tokens_not_empty(cls, v: list[int]) -> list[int]:
        if len(v) < 2:
            raise ValueError("tokens must have at least 2 elements (prompt + 1 target)")
        if len(v) > MAX_PROMPT_LEN + MAX_TOKENS_LIMIT:
            raise ValueError("tokens sequence too long")
        return v


class ScoreResponse(BaseModel):
    model_version: str
    log_probs: list[float]
    mean_log_prob: float
    perplexity: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    device: str
    loaded_models: list[str]
    registry_versions: list[str]
    uptime_s: float


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

_START_TIME = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    log.info("TFS API starting up on device=%s", _DEVICE)
    yield
    log.info("TFS API shutting down")
    _LOADED.clear()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Transformer From Scratch — Inference API",
    description="REST API for the transformer-from-scratch language model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — request timing & metrics
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    t0 = time.monotonic()
    try:
        response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT)
        ok = response.status_code < 500
        latency = (time.monotonic() - t0) * 1000
        METRICS.record(latency, ok=ok)
        response.headers["X-Latency-Ms"] = f"{latency:.2f}"
        return response
    except asyncio.TimeoutError:
        latency = (time.monotonic() - t0) * 1000
        METRICS.record(latency, ok=False)
        return JSONResponse(
            {"detail": f"Request timed out after {REQUEST_TIMEOUT}s"},
            status_code=504,
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(registry: CheckpointRegistry = Depends(_get_registry)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(_DEVICE),
        loaded_models=list(_LOADED.keys()),
        registry_versions=registry.list_names(),
        uptime_s=round(time.monotonic() - _START_TIME, 1),
    )


@app.get("/models", tags=["ops"])
def list_models(registry: CheckpointRegistry = Depends(_get_registry)) -> list[dict[str, Any]]:
    return [v.to_dict() for v in registry.list_versions()]


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate(
    req: GenerateRequest,
    _rl: None = Depends(_rate_limit),
    registry: CheckpointRegistry = Depends(_get_registry),
) -> GenerateResponse:
    model = _load_model(req.model_version)

    prompt_tensor = torch.tensor([req.prompt], dtype=torch.long, device=_DEVICE)
    t0 = time.perf_counter()
    try:
        output = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: generate_tokens(
                model,
                prompt_tensor,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    full = output[0].tolist()
    generated = full[len(req.prompt):]
    METRICS.tokens_generated += len(generated)

    return GenerateResponse(
        model_version=req.model_version,
        prompt=req.prompt,
        generated=generated,
        full_sequence=full,
        tokens_generated=len(generated),
        latency_ms=round(latency_ms, 2),
    )


@app.post("/score", response_model=ScoreResponse, tags=["inference"])
async def score(
    req: ScoreRequest,
    _rl: None = Depends(_rate_limit),
    registry: CheckpointRegistry = Depends(_get_registry),
) -> ScoreResponse:
    model = _load_model(req.model_version)

    tokens_tensor = torch.tensor([req.tokens], dtype=torch.long, device=_DEVICE)
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            logits = await asyncio.get_event_loop().run_in_executor(
                None, lambda: model(tokens_tensor[:, :-1])
            )
        log_probs_tensor = torch.log_softmax(logits, dim=-1)
        targets = tokens_tensor[:, 1:]
        gathered = log_probs_tensor.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        log_probs = gathered[0].tolist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    mean_lp = sum(log_probs) / len(log_probs)
    import math
    perplexity = math.exp(-mean_lp)

    return ScoreResponse(
        model_version=req.model_version,
        log_probs=[round(lp, 6) for lp in log_probs],
        mean_log_prob=round(mean_lp, 6),
        perplexity=round(perplexity, 4),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics() -> str:
    import psutil
    try:
        mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        mem_mb = -1.0

    lines = [
        f"tfs_requests_total {METRICS.requests_total}",
        f"tfs_requests_ok {METRICS.requests_ok}",
        f"tfs_requests_err {METRICS.requests_err}",
        f"tfs_tokens_generated {METRICS.tokens_generated}",
        f"tfs_latency_p50_ms {METRICS.p50():.2f}",
        f"tfs_latency_p95_ms {METRICS.p95():.2f}",
        f"tfs_error_rate {METRICS.error_rate():.4f}",
        f"tfs_uptime_s {time.monotonic() - _START_TIME:.1f}",
        f"tfs_memory_rss_mb {mem_mb:.1f}",
        f"tfs_loaded_models {len(_LOADED)}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TFS Inference API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--registry-dir",
        default="./registry",
        help="Directory containing registry.json and checkpoints",
    )
    parser.add_argument(
        "--runs-dir",
        default="./runs",
        help="Scan this directory to auto-register checkpoints",
    )
    parser.add_argument("--device", default=None, help="cpu / cuda / mps")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--preload",
        nargs="*",
        default=[],
        help="Version names to load into memory at startup",
    )
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn hot-reload (dev)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    global _REGISTRY, _DEVICE
    _DEVICE = resolve_device(args.device)
    _REGISTRY = CheckpointRegistry(args.registry_dir)

    # Auto-scan runs dir for new checkpoints
    new_versions = _REGISTRY.auto_scan(args.runs_dir)
    if new_versions:
        log.info("Auto-registered: %s", new_versions)

    # Eagerly load requested models
    for name in args.preload:
        try:
            _load_model(name)
            log.info("Preloaded model '%s'", name)
        except Exception as exc:
            log.warning("Could not preload '%s': %s", name, exc)

    uvicorn.run(
        "transformer_from_scratch.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
