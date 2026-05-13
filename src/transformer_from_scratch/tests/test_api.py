"""Integration tests for the FastAPI inference server."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("fastapi", reason="fastapi not installed — skipping API tests")
pytest.importorskip("httpx", reason="httpx not installed — skipping API tests")

from fastapi.testclient import TestClient

from transformer_from_scratch.config import ModelConfig
from transformer_from_scratch.model import TransformerLM
from transformer_from_scratch.registry import CheckpointRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SMALL_CFG = ModelConfig(
    vocab_size=32,
    d_model=16,
    n_layers=1,
    n_heads=2,
    seq_len=16,
    d_ff=32,
    dropout=0.0,
)


@pytest.fixture()
def tmp_registry(tmp_path: Path) -> Generator[tuple[CheckpointRegistry, str], None, None]:
    """Create a registry with one registered model saved to a temp checkpoint."""
    ckpt_path = tmp_path / "model.pt"
    model = TransformerLM(_SMALL_CFG)
    torch.save({"model": model.state_dict(), "step": 0}, ckpt_path)
    registry = CheckpointRegistry(tmp_path / "registry")
    registry.register(
        name="test-v1",
        checkpoint_path=ckpt_path,
        config=_SMALL_CFG,
        description="unit test model",
    )
    yield registry, "test-v1"


@pytest.fixture()
def client(tmp_registry: tuple[CheckpointRegistry, str]) -> Generator[TestClient, None, None]:
    registry, _ = tmp_registry
    import transformer_from_scratch.api as api_module

    api_module._REGISTRY = registry
    api_module._LOADED.clear()
    api_module._DEVICE = torch.device("cpu")

    from transformer_from_scratch.api import app
    with TestClient(app) as c:
        yield c

    api_module._LOADED.clear()
    api_module._REGISTRY = None


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "test-v1" in data["registry_versions"]


# ---------------------------------------------------------------------------
# /models
# ---------------------------------------------------------------------------

def test_list_models(client: TestClient) -> None:
    resp = client.get("/models")
    assert resp.status_code == 200
    names = [v["name"] for v in resp.json()]
    assert "test-v1" in names


# ---------------------------------------------------------------------------
# /generate
# ---------------------------------------------------------------------------

def test_generate_basic(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        json={"model_version": "test-v1", "prompt": [0, 1, 2], "max_new_tokens": 4},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["tokens_generated"] == 4
    assert len(data["full_sequence"]) == 7  # 3 prompt + 4 new


def test_generate_returns_latency(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        json={"model_version": "test-v1", "prompt": [0], "max_new_tokens": 2},
    )
    assert resp.status_code == 200
    assert resp.json()["latency_ms"] >= 0


def test_generate_temperature(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        json={"model_version": "test-v1", "prompt": [0, 1], "max_new_tokens": 3, "temperature": 0.5},
    )
    assert resp.status_code == 200


def test_generate_top_k(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        json={"model_version": "test-v1", "prompt": [0, 1], "max_new_tokens": 3, "top_k": 5},
    )
    assert resp.status_code == 200


def test_generate_unknown_version(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        json={"model_version": "nonexistent", "prompt": [0], "max_new_tokens": 1},
    )
    assert resp.status_code == 404


def test_generate_empty_prompt(client: TestClient) -> None:
    resp = client.post(
        "/generate",
        json={"model_version": "test-v1", "prompt": [], "max_new_tokens": 4},
    )
    assert resp.status_code == 422


def test_generate_exceeds_max_tokens(client: TestClient) -> None:
    import transformer_from_scratch.api as api_module
    limit = api_module.MAX_TOKENS_LIMIT
    resp = client.post(
        "/generate",
        json={"model_version": "test-v1", "prompt": [0], "max_new_tokens": limit + 1},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /score
# ---------------------------------------------------------------------------

def test_score_basic(client: TestClient) -> None:
    resp = client.post(
        "/score",
        json={"model_version": "test-v1", "tokens": [0, 1, 2, 3]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["log_probs"]) == 3  # len(tokens) - 1
    assert data["perplexity"] > 0
    assert data["mean_log_prob"] < 0


def test_score_too_short(client: TestClient) -> None:
    resp = client.post(
        "/score",
        json={"model_version": "test-v1", "tokens": [0]},
    )
    assert resp.status_code == 422


def test_score_unknown_version(client: TestClient) -> None:
    resp = client.post(
        "/score",
        json={"model_version": "nope", "tokens": [0, 1, 2]},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

def test_metrics_text(client: TestClient) -> None:
    client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "tfs_requests_total" in resp.text
    assert "tfs_latency_p95_ms" in resp.text


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_register_and_get(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    model = TransformerLM(_SMALL_CFG)
    torch.save({"model": model.state_dict(), "step": 0}, ckpt)

    reg = CheckpointRegistry(tmp_path / "reg")
    reg.register("my-model", ckpt, _SMALL_CFG, description="test")
    v = reg.get("my-model")
    assert v.name == "my-model"
    assert v.description == "test"


def test_registry_duplicate_raises(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    model = TransformerLM(_SMALL_CFG)
    torch.save({"model": model.state_dict(), "step": 0}, ckpt)

    reg = CheckpointRegistry(tmp_path / "reg")
    reg.register("v1", ckpt, _SMALL_CFG)
    with pytest.raises(ValueError, match="already exists"):
        reg.register("v1", ckpt, _SMALL_CFG)


def test_registry_overwrite(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    model = TransformerLM(_SMALL_CFG)
    torch.save({"model": model.state_dict(), "step": 0}, ckpt)

    reg = CheckpointRegistry(tmp_path / "reg")
    reg.register("v1", ckpt, _SMALL_CFG, description="first")
    reg.register("v1", ckpt, _SMALL_CFG, description="second", overwrite=True)
    assert reg.get("v1").description == "second"


def test_registry_delete(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    model = TransformerLM(_SMALL_CFG)
    torch.save({"model": model.state_dict(), "step": 0}, ckpt)

    reg = CheckpointRegistry(tmp_path / "reg")
    reg.register("v1", ckpt, _SMALL_CFG)
    reg.delete("v1")
    assert "v1" not in reg.list_names()


def test_registry_load_model(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    model = TransformerLM(_SMALL_CFG)
    torch.save({"model": model.state_dict(), "step": 0}, ckpt)

    reg = CheckpointRegistry(tmp_path / "reg")
    reg.register("v1", ckpt, _SMALL_CFG)
    loaded, version = reg.load_model("v1", device="cpu")
    assert isinstance(loaded, TransformerLM)
    assert version.name == "v1"


def test_registry_persists_to_disk(tmp_path: Path) -> None:
    ckpt = tmp_path / "m.pt"
    model = TransformerLM(_SMALL_CFG)
    torch.save({"model": model.state_dict(), "step": 0}, ckpt)
    reg_dir = tmp_path / "reg"

    reg1 = CheckpointRegistry(reg_dir)
    reg1.register("v1", ckpt, _SMALL_CFG)

    reg2 = CheckpointRegistry(reg_dir)
    assert "v1" in reg2.list_names()
