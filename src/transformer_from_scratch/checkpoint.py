"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {"model": model.state_dict(), "step": step}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(payload["scheduler"])
    return payload
