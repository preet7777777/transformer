"""Checkpoint registry — discovers and loads versioned model checkpoints."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint
from .config import ModelConfig
from .model import TransformerLM
from .utils import resolve_device


@dataclass
class ModelVersion:
    name: str
    path: str
    config: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    description: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        return cls(**data)


class CheckpointRegistry:
    """File-backed registry mapping version names to checkpoint paths + metadata."""

    INDEX_FILE = "registry.json"

    def __init__(self, registry_dir: str | Path) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / self.INDEX_FILE
        self._versions: dict[str, ModelVersion] = {}
        self._load_index()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        if self._index_path.exists():
            with self._index_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            self._versions = {k: ModelVersion.from_dict(v) for k, v in raw.items()}

    def _save_index(self) -> None:
        with self._index_path.open("w", encoding="utf-8") as f:
            json.dump({k: v.to_dict() for k, v in self._versions.items()}, f, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        checkpoint_path: str | Path,
        config: ModelConfig,
        description: str = "",
        metrics: dict[str, float] | None = None,
        overwrite: bool = False,
    ) -> ModelVersion:
        if name in self._versions and not overwrite:
            raise ValueError(f"Version '{name}' already exists. Pass overwrite=True to replace.")
        version = ModelVersion(
            name=name,
            path=str(Path(checkpoint_path).resolve()),
            config=asdict(config),
            description=description,
            metrics=metrics or {},
        )
        self._versions[name] = version
        self._save_index()
        return version

    def get(self, name: str) -> ModelVersion:
        if name not in self._versions:
            raise KeyError(f"Version '{name}' not found. Available: {self.list_names()}")
        return self._versions[name]

    def list_names(self) -> list[str]:
        return sorted(self._versions.keys())

    def list_versions(self) -> list[ModelVersion]:
        return [self._versions[n] for n in self.list_names()]

    def delete(self, name: str) -> None:
        if name not in self._versions:
            raise KeyError(f"Version '{name}' not found.")
        del self._versions[name]
        self._save_index()

    def load_model(
        self,
        name: str,
        device: str | None = None,
    ) -> tuple[TransformerLM, ModelVersion]:
        version = self.get(name)
        config = ModelConfig(**version.config)
        dev = resolve_device(device)
        model = TransformerLM(config).to(dev)
        load_checkpoint(version.path, model)
        model.eval()
        return model, version

    def auto_scan(self, runs_dir: str | Path) -> list[str]:
        """Scan a runs directory and register any checkpoint not yet in the registry.

        Looks for files named ``best.pt`` or ``checkpoint.pt`` inside each
        subdirectory of *runs_dir*.  The subdirectory name becomes the version
        name.  Returns list of newly registered version names.
        """
        runs_dir = Path(runs_dir)
        registered: list[str] = []
        if not runs_dir.exists():
            return registered
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for candidate in ("best.pt", "checkpoint.pt"):
                ckpt = run_dir / candidate
                if ckpt.exists() and run_dir.name not in self._versions:
                    try:
                        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                        raw_cfg = payload.get("config", {})
                        cfg = ModelConfig(**raw_cfg) if raw_cfg else ModelConfig()
                        self.register(
                            name=run_dir.name,
                            checkpoint_path=ckpt,
                            config=cfg,
                            description=f"Auto-scanned from {ckpt}",
                        )
                        registered.append(run_dir.name)
                    except Exception:
                        pass
                    break
        return registered
