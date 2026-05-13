"""Dataclass configuration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    seq_len: int = 32
    d_ff: int = 256
    dropout: float = 0.1
    positional_encoding: str = "learned"
    tie_embeddings: bool = True


@dataclass
class DataConfig:
    train_path: str = "train.npy"
    valid_path: str = "valid.npy"


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_NESTED_TYPES = {
    "model": ModelConfig,
    "data": DataConfig,
    "train": TrainConfig,
}


def _update_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _to_dict(config: Config) -> dict[str, Any]:
    return asdict(config)


def _from_dict(data: dict[str, Any]) -> Config:
    kwargs: dict[str, Any] = {}
    for key, typ in _NESTED_TYPES.items():
        section = data.get(key, {})
        kwargs[key] = typ(**section) if isinstance(section, dict) else section
    return Config(**kwargs)


def load_from_yaml(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _from_dict(payload)


def merge_dotlist(config: Config, overrides: list[str]) -> Config:
    data = _to_dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        _update_nested(data, key, value)
    return _from_dict(data)


def to_yaml(config: Config, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_to_dict(config), handle, sort_keys=False)
