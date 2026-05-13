"""Public text dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_text(url: str, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response:
        destination.write_bytes(response.read())
    return destination


def ensure_tiny_shakespeare(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "tinyshakespeare.txt"
    if not text_path.exists():
        download_text(TINY_SHAKESPEARE_URL, text_path)
    return text_path


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = chars
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    return np.asarray([stoi[ch] for ch in text], dtype=np.int64)


def decode(tokens: np.ndarray | list[int], itos: list[str]) -> str:
    return "".join(itos[int(i)] for i in tokens)


def make_windows(tokens: np.ndarray, seq_len: int, stride: int | None = None) -> np.ndarray:
    stride = stride or (seq_len + 1)
    if len(tokens) < seq_len + 1:
        raise ValueError("Not enough tokens to build a single training window")
    windows = []
    for start in range(0, len(tokens) - seq_len, stride):
        window = tokens[start : start + seq_len + 1]
        if len(window) == seq_len + 1:
            windows.append(window)
    if not windows:
        raise ValueError("No windows were created from the token stream")
    return np.stack(windows, axis=0)


def prepare_tiny_shakespeare(
    output_dir: str | Path,
    seq_len: int,
    train_fraction: float = 0.9,
    stride: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = ensure_tiny_shakespeare(output_dir)
    text = text_path.read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)

    encoded = encode(text, stoi)
    split_index = int(len(encoded) * train_fraction)
    train_tokens = encoded[:split_index]
    valid_tokens = encoded[split_index:]

    train_windows = make_windows(train_tokens, seq_len=seq_len, stride=stride)
    valid_windows = make_windows(valid_tokens, seq_len=seq_len, stride=stride)

    np.save(output_dir / "train.npy", train_windows)
    np.save(output_dir / "valid.npy", valid_windows)
    (output_dir / "vocab.json").write_text(
        json.dumps({"stoi": stoi, "itos": itos}, indent=2), encoding="utf-8"
    )

    return {
        "text_path": str(text_path),
        "train_path": str(output_dir / "train.npy"),
        "valid_path": str(output_dir / "valid.npy"),
        "vocab_path": str(output_dir / "vocab.json"),
        "vocab_size": len(itos),
        "train_windows": int(train_windows.shape[0]),
        "valid_windows": int(valid_windows.shape[0]),
        "sample_text": decode(valid_tokens[: min(200, len(valid_tokens))], itos),
    }
