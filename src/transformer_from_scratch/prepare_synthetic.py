"""Generate small numpy shards for quick experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_synthetic_data(
    output_dir: str | Path = ".",
    vocab_size: int = 256,
    model_seq_len: int = 32,
    n_train: int = 256,
    n_valid: int = 64,
    seed: int = 42,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total_len = model_seq_len + 1
    train = rng.integers(0, vocab_size, size=(n_train, total_len), dtype=np.int64)
    valid = rng.integers(0, vocab_size, size=(n_valid, total_len), dtype=np.int64)
    np.save(output / "train.npy", train)
    np.save(output / "valid.npy", valid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic LM data")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--model-seq-len", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-valid", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_synthetic_data(**vars(args))
