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
    mode: str = "progression",
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total_len = model_seq_len + 1

    def build_split(n_rows: int) -> np.ndarray:
        if mode == "random":
            return rng.integers(0, vocab_size, size=(n_rows, total_len), dtype=np.int64)
        if mode == "progression":
            starts = rng.integers(0, vocab_size, size=(n_rows, 1), dtype=np.int64)
            offsets = np.arange(total_len, dtype=np.int64)[None, :]
            return (starts + offsets) % vocab_size
        if mode == "repeat":
            period = max(2, min(8, vocab_size))
            base = np.arange(period, dtype=np.int64)
            repeats = int(np.ceil(total_len / period))
            template = np.tile(base, repeats)[:total_len]
            shifts = rng.integers(0, vocab_size, size=(n_rows, 1), dtype=np.int64)
            return (shifts + template[None, :]) % vocab_size
        if mode == "copy":
            tokens = rng.integers(0, vocab_size, size=(n_rows, 1), dtype=np.int64)
            return np.repeat(tokens, repeats=total_len, axis=1)
        raise ValueError(f"Unknown synthetic mode: {mode}")

    train = build_split(n_train)
    valid = build_split(n_valid)
    np.save(output / "train.npy", train)
    np.save(output / "valid.npy", valid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic LM data")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--model-seq-len", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-valid", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="progression",
        choices=["progression", "repeat", "copy", "random"],
    )
    args = parser.parse_args(argv)
    generate_synthetic_data(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
