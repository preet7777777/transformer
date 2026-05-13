"""Model benchmark CLI."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch

from .config import ModelConfig
from .model import TransformerLM
from .utils import resolve_device, set_seed


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_benchmark(
    model: TransformerLM,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    warmup_steps: int,
    iterations: int,
) -> dict[str, float | int | str]:
    model.eval()
    inputs = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    with torch.inference_mode():
        for _ in range(warmup_steps):
            _ = model(inputs)

        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(inputs)
        elapsed = time.perf_counter() - start

    total_tokens = batch_size * seq_len * iterations
    tokens_per_second = total_tokens / max(elapsed, 1e-9)
    milliseconds_per_batch = (elapsed / max(iterations, 1)) * 1000.0

    return {
        "device": str(device),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "warmup_steps": warmup_steps,
        "iterations": iterations,
        "parameters": count_parameters(model),
        "milliseconds_per_batch": round(milliseconds_per_batch, 4),
        "tokens_per_second": round(tokens_per_second, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the Transformer model")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--positional-encoding", type=str, default="learned")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    config = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        d_ff=args.d_ff,
        dropout=args.dropout,
        positional_encoding=args.positional_encoding,
    )
    model = TransformerLM(config).to(device)
    result = run_benchmark(
        model=model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        warmup_steps=args.warmup_steps,
        iterations=args.iterations,
    )
    print(json.dumps({"config": asdict(config), "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
