"""Public dataset evaluation pipeline for Tiny Shakespeare."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from .baselines import bigram_baseline_loss, unigram_baseline_loss
from .benchmark import run_benchmark
from .checkpoint import load_checkpoint
from .config import ModelConfig
from .model import TransformerLM
from .public_data import prepare_tiny_shakespeare
from .train import evaluate
from .train import main as train_main
from .utils import ensure_dir, resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run public validation on Tiny Shakespeare")
    parser.add_argument("--output-dir", type=str, default="runs/public_eval")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--positional-encoding", type=str, default="rope", choices=["learned", "rope"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    data_dir = output_dir / "tiny_shakespeare"
    ensure_dir(data_dir)

    dataset_info = prepare_tiny_shakespeare(
        output_dir=data_dir,
        seq_len=args.seq_len,
        train_fraction=args.train_fraction,
        stride=args.stride,
    )

    raw_train_tokens = np.load(dataset_info["train_tokens_path"])
    raw_valid_tokens = np.load(dataset_info["valid_tokens_path"])
    vocab_size = int(dataset_info["vocab_size"])
    baseline_unigram_loss = unigram_baseline_loss(raw_train_tokens, raw_valid_tokens, vocab_size)
    baseline_bigram_loss = bigram_baseline_loss(raw_train_tokens, raw_valid_tokens, vocab_size)

    train_args = [
        "--train-path",
        dataset_info["train_path"],
        "--valid-path",
        dataset_info["valid_path"],
        "--out-dir",
        str(output_dir / "runs"),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--vocab-size",
        str(vocab_size),
        "--seq-len",
        str(args.seq_len),
        "--d-model",
        str(args.d_model),
        "--n-layers",
        str(args.n_layers),
        "--n-heads",
        str(args.n_heads),
        "--d-ff",
        str(args.d_ff),
        "--dropout",
        str(args.dropout),
        "--positional-encoding",
        str(args.positional_encoding),
    ]
    if args.device is not None:
        train_args.extend(["--device", str(args.device)])
    train_main(train_args)

    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        d_ff=args.d_ff,
        dropout=args.dropout,
        positional_encoding=args.positional_encoding,
    )
    model = TransformerLM(config).to(device)
    checkpoint_path = output_dir / "runs" / "best.pt"

    load_checkpoint(checkpoint_path, model)

    from .data import NumpyDataset, build_dataloader

    valid_ds = NumpyDataset(dataset_info["valid_path"])
    valid_loader = build_dataloader(valid_ds, batch_size=args.batch_size, shuffle=False)

    model_loss = evaluate(model, valid_loader, device)
    benchmark = run_benchmark(
        model=model,
        batch_size=min(args.batch_size, 64),
        seq_len=args.seq_len,
        device=device,
        warmup_steps=10,
        iterations=args.benchmark_iterations,
    )

    report = {
        "dataset": "Tiny Shakespeare",
        "source": "karpathy/char-rnn",
        "vocab_size": vocab_size,
        "seq_len": args.seq_len,
        "train_windows": dataset_info["train_windows"],
        "valid_windows": dataset_info["valid_windows"],
        "baseline_unigram_loss": baseline_unigram_loss,
        "baseline_bigram_loss": baseline_bigram_loss,
        "model_validation_loss": model_loss,
        "baseline_perplexity": float(math.exp(baseline_unigram_loss)),
        "baseline_bigram_perplexity": float(math.exp(baseline_bigram_loss)),
        "model_perplexity": float(math.exp(model_loss)),
        "checkpoint": str(checkpoint_path),
        "benchmark": benchmark,
    }
    report_path = output_dir / "public_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
