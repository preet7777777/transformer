"""Public benchmark and ablation suite for WikiText-2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .baselines import bigram_baseline_loss, unigram_baseline_loss
from .benchmark import run_benchmark
from .checkpoint import load_checkpoint
from .config import ModelConfig
from .data import NumpyDataset, build_dataloader
from .model import TransformerLM
from .public_data import prepare_wikitext2
from .train import evaluate
from .train import main as train_main
from .utils import ensure_dir, resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a public WikiText-2 benchmark and ablation suite")
    parser.add_argument("--output-dir", type=str, default="runs/public_benchmark")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--benchmark-seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    return parser


def train_and_score_variant(
    *,
    run_dir: Path,
    dataset_info: dict[str, object],
    seq_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    dropout: float,
    positional_encoding: str,
    tie_embeddings: bool,
    device,
    benchmark_iterations: int,
    benchmark_seq_len: int | None = None,
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    train_main(
        [
            "--train-path",
            str(dataset_info["train_path"]),
            "--valid-path",
            str(dataset_info["valid_path"]),
            "--out-dir",
            str(run_dir),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--lr",
            str(lr),
            "--vocab-size",
            str(dataset_info["vocab_size"]),
            "--seq-len",
            str(seq_len),
            "--d-model",
            str(d_model),
            "--n-layers",
            str(n_layers),
            "--n-heads",
            str(n_heads),
            "--d-ff",
            str(d_ff),
            "--dropout",
            str(dropout),
            "--positional-encoding",
            positional_encoding,
            "--device",
            str(device),
            "--tie-embeddings" if tie_embeddings else "--no-tie-embeddings",
        ]
    )

    model = TransformerLM(
        ModelConfig(
            vocab_size=int(dataset_info["vocab_size"]),
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            seq_len=seq_len,
            d_ff=d_ff,
            dropout=dropout,
            positional_encoding=positional_encoding,
            tie_embeddings=tie_embeddings,
        )
    ).to(device)
    checkpoint_path = run_dir / "best.pt"
    load_checkpoint(checkpoint_path, model)

    valid_ds = NumpyDataset(dataset_info["valid_path"])
    valid_loader = build_dataloader(valid_ds, batch_size=batch_size, shuffle=False)
    validation_loss = evaluate(model, valid_loader, device)
    benchmark = run_benchmark(
        model=model,
        batch_size=min(batch_size, 64),
        seq_len=seq_len,
        device=device,
        warmup_steps=10,
        iterations=benchmark_iterations,
    )

    result: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "validation_loss": validation_loss,
        "perplexity": float(math.exp(validation_loss)),
        "benchmark": benchmark,
    }

    if benchmark_seq_len is not None and positional_encoding == "rope":
        result["long_context_benchmark"] = run_benchmark(
            model=model,
            batch_size=min(batch_size, 64),
            seq_len=benchmark_seq_len,
            device=device,
            warmup_steps=10,
            iterations=benchmark_iterations,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    data_dir = output_dir / "wikitext-2"
    ensure_dir(data_dir)

    dataset_info = prepare_wikitext2(
        output_dir=data_dir,
        seq_len=args.seq_len,
        stride=args.stride,
    )

    raw_train_tokens = np.load(Path(dataset_info["train_tokens_path"]))
    raw_valid_tokens = np.load(Path(dataset_info["valid_tokens_path"]))
    vocab_size = int(dataset_info["vocab_size"])
    unigram_loss = unigram_baseline_loss(raw_train_tokens, raw_valid_tokens, vocab_size)
    bigram_loss = bigram_baseline_loss(raw_train_tokens, raw_valid_tokens, vocab_size)

    runs_dir = output_dir / "runs"
    variants = [
        {
            "name": "rope_tied",
            "positional_encoding": "rope",
            "tie_embeddings": True,
        },
        {
            "name": "learned_tied",
            "positional_encoding": "learned",
            "tie_embeddings": True,
        },
        {
            "name": "rope_untied",
            "positional_encoding": "rope",
            "tie_embeddings": False,
        },
    ]

    variant_results: list[dict[str, object]] = []
    for variant in variants:
        run_dir = runs_dir / variant["name"]
        metrics = train_and_score_variant(
            run_dir=run_dir,
            dataset_info=dataset_info,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            positional_encoding=str(variant["positional_encoding"]),
            tie_embeddings=bool(variant["tie_embeddings"]),
            device=device,
            benchmark_iterations=args.benchmark_iterations,
            benchmark_seq_len=args.benchmark_seq_len,
        )
        variant_results.append({"name": variant["name"], **metrics})

    report = {
        "dataset": dataset_info["corpus"],
        "source": dataset_info.get("source", "pytorch/examples"),
        "vocab_size": vocab_size,
        "seq_len": args.seq_len,
        "benchmark_seq_len": args.benchmark_seq_len,
        "train_tokens": dataset_info["train_tokens"],
        "valid_tokens": dataset_info["valid_tokens"],
        "train_windows": dataset_info["train_windows"],
        "valid_windows": dataset_info["valid_windows"],
        "baseline_unigram_loss": unigram_loss,
        "baseline_bigram_loss": bigram_loss,
        "baseline_unigram_perplexity": float(math.exp(unigram_loss)),
        "baseline_bigram_perplexity": float(math.exp(bigram_loss)),
        "variants": variant_results,
    }

    report_path = output_dir / "public_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
