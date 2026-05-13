"""End-to-end demo pipeline."""

from __future__ import annotations

import argparse
import json

from .benchmark import run_benchmark
from .checkpoint import load_checkpoint
from .config import Config
from .data import NumpyDataset, build_dataloader
from .model import TransformerLM
from .prepare_synthetic import generate_synthetic_data
from .train import evaluate
from .train import main as train_main
from .utils import ensure_dir, resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an end-to-end Transformer demo")
    parser.add_argument("--output-dir", type=str, default="runs/demo_showcase")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--model-seq-len", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-valid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="progression",
        choices=["progression", "repeat", "copy", "random"],
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--benchmark-iterations", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    data_dir = output_dir / "data"
    runs_dir = output_dir / "runs"
    ensure_dir(data_dir)
    ensure_dir(runs_dir)

    generate_synthetic_data(
        output_dir=data_dir,
        vocab_size=args.vocab_size,
        model_seq_len=args.model_seq_len,
        n_train=args.n_train,
        n_valid=args.n_valid,
        seed=args.seed,
        mode=args.mode,
    )

    train_main(
        [
            "--train-path",
            str(data_dir / "train.npy"),
            "--valid-path",
            str(data_dir / "valid.npy"),
            "--out-dir",
            str(runs_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--vocab-size",
            str(args.vocab_size),
            "--seq-len",
            str(args.model_seq_len),
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
            "--device",
            str(device),
        ]
    )

    config = Config()
    config.model.vocab_size = args.vocab_size
    config.model.seq_len = args.model_seq_len
    config.model.d_model = args.d_model
    config.model.n_layers = args.n_layers
    config.model.n_heads = args.n_heads
    config.model.d_ff = args.d_ff
    config.model.dropout = args.dropout

    model = TransformerLM(config.model).to(device)
    checkpoint_path = runs_dir / "best.pt"
    load_checkpoint(checkpoint_path, model)

    valid_ds = NumpyDataset(data_dir / "valid.npy")
    valid_loader = build_dataloader(valid_ds, batch_size=args.batch_size, shuffle=False)
    validation_loss = evaluate(model, valid_loader, device)
    benchmark = run_benchmark(
        model=model,
        batch_size=min(args.batch_size, 8),
        seq_len=args.model_seq_len,
        device=device,
        warmup_steps=10,
        iterations=args.benchmark_iterations,
    )

    report = {
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "training_report": json.loads(
            (runs_dir / "training_report.json").read_text(encoding="utf-8")
        ),
        "validation_loss": validation_loss,
        "benchmark": benchmark,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
