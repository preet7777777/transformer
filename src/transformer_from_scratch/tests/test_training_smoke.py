from __future__ import annotations

from pathlib import Path

import torch

from transformer_from_scratch.prepare_synthetic import generate_synthetic_data
from transformer_from_scratch.train import evaluate, main as train_main
from transformer_from_scratch.config import Config
from transformer_from_scratch.data import NumpyDataset, build_dataloader
from transformer_from_scratch.model import TransformerLM


def test_training_smoke(tmp_path: Path) -> None:
    generate_synthetic_data(tmp_path, vocab_size=64, model_seq_len=16, n_train=32, n_valid=16, seed=1)
    out_dir = tmp_path / "runs"
    train_main(
        [
            "--train-path",
            str(tmp_path / "train.npy"),
            "--valid-path",
            str(tmp_path / "valid.npy"),
            "--out-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--lr",
            "0.001",
            "--vocab-size",
            "64",
            "--seq-len",
            "16",
            "--d-model",
            "32",
            "--n-layers",
            "1",
            "--n-heads",
            "4",
            "--d-ff",
            "64",
            "--dropout",
            "0.0",
        ]
    )
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "best.pt").exists()


def test_validation_loss_is_finite(tmp_path: Path) -> None:
    generate_synthetic_data(tmp_path, vocab_size=64, model_seq_len=16, n_train=16, n_valid=8, seed=2)
    config = Config()
    config.model.vocab_size = 64
    config.model.seq_len = 16
    config.model.d_model = 32
    config.model.n_layers = 1
    config.model.n_heads = 4
    config.model.d_ff = 64
    config.model.dropout = 0.0

    model = TransformerLM(config.model)
    valid_ds = NumpyDataset(tmp_path / "valid.npy")
    valid_loader = build_dataloader(valid_ds, batch_size=4, shuffle=False)
    loss = evaluate(model, valid_loader, torch.device("cpu"))
    assert torch.isfinite(torch.tensor(loss))
