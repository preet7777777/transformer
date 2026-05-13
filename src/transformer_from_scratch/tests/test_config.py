from __future__ import annotations

from pathlib import Path

from transformer_from_scratch.config import Config, load_from_yaml, merge_dotlist, to_yaml


def test_load_and_override(tmp_path: Path) -> None:
    config = Config()
    config.model.vocab_size = 128
    config.train.batch_size = 8
    config_path = tmp_path / "config.yaml"
    to_yaml(config, config_path)

    loaded = load_from_yaml(config_path)
    assert loaded.model.vocab_size == 128
    assert loaded.train.batch_size == 8

    merged = merge_dotlist(loaded, ["model.seq_len=16", "train.lr=0.001", "data.train_path=data/train.npy"])
    assert merged.model.seq_len == 16
    assert merged.train.lr == 0.001
    assert merged.data.train_path == "data/train.npy"
