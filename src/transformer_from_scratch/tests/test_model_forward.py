from __future__ import annotations

import torch

from transformer_from_scratch.config import ModelConfig
from transformer_from_scratch.model import TransformerLM, olmo_tiny


def test_olmo_tiny_forward_shape() -> None:
    model = olmo_tiny()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 12))
    logits = model(input_ids)
    assert logits.shape == (2, 12, model.config.vocab_size)


def test_custom_model_forward_shape() -> None:
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        seq_len=16,
        d_ff=64,
        dropout=0.0,
    )
    model = TransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 16))
    logits = model(input_ids)
    assert logits.shape == (3, 16, 64)
