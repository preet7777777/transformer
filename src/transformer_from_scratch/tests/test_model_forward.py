from __future__ import annotations

import pytest
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


def test_rope_model_forward_shape() -> None:
    config = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        seq_len=16,
        d_ff=64,
        dropout=0.0,
        positional_encoding="rope",
    )
    model = TransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (3, 24))
    logits = model(input_ids)
    assert logits.shape == (3, 24, 64)


def test_rope_allows_longer_context_than_learned() -> None:
    learned = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        seq_len=16,
        d_ff=64,
        dropout=0.0,
        positional_encoding="learned",
    )
    rope = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        seq_len=16,
        d_ff=64,
        dropout=0.0,
        positional_encoding="rope",
    )

    learned_model = TransformerLM(learned)
    rope_model = TransformerLM(rope)
    input_ids = torch.randint(0, learned.vocab_size, (2, 24))

    with pytest.raises(ValueError):
        _ = learned_model(input_ids)
    assert rope_model(input_ids).shape == (2, 24, 64)
