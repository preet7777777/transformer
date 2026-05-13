from __future__ import annotations

import numpy as np

from transformer_from_scratch.public_data import build_vocab, decode, encode, make_windows


def test_public_text_round_trip() -> None:
    text = "to be or not to be"
    stoi, itos = build_vocab(text)
    tokens = encode(text, stoi)
    assert decode(tokens, itos) == text


def test_make_windows() -> None:
    tokens = np.arange(20, dtype=np.int64)
    windows = make_windows(tokens, seq_len=4, stride=5)
    assert windows.shape == (4, 5)
    assert np.array_equal(windows[0], np.array([0, 1, 2, 3, 4], dtype=np.int64))
