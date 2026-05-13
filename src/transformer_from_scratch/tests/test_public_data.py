from __future__ import annotations

from pathlib import Path

import numpy as np

from transformer_from_scratch.public_data import (
    build_vocab,
    decode,
    encode,
    make_windows,
    prepare_char_corpus_from_tokens,
)


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


def test_prepare_char_corpus_from_tokens(tmp_path: Path) -> None:
    text = "hello world"
    stoi, itos = build_vocab(text)
    tokens = encode(text, stoi)
    result = prepare_char_corpus_from_tokens(
        output_dir=tmp_path,
        corpus_name="demo",
        train_tokens=tokens[:6],
        valid_tokens=tokens[6:],
        stoi=stoi,
        itos=itos,
        seq_len=2,
    )

    assert Path(result["train_tokens_path"]).exists()
    assert Path(result["valid_tokens_path"]).exists()
    assert Path(result["train_path"]).exists()
    assert Path(result["valid_path"]).exists()
    assert result["corpus"] == "demo"
