"""Classical language-model baselines for public evaluations."""

from __future__ import annotations

import numpy as np


def unigram_baseline_loss(
    train_tokens: np.ndarray, valid_tokens: np.ndarray, vocab_size: int, smoothing: float = 1.0
) -> float:
    train_tokens = np.asarray(train_tokens, dtype=np.int64).ravel()
    valid_tokens = np.asarray(valid_tokens, dtype=np.int64).ravel()
    counts = np.bincount(train_tokens, minlength=vocab_size).astype(np.float64)
    probs = (counts + smoothing) / (counts.sum() + smoothing * vocab_size)
    return float(-np.mean(np.log(probs[valid_tokens[1:]])))


def bigram_baseline_loss(
    train_tokens: np.ndarray, valid_tokens: np.ndarray, vocab_size: int, smoothing: float = 1.0
) -> float:
    train_tokens = np.asarray(train_tokens, dtype=np.int64).ravel()
    valid_tokens = np.asarray(valid_tokens, dtype=np.int64).ravel()

    if train_tokens.size < 2 or valid_tokens.size < 2:
        raise ValueError("Need at least two tokens to compute a bigram baseline")

    context = train_tokens[:-1]
    targets = train_tokens[1:]
    pair_counts = np.bincount(
        context * vocab_size + targets, minlength=vocab_size * vocab_size
    ).astype(np.float64)
    pair_counts = pair_counts.reshape(vocab_size, vocab_size)
    context_counts = pair_counts.sum(axis=1)

    valid_context = valid_tokens[:-1]
    valid_targets = valid_tokens[1:]
    probs = (pair_counts[valid_context, valid_targets] + smoothing) / (
        context_counts[valid_context] + smoothing * vocab_size
    )
    return float(-np.mean(np.log(probs)))
