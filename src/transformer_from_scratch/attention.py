"""Attention primitives."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        mask_bool = mask.bool() if mask.dtype != torch.bool else mask
        scores = scores.masked_fill(~mask_bool, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_embedding(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Apply RoPE to a tensor shaped [batch, heads, seq, head_dim]."""

    head_dim = x.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")

    device = x.device
    dtype = x.dtype
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cos = torch.cos(torch.cat([freqs, freqs], dim=-1))[None, None, :, :]
    sin = torch.sin(torch.cat([freqs, freqs], dim=-1))[None, None, :, :]
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        positional_encoding: str = "learned",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if positional_encoding not in {"learned", "rope"}:
            raise ValueError("positional_encoding must be 'learned' or 'rope'")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.positional_encoding = positional_encoding
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        if self.positional_encoding == "rope":
            q = apply_rotary_embedding(q, seq_len)
            k = apply_rotary_embedding(k, seq_len)
        attn = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.dropout)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(attn)


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return (
        torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        .unsqueeze(0)
        .unsqueeze(0)
    )
