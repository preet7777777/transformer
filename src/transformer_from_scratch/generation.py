"""Autoregressive token generation helpers."""

from __future__ import annotations

import argparse

import torch

from .checkpoint import load_checkpoint
from .config import ModelConfig
from .model import TransformerLM
from .utils import resolve_device, set_seed


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        top_values, top_indices = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(-1, top_indices, top_values)
        logits = filtered

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_tokens(
    model: TransformerLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    model.eval()
    tokens = prompt.clone()
    device = next(model.parameters()).device
    tokens = tokens.to(device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            window = tokens[:, -model.config.seq_len :]
            logits = model(window)
            next_logits = logits[:, -1, :]
            next_token = sample_next_token(next_logits, temperature=temperature, top_k=top_k)
            tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tokens from a Transformer checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--positional-encoding", type=str, default="learned")
    parser.add_argument("--prompt", type=int, nargs="*", default=[0])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)
    device = resolve_device(args.device)
    config = ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        seq_len=args.seq_len,
        d_ff=args.d_ff,
        dropout=args.dropout,
        positional_encoding=args.positional_encoding,
    )
    model = TransformerLM(config).to(device)
    load_checkpoint(args.checkpoint, model)

    prompt = torch.tensor([args.prompt], dtype=torch.long, device=device)
    generated = generate_tokens(
        model,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(" ".join(map(str, generated[0].tolist())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
