"""Training CLI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config, load_from_yaml, merge_dotlist
from .data import NumpyDataset, build_dataloader
from .model import TransformerLM
from .optim import CosineWarmupScheduler, build_adamw_optimizer
from .utils import ensure_dir, resolve_device, set_seed

LOGGER = logging.getLogger("transformer_from_scratch.train")


def evaluate(model: TransformerLM, dataloader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += float(loss.item())
            total_batches += 1
    return total_loss / max(1, total_batches)


def train_one_epoch(
    model: TransformerLM,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler | None,
    device: torch.device,
    max_grad_norm: float,
    global_step: int,
    epoch: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    progress = tqdm(dataloader, desc=f"epoch {epoch + 1}", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        global_step += 1
        if scheduler is not None:
            scheduler.step(global_step)
        total_loss += float(loss.item())
        total_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(1, total_batches), global_step


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny Transformer language model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--valid-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="runs/demo")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.config:
        config = load_from_yaml(args.config)
    else:
        config = Config()

    if args.overrides:
        config = merge_dotlist(config, args.overrides)

    if args.train_path is not None:
        config.data.train_path = args.train_path
    if args.valid_path is not None:
        config.data.valid_path = args.valid_path
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.weight_decay is not None:
        config.train.weight_decay = args.weight_decay
    if args.warmup_steps is not None:
        config.train.warmup_steps = args.warmup_steps
    if args.max_grad_norm is not None:
        config.train.max_grad_norm = args.max_grad_norm
    if args.seed is not None:
        config.train.seed = args.seed
    if args.vocab_size is not None:
        config.model.vocab_size = args.vocab_size
    if args.seq_len is not None:
        config.model.seq_len = args.seq_len
    if args.d_model is not None:
        config.model.d_model = args.d_model
    if args.n_layers is not None:
        config.model.n_layers = args.n_layers
    if args.n_heads is not None:
        config.model.n_heads = args.n_heads
    if args.d_ff is not None:
        config.model.d_ff = args.d_ff
    if args.dropout is not None:
        config.model.dropout = args.dropout

    set_seed(config.train.seed)
    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    train_ds = NumpyDataset(config.data.train_path)
    valid_ds = NumpyDataset(config.data.valid_path)
    train_loader = build_dataloader(train_ds, batch_size=config.train.batch_size, shuffle=True)
    valid_loader = build_dataloader(valid_ds, batch_size=config.train.batch_size, shuffle=False)

    model = TransformerLM(config.model).to(device)
    optimizer = build_adamw_optimizer(model, lr=config.train.lr, weight_decay=config.train.weight_decay)
    total_steps = max(1, len(train_loader) * config.train.epochs)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=config.train.warmup_steps, total_steps=total_steps)

    global_step = 0
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler)
        global_step = int(payload.get("step", 0))
        start_epoch = int(payload.get("epoch", 0)) + 1
        LOGGER.info("Resumed from %s at step %s", args.resume, global_step)

    best_val = float("inf")
    for epoch in range(start_epoch, config.train.epochs):
        train_loss, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config.train.max_grad_norm,
            global_step,
            epoch,
        )
        val_loss = evaluate(model, valid_loader, device)
        LOGGER.info("epoch=%s train_loss=%.4f val_loss=%.4f", epoch + 1, train_loss, val_loss)

        save_checkpoint(
            out_dir / "checkpoint.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=global_step,
            extra={"epoch": epoch, "config": config},
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                extra={"epoch": epoch, "config": config},
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
