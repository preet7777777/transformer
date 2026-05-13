
New Chat
Today
End to end project
This week
Understanding
preet-7777777
Get PRO
Models 130
Settings

    Pasted Content
Clipboard source

pls build this end to end and give me zip file

The user wants me to create a complete Git repository called "transformer-from-scratch" with all the specified files. They want me to output all files with their paths and contents. They also mentioned wanting a zip file, but I can't create actual zip files - I'll output all the file contents which they can use to create the repository.

Let me carefully go through each file and create the complete repository.

Let me plan out all the files:

    README.md
    LICENSE (MIT)
    pyproject.toml
    requirements.txt
    Makefile
    .pre-commit-config.yaml
    .github/workflows/main.yml
    Dockerfile
    src/transformer_from_scratch/init.py
    src/transformer_from_scratch/config.py
    src/transformer_from_scratch/registrable.py
    src/transformer_from_scratch/model.py
    src/transformer_from_scratch/attention.py
    src/transformer_from_scratch/data.py
    src/transformer_from_scratch/prepare_synthetic.py
    src/transformer_from_scratch/train.py
    src/transformer_from_scratch/optim.py
    src/transformer_from_scratch/checkpoint.py
    src/transformer_from_scratch/utils.py
    src/transformer_from_scratch/tests/test_model_forward.py
    src/transformer_from_scratch/tests/test_training_smoke.py
    src/transformer_from_scratch/tests/test_config.py
    src/transformer_from_scratch/tests/init.py

Let me write each file carefully.

For the model, I need:

    TransformerLM with token embedding, positional embeddings (I'll use learned positional embeddings for simplicity), N transformer blocks, layernorm, lm_head
    TransformerBlock with MultiHeadSelfAttention, residuals, LayerNorm, FeedForward with GELU
    olmo_tiny() factory

For attention.py, I need scaled-dot-product attention implementation.

For data.py, NumpyDataset, DataLoader wrapper, streaming-friendly API.

For train.py, CLI with argparse, checkpointing, resume, validation, logging.

For config.py, dataclass-based Config with YAML and dotlist overrides.

For registrable.py, small Registrable mixin.

Let me write all of these now.

Let me think about the attention.py vs model.py split. The requirement says attention.py should have attention backend(s), scaled-dot-product attention. And model.py should have the Transformer LM implementation. So attention.py will have the scaled dot product attention function, and model.py will import it.

Let me code everything up carefully.

For the data collation: input=seq[:-1], target=seq[1:]

For the config system, I'll use dataclasses and support YAML loading and dotlist overrides like "model.n_layers=4".

Let me write all the files now.

I need to be careful about:

    Making tests pass quickly on CPU
    Default config being tiny
    Checkpointing working correctly
    Streaming-friendly data API

Let me start writing the code.

For config.py, I'll create a nested dataclass structure:

    ModelConfig (vocab_size, d_model, n_layers, n_heads, seq_len, etc.)
    TrainConfig (epochs, batch_size, lr, etc.)
    Config (model, train)

For registrable.py, a simple registry pattern.

For model.py:

    TransformerLM class
    TransformerBlock class
    FeedForward class
    olmo_tiny() function

For attention.py:

    scaled_dot_product_attention function
    MultiHeadAttention class (or put this in model.py?)

Actually, let me put MultiHeadAttention in model.py since it uses the attention function from attention.py.

Wait, re-reading the requirements: "attention.py - attention backend(s), scaled-dot-product attention". So I'll put the core attention function and potentially a MultiHeadAttention class there.

Let me structure it as:

    attention.py: scaled_dot_product_attention function, MultiHeadAttention class
    model.py: TransformerBlock, TransformerLM, FeedForward, olmo_tiny

Actually, let me keep it simple. I'll put MultiHeadAttention in attention.py and have model.py import it.

Let me write all files now.

For the training script, I need:

    argparse CLI
    training loop
    validation loop
    checkpointing
    resume support- logging

Let me think about the streaming-friendly API. The requirement says "streaming-friendly ingestion API (able to append new .npy shards and continue training by reloading dataset)". So NumpyDataset should support loading multiple .npy files and being able to add more.

Let me write everything now. I'll be thorough but compact.

For pyproject.toml, I'll use a simple setup with:

    torch, numpy, pyyaml, tqdm, pytest as dependencies- black, isort, ruff as dev dependencies

Actually, requirements say "Only allow these runtime dependencies: torch, numpy, pyyaml, tqdm, pytest". So pytest is a runtime dependency for tests but that's fine. I'll include dev dependencies for formatting/linting.

Let me now write each file:

README.md
LICENSE
pyproject.toml
requirements.txt
Makefile
.pre-commit-config.yaml
.github/workflows/main.yml
Dockerfile
src/transformer_from_scratch/__init__.py
src/transformer_from_scratch/config.py
src/transformer_from_scratch/registrable.py
src/transformer_from_scratch/model.py
src/transformer_from_scratch/attention.py
src/transformer_from_scratch/data.py
src/transformer_from_scratch/prepare_synthetic.py
src/transformer_from_scratch/train.py
src/transformer_from_scratch/optim.py
src/transformer_from_scratch/checkpoint.py
src/transformer_from_scratch/utils.py
src/transformer_from_scratch/tests/__init__.py
src/transformer_from_scratch/tests/test_model_forward.py
src/transformer_from_scratch/tests/test_training_smoke.py
src/transformer_from_scratch/tests/test_config.py

Let me write each one carefully now.

For config.py:

"""Dataclass-based configuration with YAML and dotlist override support."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    """Configuration for the TransformerLM model."""
    vocab_size: int = 1024
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    seq_len: int = 64
    d_ff: int = 512
    dropout: float = 0.1


@dataclass
class DataConfig:
    """Configuration for data paths and processing."""
    train_path: str = "train.npy"
    valid_path: str = "valid.npy"


@dataclass 
class TrainConfig:
    """Configuration for training loop."""
    epochs: int = 1
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass
class Config:
    """Top-level configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _set_nested(d: Dict[str, Any], key: str, value: Any) -> None:
    """Set a nested dictionary value using dot-separated key."""
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass to a dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    return obj


def _dict_to_dataclass(cls: type, d: Dict[str, Any]) -> Any:
    """Recursively convert a dict to a dataclass."""
    fields = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in fields:
            continue
        # Check if the field type is itself a dataclass
        # We'll handle known nested dataclasses
        if k == "model" and cls is Config:
            kwargs[k] = _dict_to_dataclass(ModelConfig, v) if isinstance(v, dict) else v
        elif k == "data" and cls is Config:
            kwargs[k] = _dict_to_dataclass(DataConfig, v) if isinstance(v, dict) else v
        elif k == "train" and cls is Config:
            kwargs[k] = _dict_to_dataclass(TrainConfig, v) if isinstance(v, dict) else v
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_from_yaml(path: str | Path) -> Config:
    """Load configuration from a YAML file."""
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    return _dict_to_dataclass(Config, d)


def merge_dotlist(config: Config, dotlist: List[str]) -> Config:
    """Merge dotlist overrides like 'section.sub=val' into config."""
    d = _dataclass_to_dict(config)
    for item in dotlist:
        key, value = item.split("=", 1)
        # Try to parse value as int, float, or keep as string
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        _set_nested(d, key, value)
    return _dict_to_dataclass(Config, d)

Hmm, I need to be careful about the Config type annotation and dataclass_to_dict. Let me refine.

Actually, I realize I should be more careful with the typos. Let me rethink.

For _dict_to_dataclass, I need a more generic approach. Let me use a mapping of field names to their dataclass types._

Actually, let me simplify. I'll use a recursive approach that inspects the annotation to determine if a field is a nested dataclass.

Let me think about this more carefully...

Actually, the simplest approach is: convert Config to dict, apply dotlist overrides to the dict, then convert back. For converting back, I just need to know the types of nested fields.

Let me use a simpler approach:

def _dict_to_dataclass(cls, d):
    field_types = {}
    for f in dataclasses.fields(cls):
        field_types[f.name] = f.type
    kwargs = {}
    for k, v in d.items():
        if k in field_types:
            expected_type = field_types[k]
            if dataclasses.is_dataclass(expected_type) and isinstance(v, dict):
                kwargs[k] = _dict_to_dataclass(expected_type, v)
            else:
                kwargs[k] = v
    return cls(**kwargs)

Wait, f.type might be a string annotation. Let me handle this differently.

Actually in Python 3.9+, with from __future__ import annotations, all annotations are strings. So I need to resolve them differently.

Let me just use a simple mapping approach:

_NESTED_CLASSES = {
    "model": ModelConfig,
    "data": DataConfig, 
    "train": TrainConfig,
}

This is the simplest and most robust approach.

Let me write all files now carefully.

For the model, attention.py, etc.

Let me think about attention.py:

"""Attention mechanisms for the Transformer model."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Optional


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.
    
    Args:
        query: (B, H, T, D_k)
        key: (B, H, T, D_k)
        value: (B, H, T, D_v)
        mask: optional mask (B, 1, T, T) or (1, 1, T, T)
        dropout: optional dropout module
    
    Returns:
        output: (B, H, T, D_v)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    return torch.matmul(attn_weights, value)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        
        q = self.wq(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        out = scaled_dot_product_attention(q, k, v, mask=mask, dropout=self.attn_dropout)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(out))

For model.py:

"""Compact Transformer language model implementation."""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm residual connections."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Transformer language model.
    
    Consists of:
    - Token embedding
    - Learned positional embedding
    - N Transformer blocks
    - Final LayerNorm
    - LM head projecting to vocabulary    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying between token embedding and lm_head
        self.lm_head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_ids: (B, T) token indices        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        
        x = self.dropout(self.token_emb(input_ids) + self.pos_emb(positions))
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        
        for block in self.blocks:
            x = block(x, mask=causal_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def olmo_tiny() -> TransformerLM:
    """Create a tiny Transformer model for testing."""
    config = ModelConfig()
    return TransformerLM(config)

For data.py:

"""Data loading utilities for numpy-based datasets."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NumpyDataset(Dataset):
    """Dataset that reads token sequences from a .npy file.
    
    The .npy file should have shape (num_sequences, seq_len+1) so that
    input = seq[:-1] and target = seq[1:].
    """
    
    def __init__(self, path: str | Path):
        self.data = np.load(str(path), allow_pickle=False)
        if self.data.ndim == 1:
            # Reshape flat array into (N, seq_len)
            # Assume it was saved as flat, need a seq_len
            raise ValueError("Expected 2D array (num_sequences, seq_len+1), got 1D")
    
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data[idx], dtype=torch.long)


def collate_fn(batch: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function that splits each sequence into input and target.
    
    input = seq[:-1], target = seq[1:]
    """
    sequences = torch.stack(batch)
    return sequences[:, :-1], sequences[:, 1:]


def make_dataloader(dataset: Dataset, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader with the appropriate collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=False,
    )


class StreamingDataset:
    """Streaming-friendly dataset that can load multiple .npy shards.
    
    Supports appending new shards for continuous data ingestion.
    """
    
    def __init__(self, paths: Optional[List[str | Path]] = None):
        self.shards: List[str] = []
        self._current_data: Optional[np.ndarray] = None
        self._current_shard_idx: int = 0
        if paths:
            for p in paths:
                self.add_shard(p)
    
    def add_shard(self, path: str | Path) -> None:
        """Add a new .npy shard file to the dataset."""
        self.shards.append(str(path))
    
    def load_shard(self, idx: int) -> np.ndarray:
        """Load a specific shard."""
        return np.load(self.shards[idx], allow_pickle=False)
    
    def load_all(self) -> np.ndarray:
        """Load and concatenate all shards."""
        arrays = [np.load(s, allow_pickle=False) for s in self.shards]
        return np.concatenate(arrays, axis=0)
    
    def to_dataset(self) -> NumpyDataset:
        """Convert to a NumpyDataset by loading all shards."""
        all_data = self.load_all()
        ds = NumpyDataset.__new__(NumpyDataset)
        ds.data = all_data
        return ds

Hmm, for the NumpyDataset when it's From StreamingDataset, I need to set it properly. Let me think...

Actually, let me just make NumpyDataset accept arrays too:

class NumpyDataset(Dataset):
    def __init__(self, path_or_array, allow_pickle=False):
        if isinstance(path_or_array, np.ndarray):
            self.data = path_or_array
        else:
            self.data = np.load(str(path_or_array), allow_pickle=allow_pickle)

That's cleaner.

For optim.py:

"""Optimizer and scheduler utilities."""
from __future__ import annotations

import math
from typing import List

import torch
from torch.optim import Optimizer


def build_adamw_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    no_decay_params: List[str] | None = None,
) -> Optimizer:
    """Build an AdamW optimizer with optional per-parameter weight decay."""
 if no_decay_params is None:
        no_decay_params = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    
    decay_params = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_params):
            no_decay.append(param)
        else:
            decay_params.append(param)
    
    optimizer_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    
    return torch.optim.AdamW(optimizer_groups, lr=lr)


class CosineWithWarmupScheduler:
    """Cosine learning rate scheduler with linear warmup.
    
    Args:
        optimizer: The optimizer to adjust.
        warmup_steps: Number of warmup steps.
        max_steps: Total number of training steps.
        min_lr: Minimum learning rate ratio (relative to initial lr).
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
    
    def step(self, step: int) -> float:
        """Update learning rate for the given step."""
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * scale
        
        return self.optimizer.param_groups[0]["lr"]

For checkpoint.py:

"""Checkpoint save/load utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.optim import Optimizer


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    step: int,
    path: str | Path,
) -> None:
    """Save a training checkpoint.
    
    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        step: Current training step.
        path: File path for the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
) -> int:
    """Load a training checkpoint.
    
    Args:
        path: File path of the checkpoint.
        model: The model to load weights into.
        optimizer: Optional optimizer to load state into.
    
    Returns:
        The training step from the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"]

For utils.py:

"""Utility helpers."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model: torch.nn.Module, path: str | Path) -> None:
    """Save model weights to a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str | Path) -> torch.nn.Module:
    """Load model weights from a file."""
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    return model

For prepare_synthetic.py:

"""Generate small synthetic train.npy and valid.npy for smoke testing."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_synthetic_data(
    output_dir: str | Path = ".",
    vocab_size: int = 1024,
    seq_len: int = 65,  # +1 for input/target split
    n_train: int = 1024,
    n_valid: int = 128,
    seed: int = 42,
) -> None:
    """Generate synthetic .npy files for training and validation.
    
    Each sequence is of length seq_len (input=seq[:-1], target=seq[1:]).
    """
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_data = rng.integers(0, vocab_size, size=(n_train, seq_len), dtype=np.int64)
    valid_data = rng.integers(0, vocab_size, size=(n_valid, seq_len), dtype=np.int64)
    
    np.save(output_dir / "train.npy", train_data)
    np.save(output_dir / "valid.npy", valid_data)
    print(f"Generated {n_train} train and {n_valid} valid sequences in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=65)
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-valid", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_synthetic_data(**vars(args))

Wait, I need to make sure seq_len in data is 65 (one more than the model's seq_len of 64) so that we can split into input (64) and target (64). That's correct.

For train.py:

"""CLI training script for TransformerLM."""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import Config, load_from_yaml, merge_dotlist
from .model import TransformerLM, olmo_tiny
from .data import NumpyDataset, make_dataloader
from .optim import build_adamw_optimizer, CosineWithWarmupScheduler
from .checkpoint import save_checkpoint, load_checkpoint
from .utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    max_grad_norm: float,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_steps = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step(total_steps)
        
        total_loss += loss.item()
        total_steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / max(1, total_steps)


def evaluate(model: nn.Module, dataloader, device: torch.device) -> float:
    """Evaluate model and return average loss."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
            total_loss += loss.item()
            total_steps += 1
    
    return total_loss / max(1, total_steps)


def main(args=None):
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train TransformerLM")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--valid-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--overrides", nargs="*", default=[], help="Dotlist overrides like model.n_layers=4")
    args = parser.parse_args(args)
    
    # Load config
    if args.config:
        config = load_from_yaml(args.config)
    else:
        config = Config()
    
    # Apply dotlist overrides
    if args.overrides:
        config = merge_dotlist(config, args.overrides)
    
    # Apply CLI overrides
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
    if args.seed is not None:
        config.train.seed = args.seed
    
    set_seed(config.train.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model
    model = TransformerLM(config.model).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data
    train_dataset = NumpyDataset(config.data.train_path)
    valid_dataset = NumpyDataset(config.data.valid_path)
    train_loader = make_dataloader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    valid_loader = make_dataloader(valid_dataset, batch_size=config.train.batch_size, shuffle=False)
    
    # Optimizer
    optimizer = build_adamw_optimizer(model, lr=config.train.lr, weight_decay=config.train.weight_decay)
    
    # Scheduler
    total_steps = len(train_loader) * config.train.epochs
    scheduler = CosineWithWarmupScheduler(
        optimizer,
        warmup_steps=config.train.warmup_steps,
        max_steps=total_steps,
    )
    
    # Resume
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        logger.info(f"Resumed from step {start_step}")
    
    # Output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_valid_loss = float("inf")
    for epoch in range(config.train.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, config.train.max_grad_norm, device, epoch
        )
        valid_loss = evaluate(model, valid_loader, device)
        
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")
        
        # Save checkpoint
        ckpt_path = out_dir / "checkpoint.pt"
        save_checkpoint(model, optimizer, epoch, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, optimizer, epoch, out_dir / "best_checkpoint.pt")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()

Wait, the scheduler.step() call is inside train_one_epoch but total_steps is tracked there. I need to make the scheduler stepping global. Let me rethink.

Actually, the scheduler step counter should be global. Let me adjust. In the training loop function, I'll pass a starting global step and return the final step. Or I can track it outside.

Let me redesign: The scheduler should step per optimizer step, and the step counter should be maintained across epochs. Let me use a global step counter.

def train_one_epoch(model, dataloader, optimizer, scheduler, max_grad_norm, device, epoch, global_step=0):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        global_step += 1
        if scheduler is not None:
            scheduler.step(global_step)
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, global_step

OK let me now think about the train.py again. Actually let me also count per-step rather than batch-based. Since global_step increments per batch.

Let me think about the scheduler more carefully. The CosineWithWarmupScheduler.step() takes a step number. So we need to call it with the current global step.

Let me restructure. I'll have a train loop that goes epoch by epoch and batch by batch, calling scheduler.step() for each batch.

Let me now write the final version of train.py.

Also for registrable.py, I need a simple mixin. Let me think about what it should do.

The registrable pattern maps string type names to classes. For example, "adamw" -> AdamW optimizer class.

"""Simple Registrable mixin for type-based polymorphism."""
from __future__ import annotations

from typing import Dict, Type, TypeVar

T = TypeVar("T", bound="Registrable")


class Registrable:
    """Mixin that allows classes to be registered by name and looked up later.
    
    Usage:
        class Base(Registrable):
            pass
        
        @Base.register("impl1")
        class Impl1(Base):
            pass
        
        cls = Base.by_name("impl1")
    """
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a subclass under the given name."""
        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def by_name(cls, name: str) -> Type:
        """Look up a registered subclass by name."""
        if name not in cls._registry:
            raise KeyError(f"{name} not found in registry for {cls.__name__}. "
                           f"Available: {list(cls._registry.keys())}")
        return cls._registry[name]
    
    @classmethod
    def list_available(cls) -> list:
        """List all registered names."""
        return list(cls._registry.keys())

Wait, there's a subtle bug. Since Registrable itself has _registry, all subclasses share the same registry dict unless they define their own. I should make each class that inherits from Registrable have its own registry._

Actually, the typical AllenNLP-style Registrable does handle this with a default dict per class. But for simplicity, let me just have each class that uses Registrable define its own registry.

Hmm, actually the simplest approach is to use a default dict pattern:

class Registrable:
    _registry = None  # Set per-class
    
    @classmethod
    def _get_registry(cls):
        if cls._registry is None:
            cls._registry = {}
        return cls._registry

Wait, but that means all subclasses share the class-level _registry = None. And when we access cls._registry, Python will look up the MRO. We want each direct subclass to have its own registry.

Let me use a different approach - a global mapping from class to registry:

_REGISTRIES: Dict[Type, Dict[str, Type]] = {}

class Registrable:
    @classmethod
    def register(cls, name: str):
        if cls not in _REGISTRIES:
            _REGISTRIES[cls] = {}
        def decorator(subclass):
            _REGISTRIES[cls][name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def by_name(cls, name: str) -> Type:
        if cls not in _REGISTRIES:
            raise KeyError(f"No registry for {cls.__name__}")
        if name not in _REGISTRIES[cls]:
            raise KeyError(f"{name} not in registry for {cls.__name__}")
        return _REGISTRIES[cls][name]

This is cleaner. Each class gets its own registry in the global dict.

Now let me also think about the sequential training step counter for the scheduler. Let me revise train.py accordingly.

For the pyproject.toml, let me use a simple format that supports pip install -e .:

[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "transformer-from-scratch"
version = "0.1.0"
description = "Train a Transformer language model from scratch"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "pyyaml>=6.0",
    "tqdm>=4.65",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "isort>=5.12",
    "ruff>=0.1.0",
]

[tool.black]
line-length = 100

[tool.isort]
profile = "black"
line_length = 100

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["src/transformer_from_scratch/tests"]

Wait, for the project structure, I should make sure the package is discoverable. Since source files are in src/transformer_from_scratch/, I need to configure setuptools to find the package in src/.

[tool.setuptools.packages.find]
where = ["src"]

Now for the CI workflow:

name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run checks
        run: make checks
      - name: Run tests
        run: pytest -q

For the Makefile:

.PHONY: install test style checks docker-image

install:
    pip install -e ".[dev]"

test:
    pytest -q

style:
    black src/
    isort src/

checks:
    black --check --line-length 100 src/
    ruff check src/

docker-image:
    docker build -t transformer-from-scratch .

For the Dockerfile:

FROM python:3.10-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir -e .

FROM python:3.10-slim AS release
WORKDIR /app
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /app /app
CMD ["python", "-m", "transformer_from_scratch.train"]

Wait, I'm not using requirements.txt in pyproject.toml approach. Let me use both - requirements.txt for Docker and pyproject.toml for dev.

Actually, let me just use pyproject.toml everywhere and install using pip install -e .[dev].

For Dockerfile:

FROM python:3.10-slim AS build
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

FROM python:3.10-slim AS release
WORKDIR /app
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /app /app
CMD ["python", "-m", "transformer_from_scratch.train"]

Hmm, actually for a two-stage Docker build, we should be more careful. The first stage installs everything, the second stage copies only what's needed. But for simplicity, let me just do:

FROM python:3.10-slim AS build
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

FROM python:3.10-slim AS release
WORKDIR /app
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app /app
CMD ["python", "-m", "transformer_from_scratch.train"]

Actually, copying /usr/local/bin might not exist if there are no scripts. Let me simplify:

FROM python:3.10-slim AS build
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

FROM python:3.10-slim AS release
WORKDIR /app
COPY --from=build /usr/local/lib /usr/local/lib
COPY --from=build /app /app
EXPOSE 8000
CMD ["python", "-m", "transformer_from_scratch.train"]

OK, let me not over-complicate the Dockerfile. The requirement says "two-stage skeleton" so I'll keep it simple.

Now for the tests:

test_model_forward.py:

"""Test Transformer model forward pass."""
import torch
from transformer_from_scratch.model import TransformerLM, olmo_tinydef test_olmo_tiny_forward():
    """Test that olmo_tiny produces correct output shape."""
    model = olmo_tiny()
    B, T = 2, 16
    input_ids = torch.randint(0, model.config.vocab_size, (B, T))
    logits = model(input_ids)
    assert logits.shape == (B, T, model.config.vocab_size), f"Expected {(B, T, model.config.vocab_size)}, got {logits.shape}"


def test_transformer_lm_forward():
    """Test TransformerLM forward with custom config."""
    from transformer_from_scratch.config import ModelConfig
    config = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, seq_len=32, d_ff=256, dropout=0.0)
    model = TransformerLM(config)
    B, T = 4, 32
    input_ids = torch.randint(0, 256, (B, T))
    logits = model(input_ids)
    assert logits.shape == (B, T, 256)

test_training_smoke.py:

"""Smoke test for training loop."""
import os
import sys
import torch
import tempfile
from pathlib import Path

import pytest


def test_training_smoke():
    """Run a quick training loop on synthetic data."""
    # Skip if no CPU available (shouldn't happen)
    if not torch.cuda.is_available() and os.environ.get("REQUIRE_GPU") == "1":
        pytest.skip("GPU required but not available")
    
    from transformer_from_scratch.prepare_synthetic import generate_synthetic_data
    from transformer_from_scratch.train import main as train_main
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate synthetic data
        generate_synthetic_data(output_dir=tmpdir, vocab_size=256, seq_len=33, n_train=64, n_valid=32)
        
        # Run training
        train_main([
            "--train-path", str(Path(tmpdir) / "train.npy"),
            "--valid-path", str(Path(tmpdir) / "valid.npy"),
            "--out-dir", str(Path(tmpdir) / "output"),
            "--epochs", "1",
            "--batch-size", "8",
            "--seed", "42",
            "--overrides", "model.vocab_size=256", "model.seq_len=32", "model.d_model=64", "model.n_layers=2", "model.n_heads=4", "model.d_ff=256",
        ])
        
        # Check that checkpoint was saved
        assert (Path(tmpdir) / "output" / "checkpoint.pt").exists()


def test_validation_loss_finite():
    """Test that validation loss is finite after a short training run."""
    if not torch.cuda.is_available() and os.environ.get("REQUIRE_GPU") == "1":
        pytest.skip("GPU required but not available")
    
    from transformer_from_scratch.config import Config    from transformer_from_scratch.model import TransformerLM
    from transformer_from_scratch.data import NumpyDataset, make_dataloader
    from transformer_from_scratch.prepare_synthetic import generate_synthetic_data
    from transformer_from_scratch.optim import build_adamw_optimizer
    from transformer_from_scratch.train import evaluate
    import torch.nn as nn
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_synthetic_data(output_dir=tmpdir, vocab_size=256, seq_len=33, n_train=32, n_valid=16)
        
        config = Config()
        config.model.vocab_size = 256
        config.model.seq_len = 32
        config.model.d_model = 64
        config.model.n_layers = 1
        config.model.n_heads = 4
        config.model.d_ff = 256
        
        model = TransformerLM(config.model)
        device = torch.device("cpu")
        model = model.to(device)
        
        valid_ds = NumpyDataset(Path(tmpdir) / "valid.npy")
        valid_loader = make_dataloader(valid_ds, batch_size=8, shuffle=False)
        
        valid_loss = evaluate(model, valid_loader, device)
        assert torch.isfinite(torch.tensor(valid_loss)), f"Validation loss is not finite: {valid_loss}"

Hmm, wait. For the smoke test, I also need to make sure the data shape is compatible with the model. The data generates sequences of length 33 (seq_len=33). The model config has seq_len=32 (the max sequence length for positional embeddings). But when we create input/target from the data using collate_fn (seq[:-1], seq[1:]), we get inputs of length 32 and targets of length 32. The model uses positional embeddings up to seq_len positions, so inputs of length 32 are OK if model.config.seq_len >= 32.

But wait, the overrides use "model.seq_len=32" and the data has seq_len=33. After splitting, inputs are 32 tokens long. The model's positional embedding has max_seq_len=32 (since it's nn.Embedding(config.seq_len, config.d_model)). And we pass all 32 positions (0 to 31). That should work.

Actually, there's a nuance. The override "model.seq_len=32" sets config.model.seq_len = 32. But config.model.seq_len was originally 64 (from ModelConfig default). When we pass overrides, the values get string parsed. Let me check the merge_dotlist function...

In merge_dotlist:

key, value = item.split("=", 1)
try:
    value = int(value)
except ValueError:
    try:
        value = float(value)
    except ValueError:
        pass
_set_nested(d, key, value)

So "model.seq_len=32" would set d["model"]["seq_len"] = 32 (an int). Then dict_to_dataclass would create ModelConfig(seq_len=32). Good.

But wait, I'm passing these overrides along with CLI args. The train_main function first creates Config(), then applies overrides, then applies CLI overrides. The CLI args like --epochs, --batch-size, etc. are applied after. That should be fine.

Actually, I realize there's an issue with my train.py. The train_main function parses args but doesn't pass them to the scheduler correctly. Let me revise.

Also, I need to make sure tests work on CPU and are quick. The smoke test needs to complete in under a minute on typical CI.

Let me also think about the __init__.py:

"""Transformer from scratch - training a Transformer LM from scratch."""
from .model import TransformerLM, olmo_tiny
from .config import Config, ModelConfig, TrainConfig, DataConfig

__all__ = ["TransformerLM", "olmo_tiny", "Config", "ModelConfig", "TrainConfig", "DataConfig"]

And tests/init.py can be empty.

Now let me finalize everything and write it all out.

One more thing - for the prepare_synthetic.py, I need to make sure the seq_len parameter matches. The data stores sequences of length seq_len+1 so that we can split into input (first seq_len tokens) and target (last seq_len tokens). Wait, let me recheck.

In collate_fn:

sequences = torch.stack(batch)
return sequences[:, :-1], sequences[:, 1:]

So if sequences have length L, then:

    input has length L-1
    target has length L-1

If model.seq_len is 32 (the model expects sequences of length up to 32 for positions), then we need L-1 <= 32, so L <= 33.

In prepare_synthetic.py, I default seq_len=65 (since default model.seq_len=64, sequences of 65 tokens give input/target of length 64). But in the smoke test, I use seq_len=33 and override model.seq_len=32. That works.

Actually wait, I named the parameter seq_len in prepare_synthetic.py but it should really be the total sequence length (which is model_seq_len + 1). Let me rename it to be clear, or just add a comment.

Let me also double-check the NumpyDataset - it reads 2D arrays where each row is a sequence. The data has shape (n_sequences, seq_len_total) where seq_len_total is model_seq_len + 1. So for default model (64), the data would be (n_train, 65).

OK I think I've thought through everything. Let me write the final files.

Actually, one more concern: in the config.py merge_dotlist, when we parse "model.d_ff=256", it should set config.model.d_ff = 256. Let me make sure dict_to_dataclass handles this correctly.

In the dict representation:

{
    "model": {
        "vocab_size": 256,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "seq_len": 32,
        "d_ff": 256,
        "dropout": 0.1
    },
    ...
}

And then _dict_to_dataclass(Config, d) would need to convert d["model"] back to a ModelConfig. My function uses _NESTED_CLASSES dict to know which types to convert. Let me make sure this works.

In my config.py, I'll have:

_NESTED_TYPES = {
    "model": ModelConfig,
    "data": DataConfig,
    "train": TrainConfig,
}

def _dict_to_dataclass(cls, d):
    if cls is Config:
        kwargs = {}
        for k, v in d.items():
            if k in _NESTED_TYPES and isinstance(v, dict):
                kwargs[k] = _dict_to_dataclass(_NESTED_TYPES[k], v)
            else:
                kwargs[k] = v
        return cls(**kwargs)
    else:
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})

That should work. Let me now write everything out.

Let me finalize all files and write them out completely. I need to be very careful with all the details.

Let me reconsider the test_training_smoke.py. The test should:

    Auto-generate synthetic data if missing
    Run train.py for 1 epoch on CPU
    Assert training loop runs and validation loss is finite
    Complete quickly

I'll use tempfile for isolation.

Let me also make sure the scheduler step method actually modifies the optimizer's learning rate. In the current design, CosineWithWarmupScheduler.step(step) computes the scale and updates all param groups. But in my train.py, I'm calling scheduler.step(global_step) inside the training loop. That should work.

Actually, I realized there's a subtle issue. In train_one_epoch, I initialize total_steps = 0 and increment per batch. But this is per-epoch, not global. Let me pass global_step as a parameter.

Let me rewrite the train loop more carefully:

def main(args=None):
    # ...setup...
    
    global_step = 0
    for epoch in range(config.train.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            if config.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            
            global_step += 1
            scheduler.step(global_step)
            
            total_loss += loss.item()
            num_batches += 1
        
        train_loss = total_loss / max(1, num_batches)
        valid_loss = evaluate(model, valid_loader, device)
        
        # Save checkpoint
        ...

Actually, this is getting more inline. Let me keep the helper functions but fix the global_step tracking.

OK, I think I've planned enough. Let me write all the final files now. I'll make the code concise, correct, and well-documented.

Let me also think about the train.py structure. Since users might want to call it from CLI, I should make sure python -m transformer_from_scratch.train works.

For the __main__ block, I'll have:

if __name__ == "__main__":
    main()

And the test can call train_main([...]) directly.

Let me finalize:

Actually, I want to reconsider the test structure. For test_training_smoke.py, I should test that training completes and checkpoint is saved. Let me make it simple and direct.

One more consideration: for Python 3.9 compatibility, I should use from __future__ import annotations and avoid str | Path syntax (use Union or Optional instead in type hints). Actually, from __future__ import annotations makes annotations strings so they're not evaluated, which means str | Path in type hints is fine because it's not evaluated at runtime.

Wait, but list[str] in Python 3.9 without from __future__ import annotations would fail. With it, it's fine because annotations are strings. Let me make sure to use from __future__ import annotations everywhere.

OK, let me now write everything out. I'll be very thorough.

For the .pre-commit-config.yaml:

repos:
  - repo: https://github.com/psf/black
    rev: "24.4.2"
    hooks:
      - id: black
        args: [--line-length=100]
  - repo: https://github.com/PyCQA/isort
    rev: "5.13.2"
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: "v0.4.8"
    hooks:
      - id: ruff
        args: [--line-length=100]

Wait, I should check recent versions. Let me just use reasonable versions since this is a new repo being created in May 2026. Let me use slightly forward versions.

Actually, since the current date is May 2026, I should use versions that would be available by then. Let me use versions I know exist: black 24.x, isort 5.x, ruff 0.5+. These should all be fine.

Now let me finalize everything. Here's my complete file list:

    README.md
    LICENSE
    pyproject.toml
    Makefile
    .pre-commit-config.yaml
    .github/workflows/main.yml
    Dockerfile
    src/transformer_from_scratch/init.py
    src/transformer_from_scratch/config.py
    src/transformer_from_scratch/registrable.py
    src/transformer_from_scratch/attention.py
    src/transformer_from_scratch/model.py
    src/transformer_from_scratch/data.py
    src/transformer_from_scratch/prepare_synthetic.py
    src/transformer_from_scratch/train.py
    src/transformer_from_scratch/optim.py
    src/transformer_from_scratch/checkpoint.py
    src/transformer_from_scratch/utils.py
    src/transformer_from_scratch/tests/init.py
    src/transformer_from_scratch/tests/test_model_forward.py
    src/transformer_from_scratch/tests/test_training_smoke.py
    src/transformer_from_scratch/tests/test_config.py

Let me write each file now, carefully.

For train.py, I need to handle the scheduler properly. Let me use a simpler approach where the scheduler is integrated into the training loop directly.

I also need to handle the case where tqdm might not be installed during test runs. Actually, tqdm is in the dependencies so it should be fine.

Let me think about the scheduler iteration more carefully. I'll use a global step counter that's passed around.

Actually, the simplest approach for the training loop: just put everything inline in main() instead of splitting into separate functions that need to track global state. But I want to keep it modular for readability. Let me have train_one_epoch take and return global_step.

OK, here's my final approach for train.py:

def train_one_epoch(model, train_loader, optimizer, scheduler, global_step, max_grad_norm, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        global_step += 1
        if scheduler is not None:
            scheduler.step(global_step)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(1, num_batches), global_step

This is clean. Global step is passed in and returned.

Now let me write all the final code. I want it to be correct and complete.

One thing I want to double-check: the NumpyDataset creates sequences where the collate_fn splits into input=seq[:-1] and target=seq[1:]. So each data row has length model_seq_len + 1 (e.g., 65 for default model). The input from the model sees model_seq_len tokens (e.g., 64), and the target is model_seq_len tokens (shifted by 1).

The model's positional embedding can handle up to config.seq_len positions. So if config.seq_len = 64, then inputs can be up to 64 tokens long. This matches.

For the causal mask, I create it in the forward pass:

causal_mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

This creates a (1, 1, T, T) lower triangular mask where 1 means "attend" and 0 means "mask out". In the scaled_dot_product_attention, I use:

if mask is not None:
    scores = scores.masked_fill(mask == 0, float("-inf"))

This should correctly mask future positions.

Wait, there's a potential issue with the shape. The mask is (1, 1, T, T) and scores are (B, H, T, T). Broadcasting should work fine since the mask will be broadcast across B and H dimensions.

Let me verify the attention computation once more:

# In MultiHeadAttention.forward()
B, T, C = x.size()
q = self.wq(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
k = self.wk(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
v = self.wv(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)

out = scaled_dot_product_attention(q, k, v, mask=mask)  # (B, H, T, d_k)

out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
return self.resid_dropout(self.wo(out))

And in scaled_dot_product_attention:

scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, T, T)
if mask is not None:
    scores = scores.masked_fill(mask == 0, float("-inf"))
attn_weights = torch.softmax(scores, dim=-1)
return torch.matmul(attn_weights, value)

This looks correct. The causal mask ensures that position i can only attend to positions 0..i.

However, there's a potential numerical issue: when we apply softmax to a row where all entries are -inf (which can happen for the first position if we're using a causal mask incorrectly), we'd get NaN. But with a lower triangular mask, position 0 can see position 0, so at least one value is always non-neg-inf. This should be fine.

Actually wait, let me also check: do I need to handle the case where the padding would cause issues? No, because in this simple LM setting, there's no padding. All sequences are the same length.

OK, I'm confident the code is correct. Let me write everything out now.

One last thing - I need to think about the checkpoint loading when resuming. The train.py should support --resume flag to load a checkpoint. I've included this.

Let me also think about edge cases in the config system. When loading from YAML, if only some fields are specified, the rest should use defaults. My _dict_to_dataclass handles this by only passing keys that are in the dict, and dataclass fields have defaults._

Actually, wait. In _dict_to_dataclass, I only pass keys that exist in d. But what about keys that are not in d? They should use the dataclass defaults. This works correctly because dataclass fields have defaults._

But what about nested dataclasses? If the YAML has:

model:
  vocab_size: 2048

Then _dict_to_dataclass(Config, d) would call Config(model=ModelConfig(vocab_size=2048), ...). But d only has model, so we'd call _dict_to_dataclass(ModelConfig, {"vocab_size": 2048}), which would call ModelConfig(vocab_size=2048). But ModelConfig has other required fields with defaults, so this would use those defaults. Good.

Actually wait, I need to check: if the YAML has other top-level keys mixed in, would that cause problems? In my _dict_to_dataclass(Config, d), I iterate over d.items() and for keys that match _NESTED_TYPES, I recursively convert. For keys that don't match, I pass them directly to Config(), which would fail if Config doesn't have those kwargs. I should filter to only known fields.

Hmm, actually Config only has model, data, and train as fields. So any extra keys in the YAML would cause a TypeError. Let me add filtering:

def _dict_to_dataclass(cls, d):
    if cls is Config:
        known_fields = _NESTED_TYPES        kwargs = {}
        for k, v in d.items():
            if k in known_fields and isinstance(v, dict):
                kwargs[k] = _dict_to_dataclass(known_fields[k], v)
            elif k in known_fields:
                kwargs[k] = v return cls(**kwargs)
    else:
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

Wait, I also need to handle the case where a field in Config might not be a dataclass. Like what if someone has a top-level field? Actually, Config only has three fields: model, data, train. All are dataclasses. So this should work.

But what if the YAML only specifies model.vocab_size and not the rest of model? Then d["model"] would be {"vocab_size": 2048}, and we'd call ModelConfig(vocab_size=2048), which uses defaults for everything else. That works.

What if the YAML specifies model.vocab_size as a string (which YAML might do)? Then merge_dotlist would parse it... Actually, merge_dotlist is for CLI overrides with dot notation. YAML loading just uses yaml.safe_load, which should parse types correctly for numbers.

OK, I think I've covered all the edge cases. Let me write the final code now.

Actually, one more thing about the test. In test_training_smoke.py, I need to be careful about the overrides. The --overrides argument takes a list of strings. In my train.py, I parse args.overrides and pass them to merge_dotlist. Let me make sure this works correctly.

In argparse:

parser.add_argument("--overrides", nargs="*", default=[], help="Dotlist overrides")

When called as:

train_main(["--overrides", "model.vocab_size=256", "model.seq_len=32", ...])

This should set args.overrides = ["model.vocab_size=256", "model.seq_len=32", ...].

Then in merge_dotlist(config, args.overrides), each override is parsed like "model.vocab_size=256" -> key="model.vocab_size", value=256.

The function converts config to dict, applies overrides, then converts back. This should work.

Let me also think about whether the train.py needs to be runnable as python -m transformer_from_scratch.train. For this to work, I need the if __name__ == "__main__" block. I have that.

OK, I'm now going to write all the files. Let me be very careful.

Let me also double-check: the requirement says "seq_len" in the model config, which is the maximum sequence length for positional embeddings. The data sequences are of length seq_len+1 (since we split into input and target of length seq_len each). This is handled correctly.

For the prepare_synthetic.py script, I'll default seq_len=65 but explain that this is model_seq_len + 1. Actually, let me rename the parameter in the function to be clearer. I'll call it seq_len but document that it should be model_seq_len + 1.

Wait, I think it's cleaner to have the data generation accept model_seq_len and internally add 1. Let me do that:

def generate_synthetic_data(
    output_dir=".",
    vocab_size=1024,
    model_seq_len=64,
    ...
):
    seq_len = model_seq_len + 1  # +1 for input/target split

This is clearer. Let me use this approach.

Actually, looking at it again, the requirement says "NumpyDataset reads .npy and yields token sequences as torch.LongTensor" and "DataLoader wrapper with batch collate (input=seq[:-1], target=seq[1:])". So the data stores full sequences, and the collate function creates input/target pairs. This is standard.

OK, I'm going to write all the files now. Let me be thorough and careful.

Let me also think about one thing: in the train.py main function, when I call train_main, how do I handle the imports? Since the tests are inside the package, I can use relative imports. But train_main is called from the test, so the module needs to be importable.

The test does:

from transformer_from_scratch.train import main as train_main

And train.py uses relative imports like:

from .config import Config, load_from_yaml, merge_dotlist

This should work when the package is installed (via pip install -e .).

OK, let me finalize everything. I'll write each file completely.

Actually, I want to double-check one thing. When I use from __future__ import annotations, type hints like str | Path become strings that aren't evaluated, so they work in Python 3.9+. But list[str] type hints also work. Let me just use from __future__ import annotations in every file for safety.

Alright, let me now write everything. I'll organize this carefully.

For the README, I need to include:

    Quickstart: install, generate synthetic data, run training, run tests
    How to change model/data sizes

For the LICENSE, standard MIT license.

Let me start outputting all the files now.

One more consideration: pytest.ini or pyproject.toml test configuration. I've included tool.pytest.ini_options in pyproject.toml which sets testpaths. Good.

Also, for the checks target in the Makefile, I need:

checks:
    black --check --line-length 100 src/
    ruff check src/

This could fail CI if there are style issues. That's fine per the requirements.

Let me also make sure the test for config works. I need:

    test_config.py: verify YAML load and dotlist override merge behavior

def test_yaml_load():
    # Create a temp YAML, load it, verify valuesdef test_dotlist_override():
    config = Config()
    config2 = merge_dotlist(config, ["model.vocab_size=2048", "train.lr=1e-3"])
    assert config2.model.vocab_size == 2048
    assert config2.train.lr == 1e-3
    
def test_partial_yaml():
    # Create YAML with partial model config, verify defaults

OK let me write everything out now. I'm going to be very careful about correctness.

Actually, let me also consider: should I put requirements.txt alongside pyproject.toml? The Dockerfile might use it. Let me add a simple requirements.txt generated from pyproject.toml dependencies. Actually, the requirement says "pyproject.toml (or requirements.txt + minimal setup)". I'll use pyproject.toml as the primary and generate requirements.txt from it for Docker.

Actually, for simplicity, let me just have pyproject.toml and install with pip install -e . in the Dockerfile.

Wait, the Dockerfile needs to install dependencies. If I use pip install ., it will install from pyproject.toml. But for Docker layer caching, it's good to copy requirements first and install them before copying the rest of the code. But since I only have pyproject.toml, I'll copy that first.

Let me structure the Dockerfile properly:

FROM python:3.10-slim AS build
WORKDIR /app
COPY pyproject.toml .
# Install dependencies (need src dir for setup but we'll copy later)
# Create a dummy package for initial install
RUN mkdir -p src/transformer_from_scratch && \
    touch src/transformer_from_scratch/__init__.py && \
    pip install --no-cache-dir .

# Copy actual source code
COPY src/ src/
RUN pip install --no-cache-dir --force-reinstall --no-deps .

Hmm, this is getting complicated for a "skeleton" Dockerfile. Let me just keep it simple:

FROM python:3.10-slim AS build
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .

FROM python:3.10-slim AS release
WORKDIR /app
COPY --from=build /usr/local/lib /usr/local/lib
COPY --from=build /app /app
CMD ["python", "-m", "transformer_from_scratch.train"]

This is a two-stage build that's simple and works. The first stage installs everything, the second copies what's needed.

Actually, let me also add --no-cache-dir and use a virtual environment or pip in a way that's Docker-friendly. But for a skeleton, this is fine.

OK I'm now going to write everything out. Let me start.

Wait, one more thing. The src/transformer_from_scratch/tests/ directory needs an __init__.py even if it's empty, for pytest to discover the tests.

Also, I want to make sure pytest can find the tests. The pyproject.toml has:

[tool.pytest.ini_options]
testpaths = ["src/transformer_from_scratch/tests"]

This should work.

And for the CI, I need to install the package first before running tests:

- name: Install dependencies
  run: pip install -e ".[dev]"
- name: Run checks
  run: make checks
- name: Run tests
  run: pytest -q

Actually, I should run make install or pip install -e ".[dev]" in the CI. Let me use pip install -e ".[dev]" directly since the Makefile install target might also run this.

OK, one final thing. In the Makefile checks target, I use black --check and ruff check. These might modify the format or find lint issues. Since the code should be formatted correctly, these should pass. But the requirement says "can allow ruff to exit non-zero in CI if issues found" - meaning it's OK if these fail in CI initially. I'll just run them.

Let me now actually write everything.

Let me think about whether I need to handle the edge case in data where NumpyDataset.__init__ receives a numpy array directly. I'll add that capability.

Also, I should make sure the __init__.py exports are correct.

OK, starting the output now. This is going to be long but thorough.

For the main train.py, I want to also support running as a module. I'll add if __name__ == "__main__": main().

Let me also think about the logging format. I'll use a simple format.

Alright, here we go. Writing all files now.

One more thing: should I handle the case where --train-path or --valid-path doesn't exist? The NumpyDataset will fail with a FileNotFoundError if the file doesn't exist. I think that's fine for now - no need to add extra error handling for a training script.

And the test_training_smoke should:

    Create synthetic data in a temp directory
    Run training
    Check that checkpoints exist
    Check that validation loss is finite

I'll write two separate test functions in test_training_smoke.py.

OK, let me also make sure the CosineWithWarmupScheduler works correctly. The warmup starts at 0 and linearly increases to base_lr, then cosines down to min_lr_ratio * base_lr.*

Actually wait, in my implementation:

def step(self, step):
    if step < self.warmup_steps:
        scale = step / max(1, self.warmup_steps)
    else:
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
        group["lr"] = base_lr * scale
    
    return self.optimizer.param_groups[0]["lr"]

At step=0: scale = 0 (lr = 0). Hmm, that's not great. Usually warmup goes from a small lr to the base lr. Let me adjust:

At step=0, we'd have lr = 0. That might cause issues. Let me start warmup from a small value:

Actually, at step 0, we don't call scheduler.step usually. The optimizer starts at the base lr. Then step 1 would be scale = 1/warmup_steps. So at step warmup_steps, scale = 1 (full lr). Then cosine decay follows.

But wait, when step=0, the initial lr of the optimizer is base_lr. Then when we call scheduler.step(1), the lr becomes base_lr * (1/warmup_steps). This means the lr drops from base_lr to base_lr/warmup_steps immediately. That's not ideal.*

A better approach: don't call scheduler.step(0). Start from step=1. Or set the initial lr in the optimizer to0 and call scheduler.step(1) before the first optimizer step.

Actually, the common approach is to call scheduler.step(global_step) after each optimizer step, and global_step starts at 0 and increments. So:

    Before training: optimizer lr = base_lr
    After first batch: global_step = 1, scheduler.step(1)

via
deepinfra

is everything done
Model: zai-org/GLM-5.1
End to end project
