# Transformer From Scratch

<div align="center">
  <h1>Transformer From Scratch</h1>
  <h4>A clean, production-oriented PyTorch language model project built from first principles</h4>
</div>

<p align="center">
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-supported-ee4c2c.svg"></a>
  <a href="./pyproject.toml"><img alt="Packaging" src="https://img.shields.io/badge/packaging-pyproject.toml-8A2BE2.svg"></a>
</p>

## Overview

This repository contains the building blocks for a compact Transformer language model and the supporting training stack:

- dataclass-based configuration with YAML and dotlist overrides
- scaled dot-product attention and multi-head self-attention
- a small Transformer language model with tied embeddings
- `.npy` dataset loading and a streaming-friendly shard API
- synthetic data generation for quick smoke tests
- training, validation, checkpointing, and resume support
- a minimal test suite for model and training validation

## Installation

Install the package in editable mode for development:

```bash
pip install -e ".[dev]"
```

If you only want the runtime package:

```bash
pip install -e .
```

## Quick start

### 1. Generate synthetic data

```bash
python -m transformer_from_scratch.prepare_synthetic \
  --output-dir data \
  --model-seq-len 32
```

This creates:

- data/train.npy
- data/valid.npy

### 2. Train the model

```bash
python -m transformer_from_scratch.train \
  --train-path data/train.npy \
  --valid-path data/valid.npy \
  --out-dir runs/demo
```

Common overrides:

```bash
python -m transformer_from_scratch.train \
  --train-path data/train.npy \
  --valid-path data/valid.npy \
  --out-dir runs/demo \
  --epochs 3 \
  --batch-size 32 \
  --lr 1e-3 \
  --seq-len 32 \
  --d-model 128 \
  --n-layers 4 \
  --n-heads 8 \
  --d-ff 512
```

You can also override nested config fields with dotlist syntax:

```bash
python -m transformer_from_scratch.train \
  --train-path data/train.npy \
  --valid-path data/valid.npy \
  --out-dir runs/demo \
  --overrides model.n_layers=4 model.dropout=0.0 train.batch_size=16
```

### 3. Resume from a checkpoint

```bash
python -m transformer_from_scratch.train \
  --train-path data/train.npy \
  --valid-path data/valid.npy \
  --out-dir runs/demo \
  --resume runs/demo/checkpoint.pt
```

## Project structure

- [src/transformer_from_scratch/config.py](src/transformer_from_scratch/config.py) — structured configuration objects
- [src/transformer_from_scratch/attention.py](src/transformer_from_scratch/attention.py) — attention kernels and causal masking
- [src/transformer_from_scratch/model.py](src/transformer_from_scratch/model.py) — Transformer language model implementation
- [src/transformer_from_scratch/data.py](src/transformer_from_scratch/data.py) — dataset and dataloader helpers
- [src/transformer_from_scratch/prepare_synthetic.py](src/transformer_from_scratch/prepare_synthetic.py) — synthetic data generation
- [src/transformer_from_scratch/train.py](src/transformer_from_scratch/train.py) — training CLI
- [src/transformer_from_scratch/optim.py](src/transformer_from_scratch/optim.py) — optimizer and scheduler helpers
- [src/transformer_from_scratch/checkpoint.py](src/transformer_from_scratch/checkpoint.py) — checkpoint save/load helpers
- [src/transformer_from_scratch/tests](src/transformer_from_scratch/tests) — unit and smoke tests

## Testing

Run the full test suite:

```bash
pytest -q
```

Run formatting and lint checks:

```bash
make checks
```

Format the codebase:

```bash
make style
```

## Development notes

The dataset stores complete token sequences, and the collate function shifts them into language-model pairs:

- input = `seq[:-1]`
- target = `seq[1:]`

This keeps the implementation simple while still supporting standard autoregressive training.

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- PyYAML
- tqdm
- pytest

## License

Released under the MIT License. See [LICENSE](./LICENSE) for details.
