# Transformer From Scratch

<div align="center">
  <h1>Transformer From Scratch</h1>
  <h4>A compact, production-minded PyTorch language model stack built from first principles</h4>
</div>

<p align="center">
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="./pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-supported-ee4c2c.svg"></a>
  <a href="https://github.com/preet7777777/transformer/actions"><img alt="CI" src="https://img.shields.io/badge/ci-github%20actions-2088FF.svg"></a>
</p>

## Why this repo stands out

This project is intentionally small, but it is wired like a real ML codebase:

- clean packaging with `pyproject.toml`
- module-based CLI entry points
- YAML and dotlist configuration overrides
- checkpointing and resume support
- a streaming-friendly `.npy` shard API
- smoke tests that validate the full training loop
- a benchmark command for quick throughput checks

## What is included

- dataclass configuration with YAML loading
- scaled dot-product attention and multi-head attention
- a Transformer language model with learned positional embeddings
- tied input/output embeddings
- `.npy` dataset loading and shard concatenation
- synthetic data generation for fast local experiments
- training, validation, checkpointing, and resume support
- a benchmark CLI that reports tokens/sec and batch latency

## Installation

Install for development:

```bash
pip install -e ".[dev]"
```

Install runtime only:

```bash
pip install -e .
```

## Quick start

### 1. Generate synthetic data

```bash
tfs-generate --output-dir data --model-seq-len 32
```

This creates:

- `data/train.npy`
- `data/valid.npy`

### 2. Train

```bash
tfs-train \
  --train-path data/train.npy \
  --valid-path data/valid.npy \
  --out-dir runs/demo
```

You can also use the module form:

```bash
python -m transformer_from_scratch.train \
  --train-path data/train.npy \
  --valid-path data/valid.npy \
  --out-dir runs/demo
```

### 3. Benchmark

```bash
tfs-benchmark --seq-len 32 --batch-size 8 --iterations 100
```

The benchmark prints JSON with:

- parameter count
- batch latency
- tokens per second
- device information

### 4. Resume from a checkpoint

```bash
python -m transformer_from_scratch.train \
  --train-path data/train.npy \
  --valid-path data/valid.npy \
  --out-dir runs/demo \
  --resume runs/demo/checkpoint.pt
```

## Configuration

The training CLI supports both explicit flags and dotlist overrides:

```bash
python -m transformer_from_scratch.train \
  --train-path data/train.npy \
  --valid-path data/valid.npy \
  --out-dir runs/demo \
  --overrides model.n_layers=4 model.dropout=0.0 train.batch_size=16
```

## Project structure

- [src/transformer_from_scratch/config.py](src/transformer_from_scratch/config.py) — structured configuration objects
- [src/transformer_from_scratch/attention.py](src/transformer_from_scratch/attention.py) — attention kernels and causal masking
- [src/transformer_from_scratch/model.py](src/transformer_from_scratch/model.py) — Transformer language model implementation
- [src/transformer_from_scratch/data.py](src/transformer_from_scratch/data.py) — dataset and dataloader helpers
- [src/transformer_from_scratch/prepare_synthetic.py](src/transformer_from_scratch/prepare_synthetic.py) — synthetic data generation
- [src/transformer_from_scratch/train.py](src/transformer_from_scratch/train.py) — training CLI
- [src/transformer_from_scratch/benchmark.py](src/transformer_from_scratch/benchmark.py) — throughput and latency benchmarking
- [src/transformer_from_scratch/optim.py](src/transformer_from_scratch/optim.py) — optimizer and scheduler helpers
- [src/transformer_from_scratch/checkpoint.py](src/transformer_from_scratch/checkpoint.py) — checkpoint save/load helpers
- [src/transformer_from_scratch/tests](src/transformer_from_scratch/tests) — unit and smoke tests

## Development

Run tests:

```bash
pytest -q
```

Run checks:

```bash
make checks
```

Format the codebase:

```bash
make style
```

Run the benchmark:

```bash
make benchmark
```

## Design notes

The dataset stores full token sequences, and the collate function shifts them into language-model pairs:

- input = `seq[:-1]`
- target = `seq[1:]`

This keeps the implementation simple while preserving standard autoregressive training behavior.

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- PyYAML
- tqdm
- pytest

## License

Released under the MIT License. See [LICENSE](./LICENSE) for details.
