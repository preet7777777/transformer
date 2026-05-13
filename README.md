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

## v1 — Inference API + Web Demo

v1 ships a production-grade inference layer on top of the existing training stack:

- **REST API** (`tfs-serve`) — FastAPI server with `/generate`, `/score`, `/health`, and `/metrics`
- **Web demo** (`tfs-web`) — single-page UI for interactive token generation
- **Checkpoint registry** — file-backed version registry with auto-scan of `runs/`
- **Monitoring** — per-request latency (p50/p95), throughput, error rate, and RSS memory
- **Rate limiting** — sliding-window 60 req/min per IP, configurable via env vars
- **Request timeouts** — 30 s default, configurable via `TFS_REQUEST_TIMEOUT`

### Quick start: API server

```bash
pip install -e ".[api]"

# Register a checkpoint
python -c "
from transformer_from_scratch.registry import CheckpointRegistry
from transformer_from_scratch.config import ModelConfig
r = CheckpointRegistry('./registry')
r.register('v1', './runs/demo/best.pt', ModelConfig(), description='my first model')
"

# Start the API (auto-scans ./runs/ for checkpoints)
tfs-serve --registry-dir ./registry --runs-dir ./runs

# Open the web demo (points to http://localhost:8000 by default)
tfs-web
```

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — device, loaded models, uptime |
| `GET` | `/models` | List all registered versions |
| `POST` | `/generate` | Autoregressive token generation |
| `POST` | `/score` | Per-token log-probs + perplexity |
| `GET` | `/metrics` | Plain-text counters (requests, latency, tokens, memory) |

**Generate example:**

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"model_version":"v1","prompt":[0,1,2],"max_new_tokens":8,"temperature":0.8}' | python -m json.tool
```

**Score example:**

```bash
curl -s -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"model_version":"v1","tokens":[0,1,2,3,4]}' | python -m json.tool
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TFS_MAX_TOKENS` | `512` | Max `max_new_tokens` per request |
| `TFS_MAX_PROMPT_LEN` | `256` | Max prompt length |
| `TFS_REQUEST_TIMEOUT` | `30.0` | Per-request timeout in seconds |

---

## Why this repo stands out

This project is intentionally small, but it is wired like a real ML codebase:

- clean packaging with pyproject.toml
- module-based CLI entry points
- YAML and dotlist configuration overrides
- checkpointing and resume support
- a streaming-friendly .npy shard API
- smoke tests that validate the full training loop
- a benchmark command for quick throughput checks
- an end-to-end demo that generates data, trains, benchmarks, and writes a report
- rotary positional embeddings for longer-context extrapolation
- a public Tiny Shakespeare evaluation with baseline comparison
- a larger public WikiText-2 benchmark with n-gram baselines and ablations

## What is included

- dataclass configuration with YAML loading
- scaled dot-product attention and multi-head attention
- a Transformer language model with learned positional embeddings
- tied input/output embeddings
- .npy dataset loading and shard concatenation
- synthetic data generation for fast local experiments
- training, validation, checkpointing, and resume support
- a benchmark CLI that reports tokens/sec and batch latency
- a demo CLI that creates a reproducible showcase artifact
- rotary embeddings as an advanced positional encoding option
- a public evaluation CLI for Tiny Shakespeare
- a public benchmark CLI for WikiText-2 with learned-vs-RoPE and tied-vs-untied ablations

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

- data/train.npy
- data/valid.npy

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

### 4. Run the end-to-end demo

```bash
tfs-demo --output-dir runs/showcase
```

For a stronger demo result, use a learnable synthetic pattern:

```bash
tfs-demo --output-dir runs/showcase --mode copy --epochs 30
```

For a more research-style architecture, try rotary embeddings:

```bash
tfs-demo --output-dir runs/rope-showcase --positional-encoding rope --mode copy --epochs 30
```

RoPE lets the model run on sequence lengths beyond the learned-context limit, which is useful for
long-context extrapolation experiments.

This command will:

- generate fresh synthetic data
- train the model
- load the best checkpoint
- evaluate validation loss
- benchmark throughput
- write a report.json file

### 5. Resume from a checkpoint

```bash
python -m transformer_from_scratch.train \
  --train-path data/train.npy \
  --valid-path data/valid.npy \
  --out-dir runs/demo \
  --resume runs/demo/checkpoint.pt
```

## Latest demo result

The repository includes a real end-to-end CPU demo run captured in [RESULTS.md](RESULTS.md).

Highlights from the latest run:

- final training loss: 1.3406
- best validation loss: 2.2453
- benchmark throughput: 433,144.12 tokens/sec
- mean batch latency: 0.2955 ms

The raw demo artifacts live under runs/showcase-copy-30.

## Public dataset result

The repository also includes an external validation run on Tiny Shakespeare, a classic public
character-language-modeling dataset from [karpathy/char-rnn](https://github.com/karpathy/char-rnn).

Highlights from the latest public run:

- unigram baseline loss: 3.3473
- bigram baseline loss: 2.4819
- model validation loss: 2.0040
- unigram baseline perplexity: 28.4260
- bigram baseline perplexity: 11.9638
- model perplexity: 7.4190
- benchmark throughput: 143,273.09 tokens/sec
- mean batch latency: 28.5888 ms

The raw public artifacts live under [runs/public-eval](runs/public-eval), and the full summary is
captured in [RESULTS_PUBLIC.md](RESULTS_PUBLIC.md).

## Public benchmark and ablation suite

The repository now includes a larger public benchmark on WikiText-2 from
[pytorch/examples](https://github.com/pytorch/examples), plus a small ablation sweep.

This run compares:

- unigram and bigram baselines
- RoPE with tied embeddings
- learned embeddings with tied embeddings
- RoPE with untied embeddings

It also benchmarks the RoPE model at a longer context length to show the throughput profile on a
more production-like setting.

The latest benchmark summary is captured in [RESULTS_BENCHMARK.md](RESULTS_BENCHMARK.md).

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
- [src/transformer_from_scratch/demo.py](src/transformer_from_scratch/demo.py) — end-to-end showcase pipeline
- [src/transformer_from_scratch/public_eval.py](src/transformer_from_scratch/public_eval.py) — public dataset validation and baseline comparison
- [src/transformer_from_scratch/public_benchmark.py](src/transformer_from_scratch/public_benchmark.py) — public benchmark and ablation suite
- [src/transformer_from_scratch/public_data.py](src/transformer_from_scratch/public_data.py) — public text download and tokenization helpers
- [src/transformer_from_scratch/baselines.py](src/transformer_from_scratch/baselines.py) — unigram and bigram baselines
- [src/transformer_from_scratch/generation.py](src/transformer_from_scratch/generation.py) — autoregressive sampling helpers
- [src/transformer_from_scratch/optim.py](src/transformer_from_scratch/optim.py) — optimizer and scheduler helpers
- [src/transformer_from_scratch/checkpoint.py](src/transformer_from_scratch/checkpoint.py) — checkpoint save/load helpers
- [src/transformer_from_scratch/registry.py](src/transformer_from_scratch/registry.py) — checkpoint registry and model versioning
- [src/transformer_from_scratch/api.py](src/transformer_from_scratch/api.py) — FastAPI inference server (v1)
- [src/transformer_from_scratch/webapp.py](src/transformer_from_scratch/webapp.py) — web demo UI (v1)
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

Run the public dataset evaluation:

```bash
make public-eval
```

Run the public benchmark and ablations:

```bash
make public-benchmark
```

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md) and [ROADMAP.md](ROADMAP.md).
See [OPEN_SOURCE.md](OPEN_SOURCE.md) for branch protection and collaboration guidance, and [SECURITY.md](SECURITY.md) for reporting security issues.

If you want to help but do not know where to start, good first contributions include:

- documentation improvements
- extra tests
- small bug fixes
- benchmark or evaluation polish
- new baselines or ablations

## Design notes

The dataset stores full token sequences, and the collate function shifts them into language-model pairs:

- input = seq[:-1]
- target = seq[1:]

This keeps the implementation simple while preserving standard autoregressive training behavior.

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- PyYAML
- tqdm

**API / web demo (optional):**
- fastapi ≥ 0.110
- uvicorn[standard] ≥ 0.29
- pydantic ≥ 2.0
- psutil ≥ 5.9

Install with: `pip install -e ".[api]"`

## License

Released under the MIT License. See [LICENSE](./LICENSE) for details.