# Implementation Guide

This document is for any agent or contributor working on this repository.
It explains the project structure, how to run it, and how to make safe changes.

## Project summary

This repository is a compact Transformer language model implementation in PyTorch.
It includes:

- model code built from scratch
- synthetic data generation
- public dataset evaluation on Tiny Shakespeare
- a public benchmark and ablation suite on WikiText-2
- training, benchmarking, generation, checkpointing, and reporting
- tests and packaging for reproducible use

## Main goals

When working on this project, prefer changes that improve one or more of these:

1. correctness
2. reproducibility
3. documentation clarity
4. experimental credibility
5. benchmark quality
6. user experience for training and evaluation

## Repository layout

Key files and folders:

- [src/transformer_from_scratch/config.py](src/transformer_from_scratch/config.py) — configuration dataclasses and YAML helpers
- [src/transformer_from_scratch/model.py](src/transformer_from_scratch/model.py) — Transformer language model
- [src/transformer_from_scratch/attention.py](src/transformer_from_scratch/attention.py) — attention and RoPE helpers
- [src/transformer_from_scratch/data.py](src/transformer_from_scratch/data.py) — dataset and dataloader helpers
- [src/transformer_from_scratch/public_data.py](src/transformer_from_scratch/public_data.py) — public corpus preparation helpers
- [src/transformer_from_scratch/public_eval.py](src/transformer_from_scratch/public_eval.py) — Tiny Shakespeare evaluation pipeline
- [src/transformer_from_scratch/public_benchmark.py](src/transformer_from_scratch/public_benchmark.py) — WikiText-2 benchmark and ablations
- [src/transformer_from_scratch/demo.py](src/transformer_from_scratch/demo.py) — end-to-end demo pipeline
- [src/transformer_from_scratch/train.py](src/transformer_from_scratch/train.py) — training CLI
- [src/transformer_from_scratch/benchmark.py](src/transformer_from_scratch/benchmark.py) — benchmarking CLI
- [src/transformer_from_scratch/generation.py](src/transformer_from_scratch/generation.py) — token generation CLI
- [src/transformer_from_scratch/baselines.py](src/transformer_from_scratch/baselines.py) — unigram and bigram baselines
- [src/transformer_from_scratch/tests](src/transformer_from_scratch/tests) — unit and smoke tests
- [README.md](README.md) — project overview and usage
- [RESULTS.md](RESULTS.md) — synthetic demo result summary
- [RESULTS_PUBLIC.md](RESULTS_PUBLIC.md) — Tiny Shakespeare result summary
- [RESULTS_BENCHMARK.md](RESULTS_BENCHMARK.md) — WikiText-2 benchmark and ablation summary

## Working principles

### Prefer small, safe changes

- Make the smallest change that solves the problem.
- Preserve public interfaces unless a change is clearly needed.
- Avoid unrelated refactors.
- Keep style consistent with the surrounding code.

### Keep the repo reproducible

- Any new experiment should be runnable from the CLI.
- If you add a feature, add a test if practical.
- If output changes, update the relevant results document.

### Do not overclaim

This project is a strong portfolio repo, but it is not a research breakthrough, SOTA system, or production-scale platform.
Write descriptions accordingly.

## Common workflows

### 1. Check the project state

Before making changes:

- inspect the relevant source file
- inspect tests that cover the area
- inspect the README or results doc if the change affects usage or outputs
- run tests after edits

### 2. Add a feature

Typical steps:

1. find the relevant module
2. implement the feature in the smallest appropriate place
3. update CLI flags or config objects if needed
4. add or adjust tests
5. run the test suite
6. update docs or result files if behavior changed

### 3. Fix a bug

Typical steps:

1. reproduce or infer the failing path
2. inspect the code path end to end
3. patch the root cause
4. validate with tests
5. check whether any docs or generated results need updates

### 4. Add a new experiment

Use the same structure already in the repo:

- add data preparation if needed
- train through the existing CLI or a new CLI wrapper
- evaluate with a reproducible command
- save a JSON or markdown summary under the repo

## Validation commands

Use these checks when making code changes:

- `PYTHONPATH=src pytest -q`
- `make checks`
- `make public-eval`
- `make public-benchmark`
- `make benchmark`

If a change affects only a small area, run the relevant subset first, then the full suite if practical.

## Code style guidance

- Use clear, descriptive names.
- Keep functions focused.
- Prefer dataclasses for config.
- Use type hints where the file already uses them.
- Keep CLI arguments explicit and documented by parser defaults.
- Preserve the existing formatting style.
- Prefer deterministic behavior when possible.

## Experiment and results rules

If you run a new meaningful experiment:

- record the command used
- capture the important metrics
- write them to a results markdown file if they are worth keeping
- keep old results unless there is a reason to replace them

Recommended result fields:

- dataset
- model configuration
- baseline values
- validation loss
- perplexity
- benchmark throughput
- latency
- parameter count
- artifact paths

## When editing tests

- Keep tests short and deterministic.
- Prefer shape checks, config checks, and smoke tests.
- Avoid flaky tests that depend on timing or unstable network access.
- If a test covers a new feature, make sure it fails before the feature and passes after.

## When editing README or results files

Update the docs if you change:

- CLI flags
- default behavior
- model capabilities
- evaluation pipelines
- benchmark numbers
- output locations

## Before finishing work

Confirm the following:

- code changes are complete
- tests pass
- no obvious lint or formatting problems remain
- docs are updated if behavior changed
- commit message would clearly describe the change

## Suggested agent behavior

If an agent is continuing work in this repo, the safest order is:

1. read this file
2. inspect the relevant source files
3. make the minimal implementation
4. validate with tests
5. update docs or results if needed
6. summarize exactly what changed

## Notes

This repository is intended to be easy to extend, easy to test, and easy to explain.
Keep changes aligned with that goal.
