# Public WikiText-2 Benchmark and Ablation Results

Generated on 2026-05-13 from the public WikiText-2 benchmark and ablation suite.

## Dataset

- Dataset: WikiText-2
- Source: pytorch/examples
- Vocabulary size: 283
- Sequence length: 48
- Benchmark sequence length: 192
- Train tokens: 10,780,437
- Validation tokens: 1,120,192
- Train windows: 84,222
- Validation windows: 8,752

## Baselines

- Unigram baseline loss: 3.1954709865495854
- Bigram baseline loss: 2.3516948228158547
- Unigram baseline perplexity: 24.421673263636368
- Bigram baseline perplexity: 10.503355975548656

## Ablation suite

### RoPE + tied embeddings

- Validation loss: 1.8315346742021865
- Perplexity: 6.243460988418314
- Parameters: 176,160
- Benchmark throughput: 212,929.66 tokens/sec
- Mean batch latency: 14.4273 ms
- Long-context throughput at sequence length 192: 120,621.59 tokens/sec

### Learned embeddings + tied embeddings

- Validation loss: 2.0052545036094775
- Perplexity: 7.427984104981074
- Parameters: 180,768
- Benchmark throughput: 239,177.42 tokens/sec
- Mean batch latency: 12.8440 ms

### RoPE + untied embeddings

- Validation loss: 1.8178464001503543
- Perplexity: 6.158581038700923
- Parameters: 203,328
- Benchmark throughput: 226,281.17 tokens/sec
- Mean batch latency: 13.5760 ms
- Long-context throughput at sequence length 192: 134,418.72 tokens/sec

## Artifacts

- Full JSON report: runs/public_benchmark/public_benchmark_report.json
- RoPE tied run: runs/public_benchmark/runs/rope_tied/best.pt
- Learned tied run: runs/public_benchmark/runs/learned_tied/best.pt
- RoPE untied run: runs/public_benchmark/runs/rope_untied/best.pt
