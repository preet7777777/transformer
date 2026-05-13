# Public Dataset Validation Results

Generated on 2026-05-13 from the end-to-end Tiny Shakespeare evaluation.

## Dataset

- Dataset: Tiny Shakespeare
- Source: karpathy/char-rnn
- Vocabulary size: 65
- Sequence length: 64
- Train windows: 15,443
- Validation windows: 1,716

## Baselines and model result

- Unigram baseline loss: 3.347513876459042
- Unigram baseline perplexity: 28.43196033871676
- Model validation loss: 2.0015726940972463
- Model perplexity: 7.400685966556666

## Training configuration

- Epochs: 3
- Batch size: 128
- Learning rate: 0.001
- Model width: 128
- Layers: 2
- Heads: 4
- Feed-forward width: 256
- Dropout: 0.1
- Positional encoding: rope

## Benchmark

- Device: cpu
- Parameters: 272,512
- Mean batch latency: 25.9205 ms
- Throughput: 158,021.73 tokens/sec

## Artifacts

- Training report: runs/public-eval/runs/training_report.json
- Public report: runs/public-eval/public_report.json