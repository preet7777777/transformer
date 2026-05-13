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

- Unigram baseline loss: 3.347305335368931
- Bigram baseline loss: 2.4818894321158234
- Unigram baseline perplexity: 28.426031724915003
- Bigram baseline perplexity: 11.963847954655076
- Model validation loss: 2.0040488498551503
- Model perplexity: 7.419033924551305

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
- Mean batch latency: 28.5888 ms
- Throughput: 143,273.09 tokens/sec

## Artifacts

- Training report: runs/public-eval/runs/training_report.json
- Public report: runs/public-eval/public_report.json