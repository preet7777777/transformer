# End-to-End Demo Results

Generated on 2026-05-13 from the end-to-end demo pipeline.

## Run configuration

- Vocabulary size: 64
- Model sequence length: 16
- Synthetic mode: copy
- Training sequences: 64
- Validation sequences: 32
- Epochs: 30
- Batch size: 8
- Learning rate: 0.001
- Model width: 32
- Layers: 1
- Heads: 4
- Feed-forward width: 64
- Dropout: 0.0

## Training result

- Final training loss: 1.3405963331460953
- Best validation loss: 2.2452710270881653
- Checkpoint: runs/showcase-copy-30/runs/best.pt

## Benchmark result

- Device: cpu
- Batch size: 8
- Sequence length: 16
- Iterations: 20
- Parameters: 11,040
- Mean batch latency: 0.2955 ms
- Throughput: 433,144.12 tokens/sec

## Artifacts

- Training report: runs/showcase-copy-30/runs/training_report.json
- Full demo report: runs/showcase-copy-30/report.json
