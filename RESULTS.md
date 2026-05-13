# End-to-End Demo Results

Generated on 2026-05-13 from the CPU-only demo pipeline.

## Run configuration

- Vocabulary size: 64
- Model sequence length: 16
- Training sequences: 64
- Validation sequences: 32
- Epochs: 1
- Batch size: 8
- Learning rate: 0.001
- Model width: 32
- Layers: 1
- Heads: 4
- Feed-forward width: 64
- Dropout: 0.0

## Training result

- Final training loss: 4.167625367641449
- Best validation loss: 4.162992596626282
- Checkpoint: runs/showcase/runs/best.pt

## Benchmark result

- Device: cpu
- Batch size: 8
- Sequence length: 16
- Iterations: 20
- Parameters: 11,040
- Mean batch latency: 0.2373 ms
- Throughput: 539,487.65 tokens/sec

## Artifacts

- Training report: runs/showcase/runs/training_report.json
- Full demo report: runs/showcase/report.json
