from __future__ import annotations

from pathlib import Path

from transformer_from_scratch.demo import main as demo_main


def test_demo_creates_report(tmp_path: Path) -> None:
    demo_main(
        [
            "--output-dir",
            str(tmp_path / "showcase"),
            "--vocab-size",
            "32",
            "--model-seq-len",
            "8",
            "--n-train",
            "16",
            "--n-valid",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--d-model",
            "16",
            "--n-layers",
            "1",
            "--n-heads",
            "4",
            "--d-ff",
            "32",
            "--dropout",
            "0.0",
            "--benchmark-iterations",
            "1",
        ]
    )

    report_path = tmp_path / "showcase" / "report.json"
    assert report_path.exists()
