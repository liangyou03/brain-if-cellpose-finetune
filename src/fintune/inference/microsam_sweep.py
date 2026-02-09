"""
Prompt-sweep for micro-sam / μSAM to study robustness on dense/weak-boundary markers.
"""
from __future__ import annotations

from pathlib import Path
import yaml


def run_microsam_sweep(
    config_path: str,
    points_per_side,
    num_tiles: int = 10,
    seed: int = 2024,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    test_dir = Path(cfg["cellpose_test_dir"])
    report_dir = Path(cfg.get("reports_dir", cfg["logs_dir"])) / "microsam"
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stub] micro-sam sweep on {test_dir}, pps={points_per_side}, n_tiles={num_tiles}, seed={seed}, save-> {report_dir}")


__all__ = ["run_microsam_sweep"]
