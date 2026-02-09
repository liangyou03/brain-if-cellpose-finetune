"""
Run inference over all tiles for downstream donor-level aggregation.
"""
from __future__ import annotations

from pathlib import Path
import yaml


def run_full_inference(
    config_path: str,
    model_name: str = "fine_tuned",
    batch_size: int = 4,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    raw_tiles_dir = Path(cfg["raw_tiles_dir"])
    pred_dir = Path(cfg["predictions_dir"]) / f"{model_name}_full"
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stub] Full inference on {raw_tiles_dir}, model={model_name}, batch={batch_size}, save-> {pred_dir}")


__all__ = ["run_full_inference"]
