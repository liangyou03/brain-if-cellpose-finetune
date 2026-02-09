"""
Donor-level downstream analysis:
- cell-body density per donor/marker
- Spearman correlation vs pathology/cognition
- Bootstrap stability comparison between baseline & fine-tuned
"""
from __future__ import annotations

from pathlib import Path
import yaml


def run_downstream_analysis(
    config_path: str,
    pred_dir: str,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    pred_dir = Path(pred_dir)
    downstream_dir = Path(cfg["downstream_dir"])
    downstream_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[stub] Downstream correlations using {pred_dir}, bootstrap={n_bootstrap}, alpha={alpha}, output-> {downstream_dir}"
    )


__all__ = ["run_downstream_analysis"]
