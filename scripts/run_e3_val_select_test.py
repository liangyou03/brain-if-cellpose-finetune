#!/usr/bin/env python
"""
Purpose: Run E3.2 gating parameter selection on Val and locked evaluation on Test.
Output: val grid results and one final test result with selected gating setup.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from fintune.inference.cpsam_baseline import run_baseline_inference
from fintune.evaluation.gating_ablation import run_gating_ablation


def _resolve_model(cfg: dict, model_name: str) -> str:
    ckpt = Path(cfg["checkpoints_dir"]) / model_name / "models" / model_name
    if ckpt.exists():
        return str(ckpt)
    return model_name


def _read_overall_gated_f1(tsv_path: Path) -> float:
    with open(tsv_path, "r", newline="") as f:
        rr = csv.DictReader(f, delimiter="\t")
        for r in rr:
            if str(r.get("group", "")).lower() == "overall":
                return float(r["gated_f1"])
    raise RuntimeError(f"Missing overall row in {tsv_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--model", default="generic_cpsam")
    ap.add_argument("--q-low", nargs="+", type=float, default=[0.0, 0.02, 0.05])
    ap.add_argument("--q-high", nargs="+", type=float, default=[1.0, 0.98, 0.95])
    ap.add_argument("--min-area", nargs="+", type=int, default=[50, 60, 80])
    ap.add_argument("--select-metric", choices=["gated_f1"], default="gated_f1")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    reports_dir = Path(cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_spec = _resolve_model(cfg, args.model)

    run_baseline_inference(config_path=args.config, model=model_spec, split="val")
    run_baseline_inference(config_path=args.config, model=model_spec, split="test")

    val_rows = []
    best = None
    for ql in args.q_low:
        for qh in args.q_high:
            if ql >= qh:
                continue
            for ma in args.min_area:
                run_gating_ablation(
                    config_path=args.config,
                    model_name=args.model,
                    split="val",
                    intensity_thresh=(float(ql), float(qh)),
                    min_area=int(ma),
                )
                tag = f"q{int(ql*100)}_{int(qh*100)}_a{int(ma)}"
                val_tsv = reports_dir / f"e3_gating_{args.model}_val_{tag}.tsv"
                score = _read_overall_gated_f1(val_tsv)
                row = {
                    "model": args.model,
                    "split": "val",
                    "q_low": float(ql),
                    "q_high": float(qh),
                    "min_area": int(ma),
                    "gated_f1": score,
                    "val_tsv": str(val_tsv),
                }
                val_rows.append(row)
                if best is None or score > best["gated_f1"]:
                    best = row

    val_grid_tsv = reports_dir / f"e3_gating_val_grid_{args.model}.tsv"
    with open(val_grid_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(val_rows)

    run_gating_ablation(
        config_path=args.config,
        model_name=args.model,
        split="test",
        intensity_thresh=(best["q_low"], best["q_high"]),
        min_area=int(best["min_area"]),
    )
    tag = f"q{int(best['q_low']*100)}_{int(best['q_high']*100)}_a{int(best['min_area'])}"
    test_tsv = reports_dir / f"e3_gating_{args.model}_test_{tag}.tsv"

    final_tsv = reports_dir / f"e3_gating_test_final_{args.model}.tsv"
    with open(final_tsv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "selected_by", "q_low", "q_high", "min_area", "val_gated_f1", "test_tsv"],
            delimiter="\t",
        )
        w.writeheader()
        w.writerow(
            {
                "model": args.model,
                "selected_by": args.select_metric,
                "q_low": best["q_low"],
                "q_high": best["q_high"],
                "min_area": best["min_area"],
                "val_gated_f1": best["gated_f1"],
                "test_tsv": str(test_tsv),
            }
        )

    print(f"[e3.2] best={best}")
    print(f"[e3.2] val_grid={val_grid_tsv}")
    print(f"[e3.2] final={final_tsv}")


if __name__ == "__main__":
    main()
