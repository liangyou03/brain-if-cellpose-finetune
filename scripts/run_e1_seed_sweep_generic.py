#!/usr/bin/env python
"""
Purpose: Run E1.6 generic fine-tune seed sweep for training stability.
Output: per-seed test metrics plus mean/std summary tables.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from fintune.inference.cpsam_baseline import run_baseline_inference
from fintune.training.finetune_generic import finetune_generic


def _load_metrics(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _mean_std(vals):
    if len(vals) == 0:
        return None, None
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, v**0.5


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--nimg-per-epoch", type=int, default=72)
    ap.add_argument("--nimg-test-per-epoch", type=int, default=24)
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--report-prefix", default="e1_seed_sweep_generic")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    reports_dir = Path(cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    pred_root = Path(cfg["predictions_dir"])

    rows = []
    for seed in args.seeds:
        model_name = f"generic_seed_{seed}"
        metrics_path = pred_root / model_name / "test" / "metrics.json"

        if args.skip_existing and metrics_path.exists():
            metrics = _load_metrics(metrics_path)
        else:
            saved = finetune_generic(
                config_path=args.config,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch,
                resume_ckpt=None,
                model_name=model_name,
                marker_filter=None,
                nimg_per_epoch=args.nimg_per_epoch,
                nimg_test_per_epoch=args.nimg_test_per_epoch,
                zero_dapi=False,
                seed=seed,
            )
            run_baseline_inference(
                config_path=args.config,
                model=str(saved),
                split="test",
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
            )
            metrics = _load_metrics(metrics_path)

        rec = {
            "seed": seed,
            "model_name": model_name,
            "ap50": metrics["overall"]["ap50"],
            "precision": metrics["overall"]["precision"],
            "recall": metrics["overall"]["recall"],
            "f1": metrics["overall"]["f1"],
            "tp": metrics["overall"]["tp"],
            "fp": metrics["overall"]["fp"],
            "fn": metrics["overall"]["fn"],
            "metrics_json": str(metrics_path),
        }
        rows.append(rec)
        print(f"[e1.6] seed={seed} overall={metrics['overall']}")

    out_tsv = reports_dir / f"{args.report_prefix}.tsv"
    with open(out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    summary = {"n_seeds": len(rows), "seeds": args.seeds}
    for k in ["ap50", "precision", "recall", "f1"]:
        vals = [float(r[k]) for r in rows]
        mean, std = _mean_std(vals)
        summary[k] = {"mean": mean, "std": std, "min": min(vals), "max": max(vals)}

    out_json = reports_dir / f"{args.report_prefix}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[e1.6] saved {out_tsv}")
    print(f"[e1.6] saved {out_json}")


if __name__ == "__main__":
    main()
