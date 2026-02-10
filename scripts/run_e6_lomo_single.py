#!/usr/bin/env python
"""
Purpose: Run E6.1 leave-one-marker-out training and held-out marker evaluation.
Output: per-heldout-marker checkpoints and test metric tables.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile as tiff
import yaml
from cellpose import models

from fintune.training.finetune_generic import finetune_generic
from fintune.utils.dataset import list_pairs, parse_marker_donor, read_image, read_mask
from fintune.utils.metrics import summarize_metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--heldout-marker", required=True, choices=["gfap", "iba1", "neun", "olig2"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--nimg-per-epoch", type=int, default=72)
    ap.add_argument("--nimg-test-per-epoch", type=int, default=24)
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2024)
    return ap.parse_args()


def _eval_test_heldout(cfg: dict, model_spec: str, heldout: str, flow: float, cellprob: float):
    test_dir = Path(cfg["cellpose_test_dir"])
    pairs = [(i, m) for i, m in list_pairs(test_dir) if parse_marker_donor(i.stem)[0] == heldout]
    if len(pairs) == 0:
        raise RuntimeError(f"No test pairs for heldout marker={heldout}")

    model_obj = models.CellposeModel(gpu=True, pretrained_model=model_spec)
    y_true, y_pred = [], []
    for img_path, mask_path in pairs:
        img = read_image(img_path)
        if img.ndim == 2:
            img = np.stack([img, img, np.zeros_like(img)], axis=-1)
        pred, _, _ = model_obj.eval(
            img,
            channels=[1, 2],
            diameter=None,
            flow_threshold=flow,
            cellprob_threshold=cellprob,
        )
        pred = pred.astype(np.int32)
        true = read_mask(mask_path).astype(np.int32)
        y_true.append(true)
        y_pred.append(pred)
    return summarize_metrics(y_true, y_pred), pairs


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    heldout = args.heldout_marker.lower()
    model_name = f"lomo_excl_{heldout}"

    saved = finetune_generic(
        config_path=args.config,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        resume_ckpt=None,
        model_name=model_name,
        marker_exclude=[heldout],
        nimg_per_epoch=args.nimg_per_epoch,
        nimg_test_per_epoch=args.nimg_test_per_epoch,
        zero_dapi=False,
        seed=args.seed,
    )
    model_spec = str(saved)
    if not Path(model_spec).exists():
        model_spec = str(Path(cfg["checkpoints_dir"]) / model_name / "models" / model_name)

    metrics, pairs = _eval_test_heldout(
        cfg=cfg,
        model_spec=model_spec,
        heldout=heldout,
        flow=args.flow_threshold,
        cellprob=args.cellprob_threshold,
    )

    pred_dir = Path(cfg["predictions_dir"]) / model_name / "test_heldout"
    pred_dir.mkdir(parents=True, exist_ok=True)
    with open(pred_dir / "metrics.json", "w") as f:
        json.dump({"heldout_marker": heldout, "overall": metrics}, f, indent=2)

    reports_dir = Path(cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "model_name": model_name,
        "heldout_marker": heldout,
        "train_marker_exclude": heldout,
        "n_test_images": len(pairs),
        "flow_threshold": args.flow_threshold,
        "cellprob_threshold": args.cellprob_threshold,
        "ap50": metrics["ap50"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "metrics_json": str(pred_dir / "metrics.json"),
    }
    out_tsv = reports_dir / f"e6_lomo_{heldout}.tsv"
    with open(out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        w.writeheader()
        w.writerow(row)

    print(f"[e6] heldout={heldout} metrics={metrics}")
    print(f"[e6] saved={out_tsv}")


if __name__ == "__main__":
    main()
