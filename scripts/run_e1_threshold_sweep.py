#!/usr/bin/env python
"""
Purpose: Run E1.5 threshold sweep on Val and evaluate selected thresholds on Test.
Output: threshold-grid metrics and final selected-threshold test metrics.
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

from fintune.utils.dataset import list_pairs, parse_marker_donor, read_image, read_mask
from fintune.utils.metrics import summarize_metrics


def _resolve_model_spec(cfg: dict, alias: str) -> tuple[str, bool]:
    alias = alias.lower()
    ckpt_dir = Path(cfg["checkpoints_dir"])
    if alias == "baseline":
        return "cpsam", False
    if alias == "generic":
        return str(ckpt_dir / "generic_cpsam" / "models" / "generic_cpsam"), False
    if alias == "gfap":
        return str(ckpt_dir / "gfap_cpsam" / "models" / "gfap_cpsam"), False
    if alias == "marker_only":
        return str(ckpt_dir / "marker_only_cpsam" / "models" / "marker_only_cpsam"), True
    raise ValueError(f"Unknown model alias: {alias}")


def _split_dir(cfg: dict, split: str) -> Path:
    if split == "train":
        return Path(cfg["cellpose_train_dir"])
    if split == "val":
        return Path(cfg["cellpose_val_dir"])
    if split == "test":
        return Path(cfg["cellpose_test_dir"])
    raise ValueError(split)


def _eval_once(
    model_obj,
    pairs,
    flow_threshold: float,
    cellprob_threshold: float,
    zero_dapi: bool,
    save_pred_dir: Path | None = None,
):
    overall_true, overall_pred = [], []
    m_true, m_pred = {}, {}
    for img_path, mask_path in pairs:
        img = read_image(img_path)
        if img.ndim == 2:
            img = np.stack([img, img, np.zeros_like(img)], axis=-1)
        if zero_dapi and img.ndim == 3 and img.shape[-1] >= 1:
            img = img.copy()
            img[..., 0] = 0
        pred, _, _ = model_obj.eval(
            img,
            channels=[1, 2],
            diameter=None,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        pred = pred.astype(np.int32)
        true = read_mask(mask_path).astype(np.int32)
        overall_true.append(true)
        overall_pred.append(pred)
        marker, _ = parse_marker_donor(img_path.stem)
        m_true.setdefault(marker, []).append(true)
        m_pred.setdefault(marker, []).append(pred)
        if save_pred_dir is not None:
            tiff.imwrite(save_pred_dir / f"{img_path.stem}_pred.tif", pred.astype(np.uint16))

    out = {"overall": summarize_metrics(overall_true, overall_pred)}
    for marker in sorted(m_true.keys()):
        out[marker] = summarize_metrics(m_true[marker], m_pred[marker])
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--models", nargs="+", default=["baseline", "generic", "gfap", "marker_only"])
    ap.add_argument("--flow-thresholds", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    ap.add_argument("--cellprob-thresholds", nargs="+", type=float, default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    ap.add_argument("--select-metric", choices=["f1", "ap50"], default="f1")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    report_dir = Path(cfg["reports_dir"])
    pred_root = Path(cfg["predictions_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    pred_root.mkdir(parents=True, exist_ok=True)

    val_pairs = list_pairs(_split_dir(cfg, "val"))
    test_pairs = list_pairs(_split_dir(cfg, "test"))
    if len(val_pairs) == 0 or len(test_pairs) == 0:
        raise RuntimeError("No val/test pairs found.")

    final_rows = []
    for alias in args.models:
        model_spec, zero_dapi = _resolve_model_spec(cfg, alias)
        model_obj = models.CellposeModel(gpu=True, pretrained_model=model_spec)

        sweep_rows = []
        best = None
        for flow in args.flow_thresholds:
            for cellprob in args.cellprob_thresholds:
                m = _eval_once(
                    model_obj=model_obj,
                    pairs=val_pairs,
                    flow_threshold=float(flow),
                    cellprob_threshold=float(cellprob),
                    zero_dapi=zero_dapi,
                    save_pred_dir=None,
                )["overall"]
                row = {
                    "model_alias": alias,
                    "model_spec": model_spec,
                    "split": "val",
                    "flow_threshold": float(flow),
                    "cellprob_threshold": float(cellprob),
                    "ap50": m["ap50"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                }
                sweep_rows.append(row)
                score = row[args.select_metric]
                if best is None or score > best[args.select_metric]:
                    best = row

        sweep_tsv = report_dir / f"e1_threshold_sweep_{alias}_val.tsv"
        with open(sweep_tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(sweep_rows)

        pred_dir = pred_root / f"{alias}_thsel" / "test"
        pred_dir.mkdir(parents=True, exist_ok=True)
        test_metrics = _eval_once(
            model_obj=model_obj,
            pairs=test_pairs,
            flow_threshold=float(best["flow_threshold"]),
            cellprob_threshold=float(best["cellprob_threshold"]),
            zero_dapi=zero_dapi,
            save_pred_dir=pred_dir,
        )
        with open(pred_dir / "metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        final_rows.append(
            {
                "model_alias": alias,
                "selected_by": args.select_metric,
                "flow_threshold": best["flow_threshold"],
                "cellprob_threshold": best["cellprob_threshold"],
                "val_ap50": best["ap50"],
                "val_precision": best["precision"],
                "val_recall": best["recall"],
                "val_f1": best["f1"],
                "test_ap50": test_metrics["overall"]["ap50"],
                "test_precision": test_metrics["overall"]["precision"],
                "test_recall": test_metrics["overall"]["recall"],
                "test_f1": test_metrics["overall"]["f1"],
                "metrics_json": str(pred_dir / "metrics.json"),
            }
        )
        print(f"[e1.5] {alias} best val {args.select_metric}={best[args.select_metric]:.4f}")

    out_tsv = report_dir / "e1_threshold_selected_test.tsv"
    with open(out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(final_rows)
    print(f"[e1.5] saved {out_tsv}")


if __name__ == "__main__":
    main()
