"""
Compare metrics with/without marker intensity gating + area filtering.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import numpy as np
import tifffile as tiff
import yaml

from fintune.utils.dataset import list_pairs, read_image, read_mask, parse_marker_donor
from fintune.utils.metrics import summarize_metrics


def _relabel(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint16)
    ids = np.unique(mask)
    ids = ids[ids > 0]
    for new_id, old_id in enumerate(ids, start=1):
        out[mask == old_id] = new_id
    return out


def _apply_gating(mask_pred: np.ndarray, marker_img: np.ndarray, q_low: float, q_high: float, min_area: int) -> np.ndarray:
    out = np.zeros_like(mask_pred, dtype=np.uint16)
    ids = np.unique(mask_pred)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return out

    i_low = float(np.percentile(marker_img, q_low))
    i_high = float(np.percentile(marker_img, q_high))
    next_id = 1
    for obj_id in ids:
        region = mask_pred == obj_id
        area = int(region.sum())
        if area < min_area:
            continue
        mean_intensity = float(marker_img[region].mean())
        if mean_intensity < i_low or mean_intensity > i_high:
            continue
        out[region] = next_id
        next_id += 1
    return out


def run_gating_ablation(
    config_path: str,
    model_name: str = "generic_cpsam",
    intensity_thresh=(0.1, 0.3),
    min_area: int = 50,
    split: str = "test",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    split_to_dir = {
        "train": Path(cfg["cellpose_train_dir"]),
        "val": Path(cfg["cellpose_val_dir"]),
        "test": Path(cfg["cellpose_test_dir"]),
    }
    if split not in split_to_dir:
        raise ValueError(f"Unknown split={split}")
    data_dir = split_to_dir[split]
    pred_dir = Path(cfg["predictions_dir"]) / model_name / split
    report_dir = Path(cfg["reports_dir"]) if "reports_dir" in cfg else Path(cfg["logs_dir"]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    gated_dir = pred_dir / "gated"
    gated_dir.mkdir(parents=True, exist_ok=True)

    q_low = float(intensity_thresh[0])
    q_high = float(intensity_thresh[1])
    if q_low <= 1.0 and q_high <= 1.0:
        q_low *= 100.0
        q_high *= 100.0

    pairs = list_pairs(data_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No image/mask pairs in {data_dir}")

    raw_true, raw_pred = [], []
    gated_true, gated_pred = [], []
    raw_true_m, raw_pred_m = {}, {}
    gated_true_m, gated_pred_m = {}, {}

    for img_path, mask_path in pairs:
        stem = img_path.stem
        pred_path = pred_dir / f"{stem}_pred.tif"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction: {pred_path}")

        img = read_image(img_path)
        marker_img = img[..., 1] if img.ndim == 3 else img
        true = read_mask(mask_path).astype(np.int32)
        pred = read_mask(pred_path).astype(np.int32)
        gated = _apply_gating(pred, marker_img, q_low=q_low, q_high=q_high, min_area=min_area).astype(np.int32)
        gated = _relabel(gated)
        tiff.imwrite(gated_dir / f"{stem}_pred_gated.tif", gated.astype(np.uint16))

        marker, _ = parse_marker_donor(stem)
        raw_true.append(true)
        raw_pred.append(pred)
        gated_true.append(true)
        gated_pred.append(gated)
        raw_true_m.setdefault(marker, []).append(true)
        raw_pred_m.setdefault(marker, []).append(pred)
        gated_true_m.setdefault(marker, []).append(true)
        gated_pred_m.setdefault(marker, []).append(gated)

    raw_metrics = {"overall": summarize_metrics(raw_true, raw_pred)}
    gated_metrics = {"overall": summarize_metrics(gated_true, gated_pred)}
    for marker in sorted(raw_true_m):
        raw_metrics[marker] = summarize_metrics(raw_true_m[marker], raw_pred_m[marker])
    for marker in sorted(gated_true_m):
        gated_metrics[marker] = summarize_metrics(gated_true_m[marker], gated_pred_m[marker])

    tag = f"q{int(q_low)}_{int(q_high)}_a{int(min_area)}"
    out_json = {
        "model": model_name,
        "split": split,
        "intensity_percentile": [q_low, q_high],
        "min_area": int(min_area),
        "raw": raw_metrics,
        "gated": gated_metrics,
    }
    with open(report_dir / f"e3_gating_{model_name}_{split}_{tag}.json", "w") as f:
        json.dump(out_json, f, indent=2)

    rows = []
    keys = sorted(set(raw_metrics.keys()) | set(gated_metrics.keys()))
    for k in keys:
        r = raw_metrics.get(k, {})
        g = gated_metrics.get(k, {})
        rows.append(
            {
                "group": k,
                "raw_ap50": r.get("ap50", ""),
                "raw_precision": r.get("precision", ""),
                "raw_recall": r.get("recall", ""),
                "raw_f1": r.get("f1", ""),
                "gated_ap50": g.get("ap50", ""),
                "gated_precision": g.get("precision", ""),
                "gated_recall": g.get("recall", ""),
                "gated_f1": g.get("f1", ""),
                "delta_ap50": (g.get("ap50", 0.0) - r.get("ap50", 0.0)) if r else "",
                "delta_precision": (g.get("precision", 0.0) - r.get("precision", 0.0)) if r else "",
                "delta_recall": (g.get("recall", 0.0) - r.get("recall", 0.0)) if r else "",
                "delta_f1": (g.get("f1", 0.0) - r.get("f1", 0.0)) if r else "",
            }
        )
    out_tsv = report_dir / f"e3_gating_{model_name}_{split}_{tag}.tsv"
    with open(out_tsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[gating] model={model_name} split={split}")
    print(f"[gating] raw_overall={raw_metrics['overall']}")
    print(f"[gating] gated_overall={gated_metrics['overall']}")
    print(f"[gating] saved={out_tsv}")


__all__ = ["run_gating_ablation"]
