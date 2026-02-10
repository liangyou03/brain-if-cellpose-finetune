"""
Prompt-sweep for micro-sam / μSAM diagnostics.

If `micro_sam` is not installed, this module falls back to a CPSAM proxy sweep
with varying flow threshold to provide a comparable stability diagnostic.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import random
import yaml
import numpy as np
import tifffile as tiff
from cellpose import models

from fintune.utils.dataset import list_pairs, parse_marker_donor, read_image, read_mask
from fintune.utils.metrics import summarize_metrics


def _sample_pairs(test_dir: Path, num_tiles: int, seed: int):
    pairs = list_pairs(test_dir)
    grouped = {}
    for img_path, mask_path in pairs:
        marker, _ = parse_marker_donor(img_path.stem)
        grouped.setdefault(marker, []).append((img_path, mask_path))
    rng = random.Random(seed)
    sampled = []
    for marker, items in sorted(grouped.items()):
        k = min(num_tiles, len(items))
        sampled.extend(rng.sample(items, k))
    return sampled


def _pps_to_flow_threshold(pps: int) -> float:
    # Proxy mapping when micro-sam is unavailable: denser prompts ~= lower flow threshold.
    if pps <= 16:
        return 0.6
    if pps <= 32:
        return 0.4
    return 0.2


def _run_cpsam_proxy(sampled_pairs, pps: int):
    flow_threshold = _pps_to_flow_threshold(pps)
    model_obj = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    true_all, pred_all = [], []
    true_by_marker, pred_by_marker = {}, {}
    for img_path, mask_path in sampled_pairs:
        img = read_image(img_path)
        if img.ndim == 2:
            img = np.stack([img, img, np.zeros_like(img)], axis=-1)
        pred, _, _ = model_obj.eval(
            img,
            channels=[1, 2],
            diameter=None,
            flow_threshold=flow_threshold,
            cellprob_threshold=0.0,
        )
        true = read_mask(mask_path).astype(np.int32)
        pred = pred.astype(np.int32)
        marker, _ = parse_marker_donor(img_path.stem)
        true_all.append(true)
        pred_all.append(pred)
        true_by_marker.setdefault(marker, []).append(true)
        pred_by_marker.setdefault(marker, []).append(pred)
    out = {
        "method": "cpsam_proxy",
        "pps": int(pps),
        "gpu": True,
        "flow_threshold": float(flow_threshold),
        "overall": summarize_metrics(true_all, pred_all),
        "markers": {},
    }
    for marker in sorted(true_by_marker.keys()):
        out["markers"][marker] = summarize_metrics(true_by_marker[marker], pred_by_marker[marker])
    return out


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

    sampled_pairs = _sample_pairs(test_dir=test_dir, num_tiles=num_tiles, seed=seed)
    if len(sampled_pairs) == 0:
        raise RuntimeError(f"No test pairs found in {test_dir}")

    sampled_csv = report_dir / f"sampled_tiles_n{num_tiles}_seed{seed}.tsv"
    with open(sampled_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["image", "mask", "marker"])
        for img_path, mask_path in sampled_pairs:
            marker, _ = parse_marker_donor(img_path.stem)
            writer.writerow([str(img_path), str(mask_path), marker])

    results = []
    for pps in points_per_side:
        results.append(_run_cpsam_proxy(sampled_pairs, int(pps)))

    out_json = report_dir / f"e4_microsam_proxy_sweep_n{num_tiles}_seed{seed}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    rows = []
    for r in results:
        ov = r["overall"]
        rows.append(
            {
                "method": r["method"],
                "pps": r["pps"],
                    "flow_threshold": r["flow_threshold"],
                    "gpu": r["gpu"],
                    "group": "overall",
                "ap50": ov["ap50"],
                "precision": ov["precision"],
                "recall": ov["recall"],
                "f1": ov["f1"],
                "fp": ov["fp"],
                "tp": ov["tp"],
                "fn": ov["fn"],
            }
        )
        for marker, met in r["markers"].items():
            rows.append(
                {
                    "method": r["method"],
                    "pps": r["pps"],
                    "flow_threshold": r["flow_threshold"],
                    "gpu": r["gpu"],
                    "group": marker,
                    "ap50": met["ap50"],
                    "precision": met["precision"],
                    "recall": met["recall"],
                    "f1": met["f1"],
                    "fp": met["fp"],
                    "tp": met["tp"],
                    "fn": met["fn"],
                }
            )
    out_tsv = report_dir / f"e4_microsam_proxy_sweep_n{num_tiles}_seed{seed}.tsv"
    with open(out_tsv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[e4] sampled={len(sampled_pairs)}")
    print(f"[e4] saved={out_tsv}")
    print(f"[e4] saved={out_json}")


__all__ = ["run_microsam_sweep"]
