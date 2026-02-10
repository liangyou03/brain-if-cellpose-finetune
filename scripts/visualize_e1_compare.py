#!/usr/bin/env python
"""
Purpose: Visualize E1 baseline vs finetuned mask comparison on shared test tiles.
Output: side-by-side overlays, per-tile metric TSV, and optional summary boxplots.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import yaml

from fintune.utils.dataset import list_pairs, read_image, read_mask
from fintune.utils.metrics import summarize_metrics


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.percentile(x, 1))
    hi = float(np.percentile(x, 99))
    if hi <= lo:
        hi = lo + 1.0
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        v = _norm01(img)
        return np.stack([v, v, v], axis=-1)
    if img.shape[-1] >= 2:
        dapi = _norm01(img[..., 0])
        marker = _norm01(img[..., 1])
        # R=marker, G=DAPI, B=marker to make cell body signal clear.
        return np.stack([marker, dapi, marker], axis=-1)
    v = _norm01(img[..., 0])
    return np.stack([v, v, v], axis=-1)


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.int64)
    b = np.zeros_like(m, dtype=bool)
    b[1:, :] |= m[1:, :] != m[:-1, :]
    b[:, 1:] |= m[:, 1:] != m[:, :-1]
    b &= m > 0
    return b


def _overlay_boundary(base_rgb: np.ndarray, mask: np.ndarray, color=(1.0, 0.0, 0.0), alpha: float = 1.0):
    out = base_rgb.copy()
    b = _boundary(mask)
    out[b] = (1.0 - alpha) * out[b] + alpha * np.array(color, dtype=np.float32)
    return out


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.int64)
    h = (m * 0.61803398875) % 1.0
    s = np.where(m > 0, 0.75, 0.0).astype(np.float32)
    v = np.where(m > 0, 1.00, 0.0).astype(np.float32)
    hsv = np.stack([h.astype(np.float32), s, v], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


def _load_pred_mask(pred_dir: Path, stem: str) -> np.ndarray | None:
    p = pred_dir / f"{stem}_pred.tif"
    if not p.exists():
        return None
    return tiff.imread(p).astype(np.int32)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument(
        "--pred-dirs",
        nargs="+",
        default=["data/predictions/cpsam/test", "data/predictions/generic_cpsam/test"],
        help="prediction directories, each containing *_pred.tif and metrics.json",
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        default=["baseline", "generic"],
        help="display labels corresponding to --pred-dirs",
    )
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out-dir", default="data/reports/e1_visual_compare")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--plot-summary", action="store_true")
    ap.add_argument(
        "--view",
        choices=["boundary", "mask"],
        default="boundary",
        help="boundary: overlay boundaries on composite; mask: show raw instance masks",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if len(args.pred_dirs) != len(args.labels):
        raise ValueError("--pred-dirs and --labels must have the same length")

    cfg = yaml.safe_load(Path(args.config).read_text())
    split_map = {
        "train": Path(cfg["cellpose_train_dir"]),
        "val": Path(cfg["cellpose_val_dir"]),
        "test": Path(cfg["cellpose_test_dir"]),
    }
    data_dir = split_map[args.split]
    pairs = list_pairs(data_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No image/mask pairs found in {data_dir}")

    rng = random.Random(args.seed)
    k = min(args.num_samples, len(pairs))
    sampled = rng.sample(pairs, k)
    sampled = sorted(sampled, key=lambda x: x[0].stem)

    pred_dirs = [Path(p) for p in args.pred_dirs]
    for p in pred_dirs:
        if not p.exists():
            raise FileNotFoundError(f"Prediction dir not found: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path, mask_path in sampled:
        stem = img_path.stem
        img = read_image(img_path)
        gt = read_mask(mask_path).astype(np.int32)
        base_rgb = _to_rgb(img)

        if args.view == "boundary":
            ncols = 2 + len(pred_dirs)
        else:
            ncols = 1 + len(pred_dirs)
        fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.4))
        if ncols == 1:
            axes = [axes]

        if args.view == "boundary":
            axes[0].imshow(base_rgb)
            axes[0].set_title(f"{stem}\nComposite")
            axes[0].axis("off")

            gt_vis = _overlay_boundary(base_rgb, gt, color=(1.0, 1.0, 0.0), alpha=1.0)
            axes[1].imshow(gt_vis)
            axes[1].set_title("GT")
            axes[1].axis("off")
            pred_offset = 2
        else:
            axes[0].imshow(_mask_to_rgb(gt))
            axes[0].set_title(f"{stem}\nGT mask")
            axes[0].axis("off")
            pred_offset = 1

        for j, (label, pred_dir) in enumerate(zip(args.labels, pred_dirs)):
            i = pred_offset + j
            pred = _load_pred_mask(pred_dir, stem)
            if pred is None:
                axes[i].imshow(base_rgb if args.view == "boundary" else np.zeros_like(base_rgb))
                axes[i].set_title(f"{label}\nmissing pred")
                axes[i].axis("off")
                continue

            m = summarize_metrics([gt], [pred], threshold=0.5)
            if args.view == "boundary":
                vis = _overlay_boundary(base_rgb, pred, color=(1.0, 0.2, 0.2), alpha=1.0)
            else:
                vis = _mask_to_rgb(pred)
            axes[i].imshow(vis)
            axes[i].set_title(
                f"{label}\nAP50={m['ap50']:.3f} F1={m['f1']:.3f}\nP={m['precision']:.3f} R={m['recall']:.3f}"
            )
            axes[i].axis("off")

            rows.append(
                {
                    "tile": stem,
                    "model": label,
                    "pred_dir": str(pred_dir),
                    "ap50": m["ap50"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "tp": m["tp"],
                    "fp": m["fp"],
                    "fn": m["fn"],
                }
            )

        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.png", dpi=args.dpi)
        plt.close(fig)

    metrics_tsv = out_dir / "per_tile_metrics.tsv"
    if len(rows) > 0:
        keys = ["tile", "model", "pred_dir", "ap50", "precision", "recall", "f1", "tp", "fp", "fn"]
        with open(metrics_tsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    if args.plot_summary and len(rows) > 0:
        by_model_ap = {}
        by_model_f1 = {}
        for r in rows:
            by_model_ap.setdefault(r["model"], []).append(float(r["ap50"]))
            by_model_f1.setdefault(r["model"], []).append(float(r["f1"]))

        models = list(by_model_ap.keys())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].boxplot([by_model_ap[m] for m in models], tick_labels=models)
        axes[0].set_title("AP50 (sampled tiles)")
        axes[0].set_ylabel("AP50")

        axes[1].boxplot([by_model_f1[m] for m in models], tick_labels=models)
        axes[1].set_title("F1 (sampled tiles)")
        axes[1].set_ylabel("F1")

        fig.tight_layout()
        fig.savefig(out_dir / "summary_boxplot.png", dpi=args.dpi)
        plt.close(fig)

    print(f"[vis] split={args.split}, sampled={k}")
    print(f"[vis] out_dir={out_dir}")
    if metrics_tsv.exists():
        print(f"[vis] metrics={metrics_tsv}")


if __name__ == "__main__":
    main()
