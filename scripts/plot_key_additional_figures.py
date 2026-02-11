#!/usr/bin/env python
"""
Purpose: Generate key publication figures for E1 (marker gain, representative masks).
Output: Colorblind-safe 300-dpi PDF/PNG files under data/reports/fig_pub.
"""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#6F6F6F",
}


def _set_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def _save(fig, out_base: Path):
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


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


def _overlay_boundary(base_rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 1.0):
    out = base_rgb.copy()
    b = _boundary(mask)
    out[b] = (1.0 - alpha) * out[b] + alpha * np.asarray(color, dtype=np.float32)
    return out


def _seed_from_text(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def _pastel_palette(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = rng.random(n)
    s, v = 0.35, 0.95
    rgb = []
    for h in hues:
        i = int(h * 6.0) % 6
        f = h * 6.0 - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        rgb.append((r, g, b))
    return np.array(rgb, dtype=np.float32)


def _label_viz_on_gray(mask: np.ndarray, seed_text: str, fill_alpha: float = 0.9) -> np.ndarray:
    h, w = mask.shape[:2]
    img = np.full((h, w, 3), 0.98, dtype=np.float32)
    ids = [i for i in np.unique(mask) if i != 0]
    if not ids:
        return img
    pal = _pastel_palette(len(ids), _seed_from_text(seed_text))
    id2idx = {i: k for k, i in enumerate(ids)}
    for i in ids:
        m = mask == i
        if not m.any():
            continue
        c = pal[id2idx[i]]
        img[m] = img[m] * (1 - fill_alpha) + c * fill_alpha
    # dark boundaries for better separation
    b = _boundary(mask)
    img[b] = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    return img


def _read_tsv(path: Path):
    with open(path, "r", newline="") as f:
        rr = csv.DictReader(f, delimiter="\t")
        return [r for r in rr]


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _count_instances(mask: np.ndarray) -> int:
    ids = np.unique(mask.astype(np.int64))
    return int(np.sum(ids > 0))


def plot_e1_marker_gain(out_dir: Path):
    base = _load_json(Path("data/predictions/cpsam/test/metrics.json"))
    gen = _load_json(Path("data/predictions/generic_cpsam/test/metrics.json"))

    groups = ["overall", "gfap", "iba1", "neun", "olig2"]
    labels = ["Overall", "GFAP", "IBA1", "NeuN", "OLIG2"]
    metrics = [("ap50", "AP@0.5"), ("recall", "Recall"), ("f1", "F1")]

    x = np.arange(len(groups))
    w = 0.37
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True, sharey=False)
    for ax, (mkey, title) in zip(axes, metrics):
        y_base = np.array([float(base[g][mkey]) for g in groups], dtype=float)
        y_gen = np.array([float(gen[g][mkey]) for g in groups], dtype=float)
        delta = y_gen - y_base
        ax.bar(x - w / 2, y_base, width=w, color=OKABE_ITO["gray"], label="Baseline", edgecolor="white", linewidth=0.8)
        ax.bar(x + w / 2, y_gen, width=w, color=OKABE_ITO["blue"], label="Generic FT", edgecolor="white", linewidth=0.8)
        for i, d in enumerate(delta):
            ax.text(i + w / 2, min(0.98, y_gen[i] + 0.02), f"{d:+.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Test Score")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("E1 Marker-Wise Gain: Baseline vs Generic Fine-Tune", y=1.02, fontsize=13, fontweight="bold")
    _save(fig, out_dir / "fig_e1_marker_gain")


def plot_e1_mask_examples(out_dir: Path):
    metric_tsv = Path("data/reports/e1_visual_compare_mask_full/per_tile_metrics.tsv")
    rows = _read_tsv(metric_tsv)
    by_tile_model = {}
    for r in rows:
        by_tile_model[(r["tile"], r["model"])] = r

    markers = ["gfap", "iba1", "neun", "olig2"]
    selected = {}
    for marker in markers:
        candidates = sorted({r["tile"] for r in rows if r["tile"].startswith(marker + "_")})
        best_tile = None
        best_delta = -1e9
        for tile in candidates:
            rb = by_tile_model.get((tile, "baseline"))
            rg = by_tile_model.get((tile, "generic"))
            if rb is None or rg is None:
                continue
            d = float(rg["ap50"]) - float(rb["ap50"])
            if d > best_delta:
                best_delta = d
                best_tile = tile
        if best_tile is None:
            continue
        selected[marker] = best_tile

    if len(selected) == 0:
        raise RuntimeError("No representative E1 tiles found.")

    fig, axes = plt.subplots(len(selected), 4, figsize=(14, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.array([axes])

    col_titles = ["Input Composite", "GT Mask", "Baseline Mask", "Generic FT Mask"]
    for j in range(4):
        axes[0, j].set_title(col_titles[j])

    for i, marker in enumerate(markers):
        tile = selected[marker]
        img = tiff.imread(Path(f"data/cellpose/test_cells/{tile}.tif"))
        gt = tiff.imread(Path(f"data/cellpose/test_cells/{tile}_mask.tif")).astype(np.int32)
        pb = tiff.imread(Path(f"data/predictions/cpsam/test/{tile}_pred.tif")).astype(np.int32)
        pg = tiff.imread(Path(f"data/predictions/generic_cpsam/test/{tile}_pred.tif")).astype(np.int32)
        rgb = _to_rgb(img)
        h, w = gt.shape[:2]
        n_gt = _count_instances(gt)
        n_pb = _count_instances(pb)
        n_pg = _count_instances(pg)

        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(_label_viz_on_gray(gt, seed_text=f"{tile}_gt"))
        axes[i, 2].imshow(_label_viz_on_gray(pb, seed_text=f"{tile}_baseline"))
        axes[i, 3].imshow(_label_viz_on_gray(pg, seed_text=f"{tile}_generic"))

        rb = by_tile_model[(tile, "baseline")]
        rg = by_tile_model[(tile, "generic")]
        delta_ap = float(rg["ap50"]) - float(rb["ap50"])
        delta_f1 = float(rg["f1"]) - float(rb["f1"])
        info_text = (
            f"{marker.upper()} | {tile}\n"
            f"Image: {h}x{w} px\n"
            f"Instances (GT/Base/FT): {n_gt}/{n_pb}/{n_pg}"
        )
        axes[i, 0].text(
            0.02,
            0.98,
            info_text,
            transform=axes[i, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 4.0, "edgecolor": "none"},
        )
        axes[i, 2].text(
            0.02,
            0.98,
            f"AP50={float(rb['ap50']):.3f}\nF1={float(rb['f1']):.3f}",
            transform=axes[i, 2].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.80, "pad": 2.2, "edgecolor": "none"},
        )
        axes[i, 3].text(
            0.02,
            0.98,
            (
                f"AP50={float(rg['ap50']):.3f} ({delta_ap:+.3f})\n"
                f"F1={float(rg['f1']):.3f} ({delta_f1:+.3f})"
            ),
            transform=axes[i, 3].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.80, "pad": 2.2, "edgecolor": "none"},
        )
        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle("E1 Representative Masks (GT / Baseline / Generic FT)", y=1.01, fontsize=13, fontweight="bold")
    _save(fig, out_dir / "fig_e1_mask_examples")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/reports/fig_pub")
    return ap.parse_args()


def main():
    args = parse_args()
    _set_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_e1_marker_gain(out_dir)
    plot_e1_mask_examples(out_dir)

    print(f"[fig] saved to {out_dir}")
    for p in sorted(out_dir.glob("fig_e1_marker_gain.*")):
        print(f"  - {p}")
    for p in sorted(out_dir.glob("fig_e1_mask_examples.*")):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
