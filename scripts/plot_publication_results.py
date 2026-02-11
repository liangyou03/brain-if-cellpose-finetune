#!/usr/bin/env python
"""
Purpose: Create publication-grade figures for current E1/E2 results.
Output: colorblind-safe 300-dpi PDF figures (plus PNG previews) under data/reports/fig_pub.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}


def _set_pub_style():
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
            "grid.alpha": 0.7,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def _load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def _read_tsv(path: Path):
    with open(path, "r", newline="") as f:
        rr = csv.DictReader(f, delimiter="\t")
        return [r for r in rr]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None


def _save(fig, out_base: Path):
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_e1_overall(out_dir: Path):
    model_spec = [
        ("Baseline", Path("data/predictions/cpsam/test/metrics.json"), OKABE_ITO["black"]),
        ("Baseline+Thr", Path("data/predictions/baseline_thsel/test/metrics.json"), OKABE_ITO["yellow"]),
        ("Generic FT", Path("data/predictions/generic_cpsam/test/metrics.json"), OKABE_ITO["blue"]),
        ("Generic+Thr", Path("data/predictions/generic_thsel/test/metrics.json"), OKABE_ITO["sky"]),
        ("Marker-only", Path("data/predictions/marker_only_cpsam/test/metrics.json"), OKABE_ITO["orange"]),
        ("Marker-only+Thr", Path("data/predictions/marker_only_thsel/test/metrics.json"), OKABE_ITO["vermillion"]),
    ]

    labels, colors, metrics = [], [], []
    for label, p, c in model_spec:
        if not p.exists():
            continue
        d = _load_json(p)["overall"]
        labels.append(label)
        colors.append(c)
        metrics.append(d)
    if not metrics:
        raise FileNotFoundError("No E1 test metrics found.")

    metric_keys = [("ap50", "AP@0.5"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (k, title) in zip(axes, metric_keys):
        vals = [float(m[k]) for m in metrics]
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right")

    # Key gain annotation: Generic FT vs Baseline on AP50.
    base_idx = labels.index("Baseline") if "Baseline" in labels else None
    gen_idx = labels.index("Generic FT") if "Generic FT" in labels else None
    if base_idx is not None and gen_idx is not None:
        ap_base = float(metrics[base_idx]["ap50"])
        ap_gen = float(metrics[gen_idx]["ap50"])
        gain = ap_gen - ap_base
        fold = ap_gen / max(ap_base, 1e-8)
        axes[0].annotate(
            f"+{gain:.3f} ({fold:.1f}x)",
            xy=(gen_idx, ap_gen),
            xytext=(gen_idx + 0.4, min(0.95, ap_gen + 0.20)),
            arrowprops={"arrowstyle": "->", "color": OKABE_ITO["blue"], "lw": 1.2},
            color=OKABE_ITO["blue"],
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle("E1 Segmentation Performance on Test Set", y=1.02, fontsize=14, fontweight="bold")
    _save(fig, out_dir / "fig_e1_overall_test")


def plot_e1_seed_stability(out_dir: Path):
    p = Path("data/reports/e1_seed_sweep_generic.tsv")
    if not p.exists():
        return
    rows = _read_tsv(p)
    seeds = [int(r["seed"]) for r in rows]
    ap = [float(r["ap50"]) for r in rows]
    f1 = [float(r["f1"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=False)
    for ax, vals, title, color in [
        (axes[0], ap, "AP@0.5", OKABE_ITO["blue"]),
        (axes[1], f1, "F1", OKABE_ITO["green"]),
    ]:
        x = np.arange(len(vals))
        ax.scatter(x, vals, s=55, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        ax.axhline(mean, color=color, linestyle="--", linewidth=1.4, alpha=0.9)
        ax.fill_between([-0.4, len(vals) - 0.6], mean - std, mean + std, color=color, alpha=0.15)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.set_title(title)
        ax.set_xlabel("Seed")
        ax.set_ylim(min(vals) - 0.02, max(vals) + 0.03)
    axes[0].set_ylabel("Score")
    fig.suptitle("E1.6 Generic Fine-Tune Seed Stability", y=1.03, fontsize=13, fontweight="bold")
    _save(fig, out_dir / "fig_e1_seed_stability")


def plot_e2_label_efficiency(out_dir: Path, max_budget: int | None = 17):
    p = Path("data/logs/e2_active_sampling_compare/records.tsv")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    rows = _read_tsv(p)
    if max_budget is not None:
        rows = [r for r in rows if int(float(r["budget"])) <= int(max_budget)]
    if not rows:
        raise RuntimeError("No E2.2 rows after budget filtering.")

    budgets = sorted({int(float(r["budget"])) for r in rows})
    strategies = ["random", "hard"]

    by_budget_train_n = {}
    for b in budgets:
        vals = [_safe_int(r.get("train_n")) for r in rows if int(float(r["budget"])) == b]
        vals = [v for v in vals if v is not None]
        if vals:
            by_budget_train_n[b] = int(round(float(np.median(vals))))

    def collect(metric: str, strat: str, field: str):
        vals = []
        for b in budgets:
            arr = [
                _safe_float(r.get(metric))
                for r in rows
                if int(float(r["budget"])) == b and str(r["strategy"]).lower() == strat
            ]
            arr = [v for v in arr if v is not None]
            if not arr:
                raise RuntimeError(f"Missing rows for budget={b}, strategy={strat}, metric={metric}")
            if field == "mean":
                vals.append(float(np.mean(arr)))
            elif field == "min":
                vals.append(float(np.min(arr)))
            elif field == "max":
                vals.append(float(np.max(arr)))
            else:
                raise ValueError(field)
        return np.asarray(vals, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True)
    for ax, metric, title in [
        (axes[0], "overall_ap50", "AP@0.5"),
        (axes[1], "overall_f1", "F1"),
    ]:
        for strat, color in [("random", OKABE_ITO["blue"]), ("hard", OKABE_ITO["orange"])]:
            mean = collect(metric, strat, "mean")
            vmin = collect(metric, strat, "min")
            vmax = collect(metric, strat, "max")
            ax.plot(
                budgets,
                mean,
                marker="o",
                markersize=6,
                linewidth=2.0,
                color=color,
                label=strat.capitalize(),
            )
            ax.fill_between(budgets, vmin, vmax, color=color, alpha=0.15, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Budget per Marker")
        ax.set_ylabel("Validation Score")
        ax.set_ylim(0.35, 0.90)
        ax.set_xticks(budgets)
        xticklabels = []
        for b in budgets:
            n = by_budget_train_n.get(b)
            xticklabels.append(f"{b}\nN={n}" if n is not None else str(b))
        ax.set_xticklabels(xticklabels)
        ax.legend(frameon=False, loc="lower right")

    # annotate hard-random AP deltas
    deltas = list(collect("overall_ap50", "hard", "mean") - collect("overall_ap50", "random", "mean"))
    for b, dv in zip(budgets, deltas):
        axes[0].text(
            b,
            0.885,
            f"{dv:+.3f}",
            ha="center",
            va="top",
            fontsize=8,
            color=OKABE_ITO["black"],
        )

    fig.suptitle(
        "Label Efficiency: Random vs Hard-First Sampling",
        y=1.03,
        fontsize=13,
        fontweight="bold",
    )
    _save(fig, out_dir / "fig_e2_label_efficiency")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/reports/fig_pub")
    ap.add_argument("--e2-max-budget", type=int, default=17)
    return ap.parse_args()


def main():
    args = parse_args()
    _set_pub_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_e1_overall(out_dir)
    plot_e1_seed_stability(out_dir)
    plot_e2_label_efficiency(out_dir, max_budget=args.e2_max_budget)

    print(f"[fig] saved to {out_dir}")
    print("[fig] files:")
    for p in sorted(out_dir.glob("*")):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
