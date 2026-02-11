#!/usr/bin/env python
"""
Purpose: Plot generic model performance vs label budget from E2 records.
Output: publication-style 300-dpi PDF/PNG curves for overall and per-marker metrics.
"""
from __future__ import annotations

import argparse
import csv
import math
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


def _read_tsv(path: Path):
    with open(path, "r", newline="") as f:
        rr = csv.DictReader(f, delimiter="\t")
        return [r for r in rr]


def _safe_float(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _agg_by_budget(rows, key):
    by = {}
    for r in rows:
        b = int(float(r["budget"]))
        v = _safe_float(r.get(key))
        if v is None:
            continue
        by.setdefault(b, []).append(v)
    budgets = sorted(by.keys())
    means = np.array([float(np.mean(by[b])) for b in budgets], dtype=float)
    stds = np.array([float(np.std(by[b])) for b in budgets], dtype=float)
    mins = np.array([float(np.min(by[b])) for b in budgets], dtype=float)
    maxs = np.array([float(np.max(by[b])) for b in budgets], dtype=float)
    return budgets, means, stds, mins, maxs


def _budget_to_effective_n(rows):
    by = {}
    for r in rows:
        try:
            b = int(float(r["budget"]))
            n = int(float(r.get("train_n", "")))
        except Exception:
            continue
        by.setdefault(b, []).append(n)
    mapping = {}
    for b, vals in by.items():
        if not vals:
            continue
        mapping[b] = int(round(float(np.median(vals))))
    return mapping


def _save(fig, out_base: Path):
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_generic_budget(records_tsv: Path, out_dir: Path, title_suffix: str = "", max_budget: int | None = None):
    rows = _read_tsv(records_tsv)
    if max_budget is not None:
        rows = [r for r in rows if int(float(r["budget"])) <= int(max_budget)]
    if len(rows) == 0:
        raise RuntimeError(f"No rows in {records_tsv}")
    budget_to_n = _budget_to_effective_n(rows)

    # left: overall AP50 / F1
    b_ap, m_ap, s_ap, n_ap, x_ap = _agg_by_budget(rows, "overall_ap50")
    b_f1, m_f1, s_f1, n_f1, x_f1 = _agg_by_budget(rows, "overall_f1")

    if b_ap != b_f1:
        raise RuntimeError("Budget sets mismatch between AP50 and F1.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=False)

    ax = axes[0]
    budgets = b_ap
    ax.plot(
        budgets,
        m_ap,
        marker="o",
        markersize=6,
        linewidth=2.2,
        color=OKABE_ITO["blue"],
        label="AP@0.5",
    )
    ax.fill_between(budgets, m_ap - s_ap, m_ap + s_ap, color=OKABE_ITO["blue"], alpha=0.18, linewidth=0)

    ax.plot(
        budgets,
        m_f1,
        marker="s",
        markersize=5.5,
        linewidth=2.0,
        color=OKABE_ITO["green"],
        label="F1",
    )
    ax.fill_between(budgets, m_f1 - s_f1, m_f1 + s_f1, color=OKABE_ITO["green"], alpha=0.16, linewidth=0)

    for b, v in zip(budgets, m_ap):
        ax.text(b, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=OKABE_ITO["blue"])
    ax.set_title("Overall Performance vs Budget")
    ax.set_xlabel("Requested Budget per Marker")
    ax.set_ylabel("Validation Score")
    ax.set_xticks(budgets)
    xticklabels = []
    for b in budgets:
        n_eff = budget_to_n.get(b)
        if n_eff is None:
            xticklabels.append(str(b))
        else:
            xticklabels.append(f"{b}\nN={n_eff}")
    ax.set_xticklabels(xticklabels)
    ymin = min(np.min(m_ap - s_ap), np.min(m_f1 - s_f1)) - 0.03
    ymax = min(0.98, max(np.max(m_ap + s_ap), np.max(m_f1 + s_f1)) + 0.06)
    ax.set_ylim(max(0.0, ymin), max(0.15, ymax))
    ax.legend(frameon=False, loc="upper left")

    # right: per-marker AP50
    marker_cfg = [
        ("gfap_ap50", "GFAP", OKABE_ITO["vermillion"]),
        ("iba1_ap50", "IBA1", OKABE_ITO["orange"]),
        ("neun_ap50", "NeuN", OKABE_ITO["purple"]),
        ("olig2_ap50", "OLIG2", OKABE_ITO["sky"]),
    ]
    ax = axes[1]
    ymins, ymaxs = [], []
    for key, label, color in marker_cfg:
        b, m, s, _, _ = _agg_by_budget(rows, key)
        if len(b) == 0:
            continue
        ax.plot(b, m, marker="o", markersize=5, linewidth=1.9, color=color, label=label)
        ax.fill_between(b, m - s, m + s, color=color, alpha=0.13, linewidth=0)
        ymins.append(np.min(m - s))
        ymaxs.append(np.max(m + s))

    ax.set_title("Per-Marker AP@0.5 vs Budget")
    ax.set_xlabel("Requested Budget per Marker")
    ax.set_ylabel("Validation AP@0.5")
    ax.set_xticks(budgets)
    ax.set_xticklabels(xticklabels)
    if ymins and ymaxs:
        ax.set_ylim(max(0.0, min(ymins) - 0.03), min(1.0, max(ymaxs) + 0.06))
    ax.legend(frameon=False, loc="upper left")

    main_title = "Generic Model Label-Efficiency Curve"
    if title_suffix.strip():
        main_title += f" ({title_suffix})"
    fig.suptitle(main_title, y=1.03, fontsize=14, fontweight="bold")

    out_base = out_dir / "fig_generic_budget_curve"
    _save(fig, out_base)

    # Also save a copy with explicit name highlighting effective N annotation.
    out_base_effective = out_dir / "fig_generic_budget_curve_with_effective_n"
    fig2, ax2 = plt.subplots(1, 1, figsize=(6.8, 4.4))
    ax2.plot(budgets, m_ap, marker="o", linewidth=2.2, color=OKABE_ITO["blue"], label="AP@0.5")
    ax2.fill_between(budgets, m_ap - s_ap, m_ap + s_ap, color=OKABE_ITO["blue"], alpha=0.16, linewidth=0)
    ax2.plot(budgets, m_f1, marker="s", linewidth=2.0, color=OKABE_ITO["green"], label="F1")
    ax2.fill_between(budgets, m_f1 - s_f1, m_f1 + s_f1, color=OKABE_ITO["green"], alpha=0.14, linewidth=0)
    ax2.set_xlabel("Requested Budget per Marker")
    ax2.set_ylabel("Validation Score")
    ax2.set_xticks(budgets)
    ax2.set_xticklabels(xticklabels)
    ax2.set_ylim(max(0.0, ymin), max(0.15, ymax))
    ax2.set_title("Overall Budget Curve (Tick Label: Budget / Effective N)")
    ax2.legend(frameon=False, loc="upper left")
    _save(fig2, out_base_effective)

    # Save simple summary table for manuscript text.
    out_tsv = out_dir / "fig_generic_budget_curve_summary.tsv"
    with open(out_tsv, "w", newline="") as f:
        f.write("budget\toverall_ap50_mean\toverall_ap50_std\toverall_f1_mean\toverall_f1_std\n")
        for i, b in enumerate(budgets):
            f.write(f"{b}\t{m_ap[i]:.6f}\t{s_ap[i]:.6f}\t{m_f1[i]:.6f}\t{s_f1[i]:.6f}\n")

    print(f"[fig] saved {out_base.with_suffix('.pdf')}")
    print(f"[fig] saved {out_base.with_suffix('.png')}")
    print(f"[fig] saved {out_base_effective.with_suffix('.pdf')}")
    print(f"[fig] saved {out_base_effective.with_suffix('.png')}")
    print(f"[fig] saved {out_tsv}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--records-tsv",
        default="data/logs/budget_curve_long_e6/records.tsv",
        help="E2 records.tsv containing budget-repeat metrics.",
    )
    ap.add_argument("--out-dir", default="data/reports/fig_pub")
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--max-budget", type=int, default=None, help="Optional cap to plot budgets <= this value.")
    return ap.parse_args()


def main():
    args = parse_args()
    _set_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_generic_budget(
        Path(args.records_tsv),
        out_dir=out_dir,
        title_suffix=args.title_suffix,
        max_budget=args.max_budget,
    )


if __name__ == "__main__":
    main()
