#!/usr/bin/env python
"""
Purpose: Run E5.4 marker-positive threshold sensitivity analysis.
Output: threshold-wise donor ratios, correlations, and bootstrap stability tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import tifffile as tiff
import yaml
from cellpose import models
from scipy.stats import spearmanr

OUTCOMES_DEFAULT = ["braaksc", "ceradsc", "cogdx", "dcfdx_lv", "plaq_d", "plaq_n", "nft", "gpath"]
MARKERS = {"gfap", "iba1", "neun", "olig2", "pecam"}


def _canonical_pair_key(path: Path, channel: int):
    name = path.name
    pat = re.compile(rf"(?:_[A-Za-z0-9]+-)?b0c{channel}", flags=re.IGNORECASE)
    if pat.search(name) is None:
        return None
    core = pat.sub("_CHAN", name, count=1).lower()
    return f"{str(path.parent).lower()}::{core}"


def _norm_id(v: str) -> str:
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _safe_float(v):
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def _fdr_bh(pvals):
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    qvals = [1.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = order[-rank]
        q = min(prev, pvals[i] * n / (n - rank + 1))
        qvals[i] = q
        prev = q
    return qvals


def _discover_tiles(raw_tiles_dir: Path):
    tif_files = [p for p in raw_tiles_dir.rglob("*.tif*") if p.suffix.lower() in {".tif", ".tiff"}]
    c1_by_key = {}
    c2_by_key = {}
    c0_files = []
    for p in tif_files:
        k1 = _canonical_pair_key(p, channel=1)
        if k1 is not None:
            c1_by_key[k1] = p
        k2 = _canonical_pair_key(p, channel=2)
        if k2 is not None:
            c2_by_key[k2] = p
        k0 = _canonical_pair_key(p, channel=0)
        if k0 is not None:
            c0_files.append(p)
    pairs = []
    for c0 in sorted(c0_files):
        k0 = _canonical_pair_key(c0, channel=0)
        if k0 is None:
            continue
        c1 = c1_by_key.get(k0)
        if c1 is None:
            c1 = c2_by_key.get(k0)
        if c1 is None:
            continue
        pairs.append((c0, c1))
    return pairs


def _infer_marker_and_donor(path: Path):
    marker = "unknown"
    for p in [x.lower() for x in path.parts]:
        if p in MARKERS:
            marker = p
            break
    donor = "unknown"
    for p in path.parts:
        m = re.search(r"(\d{6,12})", p)
        if m:
            donor = m.group(1)
            break
    return marker, donor


def _load_3ch(c0_path: Path, c1_path: Path):
    dapi = tiff.imread(c0_path)
    marker = tiff.imread(c1_path)
    if dapi.ndim == 3:
        dapi = dapi[..., 0]
    if marker.ndim == 3:
        marker = marker[..., 0]
    z = np.zeros_like(dapi)
    return np.stack([dapi, marker, z], axis=-1)


def _read_table(path: Path):
    lines = path.read_text().splitlines()
    delim = "\t" if lines and "\t" in lines[0] else ","
    with open(path, "r", newline="") as f:
        rr = csv.DictReader(f, delimiter=delim)
        rows = [x for x in rr]
        return rr.fieldnames or [], rows


def _aggregate_donor(tile_rows):
    agg = {}
    for r in tile_rows:
        key = (r["model"], r["q"], r["donor"], r["marker"])
        if key not in agg:
            agg[key] = {"n_total_cells": 0, "n_marker_positive": 0}
        agg[key]["n_total_cells"] += int(r["n_total_cells"])
        agg[key]["n_marker_positive"] += int(r["n_marker_positive"])
    out = []
    for (model, q, donor, marker), v in agg.items():
        total = v["n_total_cells"]
        pos = v["n_marker_positive"]
        out.append(
            {
                "model": model,
                "q": float(q),
                "donor": donor,
                "marker": marker,
                "n_total_cells": total,
                "n_marker_positive": pos,
                "ratio": (pos / total) if total > 0 else 0.0,
            }
        )
    return out


def _bootstrap_tile_stability(tile_rows, clinical_by_donor, outcome: str, n_bootstrap: int, seed: int):
    by_donor = {}
    for r in tile_rows:
        donor = _norm_id(r["donor"])
        by_donor.setdefault(donor, []).append((int(r["n_marker_positive"]), int(r["n_total_cells"])))
    rng = random.Random(seed)
    rhos, pvals = [], []
    for _ in range(n_bootstrap):
        samples = []
        for donor, tiles in by_donor.items():
            if donor not in clinical_by_donor:
                continue
            k = max(1, int(math.ceil(len(tiles) * 0.8)))
            idxs = [rng.randrange(0, len(tiles)) for _ in range(k)]
            pos = sum(tiles[i][0] for i in idxs)
            total = sum(tiles[i][1] for i in idxs)
            if total <= 0:
                continue
            y = _safe_float(clinical_by_donor[donor].get(outcome))
            if y is None:
                continue
            samples.append((pos / total, y))
        if len(samples) < 3:
            continue
        x = [a for a, _ in samples]
        y = [b for _, b in samples]
        rho, p = spearmanr(x, y)
        if rho is None or p is None or math.isnan(rho) or math.isnan(p):
            continue
        rhos.append(float(rho))
        pvals.append(float(p))
    if len(rhos) == 0:
        return None, None, None
    mean_rho = sum(rhos) / len(rhos)
    var_rho = sum((r - mean_rho) ** 2 for r in rhos) / len(rhos)
    sign_consistency = max(sum(1 for r in rhos if r > 0), sum(1 for r in rhos if r < 0)) / len(rhos)
    p_pass = sum(1 for p in pvals if p < 0.05) / len(pvals)
    return var_rho, sign_consistency, p_pass


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--baseline-model", default="cpsam")
    ap.add_argument("--finetuned-model", default="generic_cpsam")
    ap.add_argument("--percentiles", nargs="+", type=float, default=[60, 70, 75, 80, 90])
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2024)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    raw_tiles_dir = Path(cfg["raw_tiles_dir"])
    reports_dir = Path(cfg["reports_dir"])
    downstream_dir = Path(cfg["downstream_dir"])
    clinical_csv = Path(cfg["data_root"]) / "ROSMAP_clinical_n69.csv"
    ckpt = Path(cfg["checkpoints_dir"]) / args.finetuned_model / "models" / args.finetuned_model
    finetuned_spec = str(ckpt) if ckpt.exists() else args.finetuned_model

    reports_dir.mkdir(parents=True, exist_ok=True)
    downstream_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_tiles(raw_tiles_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No tile pairs found in {raw_tiles_dir}")

    model_specs = {"baseline": args.baseline_model, "finetuned": finetuned_spec}
    q_list = [float(q) for q in args.percentiles]
    tile_rows = []

    for model_name, model_spec in model_specs.items():
        model_obj = models.CellposeModel(gpu=True, pretrained_model=model_spec)
        for c0, c1 in pairs:
            marker, donor = _infer_marker_and_donor(c0)
            if marker == "unknown" or donor == "unknown":
                continue
            try:
                img = _load_3ch(c0, c1)
            except Exception:
                continue
            pred, _, _ = model_obj.eval(
                img,
                channels=[1, 2],
                diameter=None,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
            )
            pred = pred.astype(np.int32)
            ids = np.unique(pred)
            ids = ids[ids > 0]
            n_total = int(len(ids))
            marker_ch = img[..., 1]
            if n_total == 0:
                means = np.array([], dtype=float)
            else:
                means = np.array([float(marker_ch[pred == oid].mean()) for oid in ids], dtype=float)
            for q in q_list:
                thr = float(np.percentile(marker_ch, q))
                n_pos = int((means > thr).sum()) if n_total > 0 else 0
                tile_rows.append(
                    {
                        "model": model_name,
                        "q": q,
                        "tile_c0": str(c0),
                        "tile_c1": str(c1),
                        "donor": donor,
                        "marker": marker,
                        "n_total_cells": n_total,
                        "n_marker_positive": n_pos,
                        "ratio": (n_pos / n_total) if n_total > 0 else 0.0,
                    }
                )
        print(f"[e5.4] done model={model_name}")

    tile_out = reports_dir / "e5_threshold_sensitivity_tile_counts.tsv"
    with open(tile_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tile_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(tile_rows)

    donor_rows = _aggregate_donor(tile_rows)
    donor_out = reports_dir / "e5_threshold_sensitivity_donor_ratio.tsv"
    with open(donor_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(donor_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(donor_rows)

    c_cols, c_rows = _read_table(clinical_csv)
    donor_col = "projid" if "projid" in c_cols else c_cols[0]
    clinical_by_donor = {_norm_id(r[donor_col]): r for r in c_rows if _norm_id(r.get(donor_col, "")) != ""}
    outcomes = [o for o in OUTCOMES_DEFAULT if o in c_cols]

    corr_rows = []
    boot_rows = []
    for model in ["baseline", "finetuned"]:
        for marker in sorted(MARKERS):
            for q in q_list:
                dm = [r for r in donor_rows if r["model"] == model and r["marker"] == marker and float(r["q"]) == q]
                tm = [r for r in tile_rows if r["model"] == model and r["marker"] == marker and float(r["q"]) == q]
                for out in outcomes:
                    x, y = [], []
                    for r in dm:
                        donor = _norm_id(r["donor"])
                        if donor not in clinical_by_donor:
                            continue
                        yy = _safe_float(clinical_by_donor[donor].get(out))
                        if yy is None:
                            continue
                        x.append(float(r["ratio"]))
                        y.append(yy)
                    if len(x) >= 3:
                        rho, p = spearmanr(x, y)
                        rho = float(rho)
                        p = float(p)
                    else:
                        rho, p = None, None
                    corr_rows.append(
                        {
                            "model": model,
                            "marker": marker,
                            "q": q,
                            "outcome": out,
                            "n": len(x),
                            "rho": rho,
                            "pval": p,
                        }
                    )
                    var_rho, sign_consistency, p_pass = _bootstrap_tile_stability(
                        tm, clinical_by_donor, outcome=out, n_bootstrap=args.bootstrap, seed=args.seed
                    )
                    boot_rows.append(
                        {
                            "model": model,
                            "marker": marker,
                            "q": q,
                            "outcome": out,
                            "var_rho": var_rho,
                            "sign_consistency": sign_consistency,
                            "p_pass": p_pass,
                        }
                    )

    idx = [i for i, r in enumerate(corr_rows) if r["pval"] is not None]
    pvals = [corr_rows[i]["pval"] for i in idx]
    qvals = _fdr_bh(pvals) if len(pvals) > 0 else []
    for i, qv in zip(idx, qvals):
        corr_rows[i]["fdr"] = qv
        corr_rows[i]["pass_fdr"] = int(qv < 0.05)
    for r in corr_rows:
        if "fdr" not in r:
            r["fdr"] = None
            r["pass_fdr"] = 0

    corr_out = reports_dir / "e5_threshold_sensitivity.tsv"
    with open(corr_out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "marker", "q", "outcome", "n", "rho", "pval", "fdr", "pass_fdr"],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(corr_rows)

    boot_out = reports_dir / "e5_threshold_sensitivity_bootstrap.tsv"
    with open(boot_out, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "marker", "q", "outcome", "var_rho", "sign_consistency", "p_pass"],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(boot_rows)

    summary = {
        "raw_tiles_dir": str(raw_tiles_dir),
        "clinical_csv": str(clinical_csv),
        "models": model_specs,
        "percentiles": q_list,
        "outcomes": outcomes,
        "n_tile_pairs": len(pairs),
        "n_bootstrap": args.bootstrap,
        "outputs": {
            "tile_counts": str(tile_out),
            "donor_ratio": str(donor_out),
            "spearman": str(corr_out),
            "bootstrap": str(boot_out),
        },
    }
    with open(reports_dir / "e5_threshold_sensitivity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[e5.4] saved {corr_out}")


if __name__ == "__main__":
    main()
