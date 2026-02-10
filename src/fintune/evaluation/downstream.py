"""
Downstream analysis for baseline vs finetuned ratio comparison.

Inputs are expected to come from `run_full_inference` outputs under `finetune/data`.
No hardcoded external dataset paths are used.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import random
from typing import Dict, List, Tuple

import yaml
from scipy.stats import spearmanr

OUTCOMES_DEFAULT = ["braaksc", "ceradsc", "cogdx", "dcfdx_lv", "plaq_d", "plaq_n", "nft", "gpath"]


def _read_table(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    text = path.read_text().splitlines()
    if len(text) == 0:
        return [], []
    delim = "\t" if "\t" in text[0] else ","
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter=delim)
        rows = [x for x in r]
        return r.fieldnames or [], rows


def _safe_float(v):
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except Exception:
        return None


def _norm_id(v: str) -> str:
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _fdr_bh(pvals: List[float]) -> List[float]:
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    out = [1.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = order[-rank]
        p = pvals[i]
        q = min(prev, p * n / (n - rank + 1))
        out[i] = q
        prev = q
    return out


def _ensure_local_path(p: Path, repo_root: Path):
    if not str(p).startswith(str(repo_root)):
        raise ValueError(f"Path must be under repository root: {p}")


def _aggregate_donor_from_tiles(tile_rows: List[Dict[str, str]]):
    agg = {}
    for r in tile_rows:
        model = str(r.get("model", "")).strip()
        donor = _norm_id(r.get("donor", ""))
        marker = str(r.get("marker", "")).strip().lower()
        n_total = _safe_float(r.get("n_total_cells"))
        n_pos = _safe_float(r.get("n_marker_positive"))
        if model == "" or donor == "" or marker == "" or n_total is None or n_pos is None or n_total <= 0:
            continue
        key = (model, donor, marker)
        if key not in agg:
            agg[key] = {"n_total_cells": 0.0, "n_marker_positive": 0.0}
        agg[key]["n_total_cells"] += n_total
        agg[key]["n_marker_positive"] += n_pos

    out = []
    for (model, donor, marker), v in agg.items():
        ratio = v["n_marker_positive"] / v["n_total_cells"] if v["n_total_cells"] > 0 else 0.0
        out.append(
            {
                "model": model,
                "donor": donor,
                "marker": marker,
                "n_total_cells": int(v["n_total_cells"]),
                "n_marker_positive": int(v["n_marker_positive"]),
                "ratio": ratio,
            }
        )
    return out


def _correlate(donor_rows, outcome: str):
    x, y = [], []
    for r in donor_rows:
        ratio = _safe_float(r.get("ratio"))
        o = _safe_float(r.get(outcome))
        if ratio is None or o is None:
            continue
        x.append(ratio)
        y.append(o)
    if len(x) < 3:
        return {"n": len(x), "rho": None, "pval": None}
    rho, p = spearmanr(x, y)
    return {"n": len(x), "rho": float(rho), "pval": float(p)}


def _bootstrap_tile_stability(tile_rows, clinical_by_donor, outcome: str, n_bootstrap: int, seed: int):
    # Tile-level bootstrap: resample 80% tiles per donor, then compute donor ratio -> Spearman.
    by_donor = {}
    for r in tile_rows:
        donor = _norm_id(r.get("donor", ""))
        n_total = _safe_float(r.get("n_total_cells"))
        n_pos = _safe_float(r.get("n_marker_positive"))
        if donor == "" or n_total is None or n_pos is None or n_total <= 0:
            continue
        by_donor.setdefault(donor, []).append((n_pos, n_total))

    rng = random.Random(seed)
    rhos = []
    pvals = []
    for _ in range(n_bootstrap):
        donor_ratio = []
        for donor, tiles in by_donor.items():
            if donor not in clinical_by_donor:
                continue
            k = max(1, int(math.ceil(len(tiles) * 0.8)))
            idxs = [rng.randrange(0, len(tiles)) for _ in range(k)]
            pos = sum(tiles[i][0] for i in idxs)
            total = sum(tiles[i][1] for i in idxs)
            if total <= 0:
                continue
            o = _safe_float(clinical_by_donor[donor].get(outcome))
            if o is None:
                continue
            donor_ratio.append((pos / total, o))
        if len(donor_ratio) < 3:
            continue
        x = [a for a, _ in donor_ratio]
        y = [b for _, b in donor_ratio]
        rho, p = spearmanr(x, y)
        if rho is None or p is None or math.isnan(rho) or math.isnan(p):
            continue
        rhos.append(float(rho))
        pvals.append(float(p))

    if len(rhos) == 0:
        return {"var_rho": None, "sign_consistency": None, "p_pass": None}
    mean_rho = sum(rhos) / len(rhos)
    var_rho = sum((r - mean_rho) ** 2 for r in rhos) / len(rhos)
    pos = sum(1 for r in rhos if r > 0)
    neg = sum(1 for r in rhos if r < 0)
    sign_consistency = max(pos, neg) / len(rhos)
    p_pass = sum(1 for p in pvals if p < 0.05) / len(pvals)
    return {"var_rho": var_rho, "sign_consistency": sign_consistency, "p_pass": p_pass}


def run_downstream_analysis(
    config_path: str,
    pred_dir: str,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    baseline_tile_csv: str | None = None,
    finetuned_tile_csv: str | None = None,
    clinical_csv: str | None = None,
    seed: int = 2024,
):
    del pred_dir  # compatibility

    cfg = yaml.safe_load(Path(config_path).read_text())
    repo_root = Path(config_path).resolve().parents[1]
    downstream_dir = Path(cfg["downstream_dir"])
    downstream_dir.mkdir(parents=True, exist_ok=True)
    _ensure_local_path(downstream_dir, repo_root)

    fi_dir = downstream_dir / "full_inference"
    if baseline_tile_csv is None:
        baseline_tile_csv = str(fi_dir / "baseline" / "tile_counts.tsv")
    if finetuned_tile_csv is None:
        finetuned_tile_csv = str(fi_dir / "finetuned" / "tile_counts.tsv")
    if clinical_csv is None:
        clinical_csv = str(Path(cfg["data_root"]) / "ROSMAP_clinical_n69.csv")

    baseline_path = Path(baseline_tile_csv)
    finetuned_path = Path(finetuned_tile_csv)
    clinical_path = Path(clinical_csv)
    for p in [baseline_path, finetuned_path, clinical_path]:
        _ensure_local_path(p, repo_root)
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    c_cols, c_rows = _read_table(clinical_path)
    donor_col = "projid" if "projid" in c_cols else c_cols[0]
    clinical_by_donor = {_norm_id(r[donor_col]): r for r in c_rows if _norm_id(r.get(donor_col, "")) != ""}

    _, b_tile_rows = _read_table(baseline_path)
    _, f_tile_rows = _read_table(finetuned_path)
    donor_rows = _aggregate_donor_from_tiles(b_tile_rows) + _aggregate_donor_from_tiles(f_tile_rows)

    donor_out = downstream_dir / "e5_donor_marker_ratio_from_tiles.tsv"
    with open(donor_out, "w", newline="") as f:
        fields = ["model", "donor", "marker", "n_total_cells", "n_marker_positive", "ratio"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(donor_rows)

    outcomes = [o for o in OUTCOMES_DEFAULT if o in c_cols]
    corr_rows = []
    boot_rows = []
    for model in ["baseline", "finetuned"]:
        for marker in ["gfap", "iba1", "neun", "olig2", "pecam"]:
            dm = [r for r in donor_rows if r["model"] == model and r["marker"] == marker]
            merged = []
            for r in dm:
                d = _norm_id(r["donor"])
                if d in clinical_by_donor:
                    merged.append({**r, **clinical_by_donor[d]})
            tm = [r for r in (b_tile_rows if model == "baseline" else f_tile_rows) if str(r.get("marker", "")).lower() == marker]
            for out in outcomes:
                cor = _correlate(merged, outcome=out)
                corr_rows.append(
                    {
                        "model": model,
                        "marker": marker,
                        "outcome": out,
                        "n": cor["n"],
                        "rho": cor["rho"],
                        "pval": cor["pval"],
                    }
                )
                bs = _bootstrap_tile_stability(
                    tile_rows=tm,
                    clinical_by_donor=clinical_by_donor,
                    outcome=out,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                boot_rows.append(
                    {
                        "model": model,
                        "marker": marker,
                        "outcome": out,
                        "var_rho": bs["var_rho"],
                        "sign_consistency": bs["sign_consistency"],
                        "p_pass": bs["p_pass"],
                    }
                )

    idx = [i for i, r in enumerate(corr_rows) if r["pval"] is not None]
    pvals = [corr_rows[i]["pval"] for i in idx]
    qvals = _fdr_bh(pvals) if len(pvals) > 0 else []
    for i, q in zip(idx, qvals):
        corr_rows[i]["fdr"] = q
        corr_rows[i]["pass_fdr"] = int(q < alpha)
    for i, r in enumerate(corr_rows):
        if "fdr" not in r:
            corr_rows[i]["fdr"] = None
            corr_rows[i]["pass_fdr"] = 0

    corr_out = downstream_dir / "e5_spearman.tsv"
    with open(corr_out, "w", newline="") as f:
        fields = ["model", "marker", "outcome", "n", "rho", "pval", "fdr", "pass_fdr"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(corr_rows)

    boot_out = downstream_dir / "e5_bootstrap_stability.tsv"
    with open(boot_out, "w", newline="") as f:
        fields = ["model", "marker", "outcome", "var_rho", "sign_consistency", "p_pass"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(boot_rows)

    summary = {
        "baseline_tile_csv": str(baseline_path),
        "finetuned_tile_csv": str(finetuned_path),
        "clinical_csv": str(clinical_path),
        "outcomes": outcomes,
        "n_corr_rows": len(corr_rows),
        "n_boot_rows": len(boot_rows),
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "outputs": {
            "donor_marker_ratio": str(donor_out),
            "spearman": str(corr_out),
            "bootstrap_stability": str(boot_out),
        },
    }
    with open(downstream_dir / "e5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[e5] donor={donor_out}")
    print(f"[e5] spearman={corr_out}")
    print(f"[e5] bootstrap={boot_out}")


__all__ = ["run_downstream_analysis"]

