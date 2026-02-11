"""
E5.5 adjusted association models with donor-level covariate adjustment.

Input files are generated within this repository:
- full inference tile counts (baseline + finetuned)
- clinical summary csv under data/
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random

import numpy as np
from scipy import stats
import yaml

OUTCOMES_DEFAULT = ["braaksc", "ceradsc", "cogdx", "dcfdx_lv", "plaq_d", "plaq_n", "nft", "gpath"]


def _read_table(path: Path):
    lines = path.read_text().splitlines()
    delim = "\t" if lines and "\t" in lines[0] else ","
    with open(path, "r", newline="") as f:
        rr = csv.DictReader(f, delimiter=delim)
        rows = [x for x in rr]
        return rr.fieldnames or [], rows


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


def _ensure_local_path(p: Path, repo_root: Path) -> Path:
    root = repo_root.resolve()
    rp = p.expanduser()
    rp = (root / rp) if not rp.is_absolute() else rp
    rp = Path(str(rp))
    root_s = str(root)
    rp_s = str(rp)
    if not (rp_s == root_s or rp_s.startswith(root_s + "/")):
        raise ValueError(f"Path must be under repository root: {p}")
    return rp


def _aggregate_donor_from_tiles(tile_rows):
    agg = {}
    for r in tile_rows:
        model = str(r.get("model", "")).strip()
        donor = _norm_id(r.get("donor", ""))
        marker = str(r.get("marker", "")).strip().lower()
        n_total = _safe_float(r.get("n_total_cells"))
        n_pos = _safe_float(r.get("n_marker_positive"))
        area = _safe_float(r.get("area_px"))
        if model == "" or donor == "" or marker == "" or n_total is None or n_pos is None:
            continue
        key = (model, donor, marker)
        if key not in agg:
            agg[key] = {"n_total_cells": 0.0, "n_marker_positive": 0.0, "area_px": 0.0}
        agg[key]["n_total_cells"] += n_total
        agg[key]["n_marker_positive"] += n_pos
        if area is not None and area > 0:
            agg[key]["area_px"] += area

    out = []
    for (model, donor, marker), v in agg.items():
        total = v["n_total_cells"]
        pos = v["n_marker_positive"]
        area = v["area_px"]
        ratio = (pos / total) if total > 0 else None
        total_density_mpx = (total / area) * 1e6 if area > 0 else None
        pos_density_mpx = (pos / area) * 1e6 if area > 0 else None
        out.append(
            {
                "model": model,
                "donor": donor,
                "marker": marker,
                "n_total_cells": int(total),
                "n_marker_positive": int(pos),
                "area_px": int(area) if area > 0 else 0,
                "ratio": ratio,
                "total_density_mpx": total_density_mpx,
                "pos_density_mpx": pos_density_mpx,
            }
        )
    return out


def _to_numeric_vector(values):
    nums = [_safe_float(v) for v in values]
    if all(v is not None for v in nums):
        return nums, "numeric"
    cats = []
    for v in values:
        s = str(v).strip()
        if s == "" or s.lower() == "nan":
            cats.append(None)
        else:
            cats.append(s)
    uniq = sorted(set(x for x in cats if x is not None))
    if len(uniq) == 0:
        return [None for _ in values], "empty"
    mapping = {u: float(i) for i, u in enumerate(uniq)}
    out = [mapping.get(x) if x is not None else None for x in cats]
    return out, "categorical"


def _rank_ols(y, x, covs):
    n = len(y)
    if n < 4:
        return None

    cols = [np.asarray(x, dtype=float)]
    for c in covs:
        cols.append(np.asarray(c, dtype=float))

    # Rank-transform for robust monotonic adjusted effect.
    y_rank = stats.rankdata(np.asarray(y, dtype=float))
    x_rank = stats.rankdata(cols[0])
    cov_rank = [stats.rankdata(c) for c in cols[1:]]

    X = np.column_stack([np.ones(n), x_rank] + cov_rank)
    yv = y_rank

    p = X.shape[1]
    if n <= p:
        return None
    try:
        beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X.dot(beta)
        dof = n - p
        if dof <= 0:
            return None
        s2 = float(np.dot(resid, resid) / dof)
        xtx_inv = np.linalg.pinv(X.T.dot(X))
        cov_beta = s2 * xtx_inv
        se = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
        beta_x = float(beta[1])
        se_x = float(se[1]) if len(se) > 1 else None
        if se_x is None or se_x == 0.0:
            t_val = None
            p_val = None
        else:
            t_val = beta_x / se_x
            p_val = float(2 * (1 - stats.t.cdf(abs(t_val), df=dof)))
        return {
            "n": n,
            "beta": beta_x,
            "se": se_x,
            "t": float(t_val) if t_val is not None else None,
            "pval": p_val,
            "dof": dof,
        }
    except Exception:
        return None


def _fit_adjusted(samples, outcome, metric, covariates):
    raw_cov = {c: [] for c in covariates}
    y_raw = []
    x_raw = []
    for s in samples:
        y_raw.append(s.get(outcome))
        x_raw.append(s.get(metric))
        for c in covariates:
            raw_cov[c].append(s.get(c))

    y = [_safe_float(v) for v in y_raw]
    x = [_safe_float(v) for v in x_raw]
    cov_num = {}
    for c in covariates:
        cov_num[c], _ = _to_numeric_vector(raw_cov[c])

    keep = []
    for i in range(len(samples)):
        if y[i] is None or x[i] is None:
            continue
        ok = True
        for c in covariates:
            if cov_num[c][i] is None:
                ok = False
                break
        if ok:
            keep.append(i)

    if len(keep) < 4:
        return None

    yy = [y[i] for i in keep]
    xx = [x[i] for i in keep]
    cc = [[cov_num[c][i] for i in keep] for c in covariates]
    fit = _rank_ols(yy, xx, cc)
    if fit is None:
        return None
    fit["n_dropped"] = len(samples) - len(keep)
    return fit


def _tile_metric(n_pos, n_total, area, metric: str):
    if metric == "ratio":
        return (n_pos / n_total) if n_total > 0 else None
    if metric == "pos_density_mpx":
        return (n_pos / area) * 1e6 if area > 0 else None
    if metric == "total_density_mpx":
        return (n_total / area) * 1e6 if area > 0 else None
    raise ValueError(f"Unknown metric: {metric}")


def _bootstrap_adjusted(
    tile_rows,
    clinical_by_donor,
    outcome: str,
    metric: str,
    covariates,
    n_bootstrap: int,
    seed: int,
):
    by_donor = {}
    for r in tile_rows:
        donor = _norm_id(r.get("donor", ""))
        n_total = _safe_float(r.get("n_total_cells"))
        n_pos = _safe_float(r.get("n_marker_positive"))
        area = _safe_float(r.get("area_px"))
        if donor == "" or n_total is None or n_pos is None:
            continue
        by_donor.setdefault(donor, []).append((n_pos, n_total, area))

    rng = random.Random(seed)
    betas = []
    pvals = []
    for _ in range(n_bootstrap):
        samples = []
        for donor, tiles in by_donor.items():
            clin = clinical_by_donor.get(donor)
            if clin is None:
                continue
            k = max(1, int(math.ceil(len(tiles) * 0.8)))
            idxs = [rng.randrange(0, len(tiles)) for _ in range(k)]
            pos = sum(tiles[i][0] for i in idxs)
            total = sum(tiles[i][1] for i in idxs)
            area = sum((tiles[i][2] or 0.0) for i in idxs)
            val = _tile_metric(pos, total, area, metric)
            if val is None:
                continue
            row = {"donor": donor, metric: val}
            row.update(clin)
            samples.append(row)
        fit = _fit_adjusted(samples, outcome=outcome, metric=metric, covariates=covariates)
        if fit is None:
            continue
        if fit["beta"] is not None:
            betas.append(float(fit["beta"]))
        if fit["pval"] is not None:
            pvals.append(float(fit["pval"]))

    if len(betas) == 0:
        return {"var_beta": None, "sign_consistency": None, "p_pass": None, "n_boot_effective": 0}
    mean_beta = sum(betas) / len(betas)
    var_beta = sum((b - mean_beta) ** 2 for b in betas) / len(betas)
    sign_consistency = max(sum(1 for b in betas if b > 0), sum(1 for b in betas if b < 0)) / len(betas)
    p_pass = (sum(1 for p in pvals if p < 0.05) / len(pvals)) if len(pvals) > 0 else None
    return {
        "var_beta": var_beta,
        "sign_consistency": sign_consistency,
        "p_pass": p_pass,
        "n_boot_effective": len(betas),
    }


def run_adjusted_models(
    config_path: str,
    baseline_tile_csv: str | None = None,
    finetuned_tile_csv: str | None = None,
    clinical_csv: str | None = None,
    outcomes: list[str] | None = None,
    metrics: list[str] | None = None,
    covariates: list[str] | None = None,
    alpha: float = 0.05,
    n_bootstrap: int = 200,
    seed: int = 2024,
):
    cfg = yaml.safe_load(Path(config_path).read_text())
    repo_root = Path(config_path).resolve().parents[1]
    downstream_dir = _ensure_local_path(Path(cfg["downstream_dir"]), repo_root)
    reports_dir = _ensure_local_path(Path(cfg["reports_dir"]), repo_root)
    downstream_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    fi_dir = downstream_dir / "full_inference"
    if baseline_tile_csv is None:
        baseline_tile_csv = str(fi_dir / "baseline" / "tile_counts.tsv")
    if finetuned_tile_csv is None:
        finetuned_tile_csv = str(fi_dir / "finetuned" / "tile_counts.tsv")
    if clinical_csv is None:
        clinical_csv = str(Path(cfg["data_root"]) / "ROSMAP_clinical_n69.csv")

    baseline_path = _ensure_local_path(Path(baseline_tile_csv), repo_root)
    finetuned_path = _ensure_local_path(Path(finetuned_tile_csv), repo_root)
    clinical_path = _ensure_local_path(Path(clinical_csv), repo_root)
    for p in [baseline_path, finetuned_path, clinical_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    c_cols, c_rows = _read_table(clinical_path)
    donor_col = "projid" if "projid" in c_cols else c_cols[0]
    clinical_by_donor = {_norm_id(r[donor_col]): r for r in c_rows if _norm_id(r.get(donor_col, "")) != ""}

    _, b_tile_rows = _read_table(baseline_path)
    _, f_tile_rows = _read_table(finetuned_path)
    donor_rows = _aggregate_donor_from_tiles(b_tile_rows) + _aggregate_donor_from_tiles(f_tile_rows)

    outcome_list = [o for o in (outcomes or OUTCOMES_DEFAULT) if o in c_cols]
    metric_list = metrics or ["ratio", "pos_density_mpx", "total_density_mpx"]
    cov_list = [c for c in (covariates or ["age_death", "msex", "pmi", "educ"]) if c in c_cols]

    adjusted_rows = []
    bootstrap_rows = []
    markers = sorted(set(str(r["marker"]).lower() for r in donor_rows))

    for model in ["baseline", "finetuned"]:
        model_tiles = b_tile_rows if model == "baseline" else f_tile_rows
        for marker in markers:
            dm = [r for r in donor_rows if r["model"] == model and r["marker"] == marker]
            merged = []
            for r in dm:
                donor = _norm_id(r["donor"])
                clin = clinical_by_donor.get(donor)
                if clin is None:
                    continue
                merged.append({**r, **clin})
            tm = [r for r in model_tiles if str(r.get("marker", "")).lower() == marker]
            for metric in metric_list:
                for outcome in outcome_list:
                    fit = _fit_adjusted(
                        samples=merged,
                        outcome=outcome,
                        metric=metric,
                        covariates=cov_list,
                    )
                    adjusted_rows.append(
                        {
                            "model": model,
                            "marker": marker,
                            "metric": metric,
                            "outcome": outcome,
                            "n": fit["n"] if fit is not None else 0,
                            "n_dropped": fit["n_dropped"] if fit is not None else None,
                            "beta": fit["beta"] if fit is not None else None,
                            "se": fit["se"] if fit is not None else None,
                            "t": fit["t"] if fit is not None else None,
                            "pval": fit["pval"] if fit is not None else None,
                            "covariates": ",".join(cov_list),
                        }
                    )
                    bs = _bootstrap_adjusted(
                        tile_rows=tm,
                        clinical_by_donor=clinical_by_donor,
                        outcome=outcome,
                        metric=metric,
                        covariates=cov_list,
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                    bootstrap_rows.append(
                        {
                            "model": model,
                            "marker": marker,
                            "metric": metric,
                            "outcome": outcome,
                            "var_beta": bs["var_beta"],
                            "sign_consistency": bs["sign_consistency"],
                            "p_pass": bs["p_pass"],
                            "n_boot_effective": bs["n_boot_effective"],
                        }
                    )

    idx = [i for i, r in enumerate(adjusted_rows) if r["pval"] is not None]
    pvals = [adjusted_rows[i]["pval"] for i in idx]
    qvals = _fdr_bh(pvals) if len(pvals) > 0 else []
    for i, q in zip(idx, qvals):
        adjusted_rows[i]["fdr"] = q
        adjusted_rows[i]["pass_fdr"] = int(q < alpha)
    for r in adjusted_rows:
        if "fdr" not in r:
            r["fdr"] = None
            r["pass_fdr"] = 0

    out_adj = reports_dir / "e5_adjusted_models.tsv"
    with open(out_adj, "w", newline="") as f:
        fields = [
            "model",
            "marker",
            "metric",
            "outcome",
            "n",
            "n_dropped",
            "beta",
            "se",
            "t",
            "pval",
            "fdr",
            "pass_fdr",
            "covariates",
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(adjusted_rows)

    out_boot = reports_dir / "e5_adjusted_models_bootstrap.tsv"
    with open(out_boot, "w", newline="") as f:
        fields = ["model", "marker", "metric", "outcome", "var_beta", "sign_consistency", "p_pass", "n_boot_effective"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(bootstrap_rows)

    summary = {
        "baseline_tile_csv": str(baseline_path),
        "finetuned_tile_csv": str(finetuned_path),
        "clinical_csv": str(clinical_path),
        "outcomes": outcome_list,
        "metrics": metric_list,
        "covariates": cov_list,
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "n_adjusted_rows": len(adjusted_rows),
        "n_bootstrap_rows": len(bootstrap_rows),
        "outputs": {
            "adjusted_models": str(out_adj),
            "bootstrap_stability": str(out_boot),
        },
    }
    out_summary = reports_dir / "e5_adjusted_models_summary.json"
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[e5.5] adjusted={out_adj}")
    print(f"[e5.5] bootstrap={out_boot}")
    print(f"[e5.5] summary={out_summary}")


__all__ = ["run_adjusted_models"]
