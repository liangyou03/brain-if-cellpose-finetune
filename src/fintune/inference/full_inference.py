"""
Full-tile inference and donor-level ratio aggregation.

This implementation compares an initial model (baseline) and a finetuned model
using only paths under this repository (`finetune/`).
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import tifffile as tiff
import yaml
from cellpose import models

MARKERS = {"gfap", "iba1", "neun", "olig2", "pecam"}


def _canonical_pair_key(path: Path, channel: int) -> str | None:
    name = path.name
    pat = re.compile(rf"(?:_[A-Za-z0-9]+-)?b0c{channel}", flags=re.IGNORECASE)
    if pat.search(name) is None:
        return None
    core = pat.sub("_CHAN", name, count=1).lower()
    return f"{str(path.parent).lower()}::{core}"


def _load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _ensure_local_path(p: Path, repo_root: Path) -> Path:
    # Must be configured via paths under this repo. Symlinks are allowed.
    root = repo_root.resolve()
    rp = p.expanduser()
    rp = (root / rp) if not rp.is_absolute() else rp
    rp = Path(str(rp))
    root_s = str(root)
    rp_s = str(rp)
    if not (rp_s == root_s or rp_s.startswith(root_s + "/")):
        raise ValueError(f"Path must be under repository root: {p}")
    return rp


def _discover_tiles(raw_tiles_dir: Path, max_tiles: int | None = None):
    # Support both patterns:
    # 1) ..._b0c0... + ..._b0c1...
    # 2) ..._dapi-b0c0... + ..._<marker>-b0c1...
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
        if max_tiles is not None and len(pairs) >= max_tiles:
            break
    return pairs


def _infer_marker_and_donor(path: Path) -> Tuple[str, str]:
    parts = [p.lower() for p in path.parts]
    marker = "unknown"
    for p in parts:
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
    img = np.stack([dapi, marker, z], axis=-1)
    return img


def _count_positive(pred: np.ndarray, marker_channel: np.ndarray, q: float = 75.0):
    obj_ids = np.unique(pred)
    obj_ids = obj_ids[obj_ids > 0]
    n_total = int(len(obj_ids))
    if n_total == 0:
        return 0, 0, 0.0
    thr = float(np.percentile(marker_channel, q))
    n_pos = 0
    for oid in obj_ids:
        reg = pred == oid
        if float(marker_channel[reg].mean()) > thr:
            n_pos += 1
    ratio = n_pos / n_total if n_total > 0 else 0.0
    return n_total, n_pos, ratio


def _run_model(
    tile_pairs,
    model_label: str,
    model_spec: str,
    out_base: Path,
    save_masks: bool,
    resume: bool,
    flow_threshold: float,
    cellprob_threshold: float,
):
    out_dir = out_base / model_label
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "pred_masks"
    if save_masks:
        pred_dir.mkdir(parents=True, exist_ok=True)
    tile_tsv = out_dir / "tile_counts.tsv"

    done = set()
    rows = []
    if resume and tile_tsv.exists():
        with open(tile_tsv, "r", newline="") as f:
            rr = csv.DictReader(f, delimiter="\t")
            required_fields = {
                "tile_key",
                "model",
                "donor",
                "marker",
                "n_total_cells",
                "n_marker_positive",
                "area_px",
                "total_density_mpx",
                "pos_density_mpx",
                "ratio",
            }
            if rr.fieldnames is None or not required_fields.issubset(set(rr.fieldnames)):
                print(f"[full_inference] resume disabled for {tile_tsv} due to legacy schema")
            else:
                for r in rr:
                    done.add(r["tile_key"])
                    rows.append(r)

    model_obj = models.CellposeModel(gpu=True, pretrained_model=model_spec)
    for c0, c1 in tile_pairs:
        tile_key = str(c0)
        if tile_key in done:
            continue
        marker, donor = _infer_marker_and_donor(c0)
        try:
            img = _load_3ch(c0, c1)
        except Exception as e:
            print(f"[full_inference][warn] skip unreadable tile: {c0} ({e})")
            continue
        pred, _, _ = model_obj.eval(
            img,
            channels=[1, 2],
            diameter=None,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        pred = pred.astype(np.int32)
        area_px = int(img.shape[0] * img.shape[1])
        n_total, n_pos, ratio = _count_positive(pred, img[..., 1])
        total_density_mpx = (float(n_total) / float(area_px)) * 1e6 if area_px > 0 else 0.0
        pos_density_mpx = (float(n_pos) / float(area_px)) * 1e6 if area_px > 0 else 0.0

        if save_masks:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{donor}_{marker}_{c0.stem}_pred.tif")
            tiff.imwrite(pred_dir / safe, pred.astype(np.uint16))

        row = {
            "model": model_label,
            "tile_key": tile_key,
            "tile_c0": str(c0),
            "tile_c1": str(c1),
            "donor": donor,
            "marker": marker,
            "n_total_cells": int(n_total),
            "n_marker_positive": int(n_pos),
            "area_px": area_px,
            "total_density_mpx": total_density_mpx,
            "pos_density_mpx": pos_density_mpx,
            "ratio": float(ratio),
        }
        rows.append(row)

    fields = [
        "model",
        "tile_key",
        "tile_c0",
        "tile_c1",
        "donor",
        "marker",
        "n_total_cells",
        "n_marker_positive",
        "area_px",
        "total_density_mpx",
        "pos_density_mpx",
        "ratio",
    ]
    with open(tile_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return rows, tile_tsv


def _aggregate_donor(rows: List[Dict]):
    agg = {}
    for r in rows:
        marker = str(r["marker"]).lower()
        donor = str(r["donor"])
        if marker not in MARKERS or donor == "unknown":
            continue
        key = (str(r["model"]), donor, marker)
        if key not in agg:
            agg[key] = {"n_total_cells": 0, "n_marker_positive": 0, "area_px": 0}
        agg[key]["n_total_cells"] += int(r["n_total_cells"])
        agg[key]["n_marker_positive"] += int(r["n_marker_positive"])
        agg[key]["area_px"] += int(r.get("area_px", 0))

    out = []
    for (model, donor, marker), v in agg.items():
        n_total = v["n_total_cells"]
        n_pos = v["n_marker_positive"]
        area_px = v["area_px"]
        ratio = (n_pos / n_total) if n_total > 0 else 0.0
        total_density_mpx = (float(n_total) / float(area_px)) * 1e6 if area_px > 0 else 0.0
        pos_density_mpx = (float(n_pos) / float(area_px)) * 1e6 if area_px > 0 else 0.0
        out.append(
            {
                "model": model,
                "donor": donor,
                "marker": marker,
                "n_total_cells": n_total,
                "n_marker_positive": n_pos,
                "area_px": area_px,
                "total_density_mpx": total_density_mpx,
                "pos_density_mpx": pos_density_mpx,
                "ratio": ratio,
            }
        )
    return out


def run_full_inference(
    config_path: str,
    baseline_model: str = "cpsam",
    finetuned_model: str = "generic_cpsam",
    max_tiles: int | None = None,
    save_masks: bool = False,
    resume: bool = True,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
):
    cfg = _load_yaml(Path(config_path))
    repo_root = Path(config_path).resolve().parents[1]
    raw_tiles_dir = _ensure_local_path(Path(cfg["raw_tiles_dir"]), repo_root)
    downstream_dir = _ensure_local_path(Path(cfg["downstream_dir"]), repo_root) / "full_inference"
    downstream_dir.mkdir(parents=True, exist_ok=True)

    tile_pairs = _discover_tiles(raw_tiles_dir, max_tiles=max_tiles)
    if len(tile_pairs) == 0:
        raise RuntimeError(f"No channel-paired tiles found in {raw_tiles_dir}")

    # Resolve finetuned checkpoint name to local checkpoint path if needed.
    finetuned_spec = finetuned_model
    ckpt_guess = Path(cfg["checkpoints_dir"]) / finetuned_model / "models" / finetuned_model
    if ckpt_guess.exists():
        finetuned_spec = str(ckpt_guess)

    b_rows, b_tsv = _run_model(
        tile_pairs=tile_pairs,
        model_label="baseline",
        model_spec=baseline_model,
        out_base=downstream_dir,
        save_masks=save_masks,
        resume=resume,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    f_rows, f_tsv = _run_model(
        tile_pairs=tile_pairs,
        model_label="finetuned",
        model_spec=finetuned_spec,
        out_base=downstream_dir,
        save_masks=save_masks,
        resume=resume,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    all_rows = b_rows + f_rows
    n_unknown = sum(1 for r in all_rows if str(r.get("donor", "")) == "unknown")
    all_tile_tsv = downstream_dir / "tile_counts_all.tsv"
    with open(all_tile_tsv, "w", newline="") as f:
        fields = list(all_rows[0].keys()) if len(all_rows) > 0 else [
            "model",
            "tile_key",
            "tile_c0",
            "tile_c1",
            "donor",
            "marker",
            "n_total_cells",
            "n_marker_positive",
            "area_px",
            "total_density_mpx",
            "pos_density_mpx",
            "ratio",
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        if len(all_rows) > 0:
            w.writerows(all_rows)

    donor_rows = _aggregate_donor(all_rows)
    donor_tsv = downstream_dir / "donor_marker_ratio.tsv"
    with open(donor_tsv, "w", newline="") as f:
        fields = [
            "model",
            "donor",
            "marker",
            "n_total_cells",
            "n_marker_positive",
            "area_px",
            "total_density_mpx",
            "pos_density_mpx",
            "ratio",
        ]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(donor_rows)

    summary = {
        "raw_tiles_dir": str(raw_tiles_dir),
        "n_tile_pairs": len(tile_pairs),
        "n_marker_from_c2_fallback": int(sum(1 for _, c1 in tile_pairs if "b0c2" in c1.name.lower())),
        "n_rows_total": len(all_rows),
        "n_rows_unknown_donor": n_unknown,
        "baseline_model": baseline_model,
        "finetuned_model": finetuned_spec,
        "outputs": {
            "baseline_tile_counts": str(b_tsv),
            "finetuned_tile_counts": str(f_tsv),
            "all_tile_counts": str(all_tile_tsv),
            "donor_marker_ratio": str(donor_tsv),
        },
    }
    with open(downstream_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[full_inference] n_tile_pairs={len(tile_pairs)}")
    print("[full_inference] gpu=True")
    print(f"[full_inference] baseline={b_tsv}")
    print(f"[full_inference] finetuned={f_tsv}")
    print(f"[full_inference] donor={donor_tsv}")


__all__ = ["run_full_inference"]
