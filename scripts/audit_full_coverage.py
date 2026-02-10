#!/usr/bin/env python
"""
Purpose: Audit E5 full-inference tile coverage and donor/marker matching quality.
Output: coverage report tables for missing/invalid entries.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    return ap.parse_args()


def _infer_marker_and_donor(path: Path):
    marker = "unknown"
    for p in [x.lower() for x in path.parts]:
        if p in {"gfap", "iba1", "neun", "olig2", "pecam"}:
            marker = p
            break
    donor = "unknown"
    for p in path.parts:
        m = re.search(r"(\d{6,12})", p)
        if m:
            donor = m.group(1)
            break
    return marker, donor


def _canonical_pair_key(path: Path, channel: int):
    name = path.name
    pat = re.compile(rf"(?:_[A-Za-z0-9]+-)?b0c{channel}", flags=re.IGNORECASE)
    if pat.search(name) is None:
        return None
    core = pat.sub("_CHAN", name, count=1).lower()
    return f"{str(path.parent).lower()}::{core}"


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw_dir = Path(cfg["raw_tiles_dir"])
    clinical = Path(cfg["data_root"]) / "ROSMAP_clinical_n69.csv"
    reports = Path(cfg["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    tif_files = [p for p in raw_dir.rglob("*.tif*") if p.suffix.lower() in {".tif", ".tiff"}]
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

    donor_ctr = Counter()
    marker_ctr = Counter()
    for c0, _ in pairs:
        marker, donor = _infer_marker_and_donor(c0)
        donor_ctr[donor] += 1
        marker_ctr[marker] += 1

    clin_donors = set()
    with open(clinical, "r", newline="") as f:
        rr = csv.DictReader(f)
        for r in rr:
            clin_donors.add(str(r["projid"]).strip().replace(".0", ""))

    tile_donors = set(donor_ctr.keys())
    summary = {
        "raw_tiles_dir": str(raw_dir),
        "clinical_csv": str(clinical),
        "n_paired_tiles": len(pairs),
        "n_marker_from_c2_fallback": int(sum(1 for _, c1 in pairs if "b0c2" in c1.name.lower())),
        "n_tile_donors": len(tile_donors),
        "n_clinical_donors": len(clin_donors),
        "unknown_donor_tiles": donor_ctr.get("unknown", 0),
        "donors_not_in_clinical": sorted(list(tile_donors - clin_donors)),
        "clinical_donors_missing_tiles": sorted(list(clin_donors - tile_donors)),
        "marker_tile_counts": dict(sorted(marker_ctr.items())),
    }
    out = reports / "e5_coverage_audit.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[coverage] saved {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
