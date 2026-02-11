#!/usr/bin/env python
"""
Purpose: Audit train/val/test leakage for donor split integrity and duplicate images.
Output: JSON/TSV reports for donor overlap, sample overlap, exact hash collisions, and near-duplicate pairs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile as tiff
import yaml

from fintune.utils.dataset import list_pairs, parse_marker_donor


@dataclass
class SampleRec:
    split: str
    stem: str
    donor: str
    marker: str
    img_path: Path
    mask_path: Path
    img_sha1: str
    mask_sha1: str
    dhash: int


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _to_gray_small(arr: np.ndarray, out_h: int = 8, out_w: int = 9) -> np.ndarray:
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.size == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)
    h, w = arr.shape[:2]
    ys = np.linspace(0, max(h - 1, 0), out_h).astype(np.int32)
    xs = np.linspace(0, max(w - 1, 0), out_w).astype(np.int32)
    return arr[np.ix_(ys, xs)]


def _dhash(arr: np.ndarray) -> int:
    small = _to_gray_small(arr, out_h=8, out_w=9)
    diff = small[:, 1:] > small[:, :-1]
    bitstr = "".join("1" if x else "0" for x in diff.flatten().tolist())
    return int(bitstr, 2) if bitstr else 0


def _hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _split_dir(cfg: dict, split: str) -> Path:
    if split == "train":
        return Path(cfg["cellpose_train_dir"])
    if split == "val":
        return Path(cfg["cellpose_val_dir"])
    if split == "test":
        return Path(cfg["cellpose_test_dir"])
    raise ValueError(split)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--out-dir", default="data/reports/e0_leakage_audit")
    ap.add_argument("--near-dup-hamming", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    recs: list[SampleRec] = []
    for split in splits:
        d = _split_dir(cfg, split)
        for img_path, mask_path in list_pairs(d):
            stem = img_path.stem
            marker, donor = parse_marker_donor(stem)
            img = tiff.imread(img_path)
            recs.append(
                SampleRec(
                    split=split,
                    stem=stem,
                    donor=str(donor),
                    marker=str(marker),
                    img_path=img_path,
                    mask_path=mask_path,
                    img_sha1=_sha1_file(img_path),
                    mask_sha1=_sha1_file(mask_path),
                    dhash=_dhash(img),
                )
            )

    donors_by_split = {s: sorted({r.donor for r in recs if r.split == s}) for s in splits}
    stems_by_split = {s: sorted({r.stem for r in recs if r.split == s}) for s in splits}

    overlaps = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            donor_overlap = sorted(set(donors_by_split[a]) & set(donors_by_split[b]))
            stem_overlap = sorted(set(stems_by_split[a]) & set(stems_by_split[b]))
            overlaps[f"{a}__{b}"] = {
                "donor_overlap_n": len(donor_overlap),
                "donor_overlap_examples": donor_overlap[:20],
                "stem_overlap_n": len(stem_overlap),
                "stem_overlap_examples": stem_overlap[:20],
            }

    # exact duplicate collisions across different splits
    by_img_hash = {}
    by_mask_hash = {}
    for r in recs:
        by_img_hash.setdefault(r.img_sha1, []).append(r)
        by_mask_hash.setdefault(r.mask_sha1, []).append(r)

    exact_collisions = []
    for hash_map, kind in [(by_img_hash, "image"), (by_mask_hash, "mask")]:
        for h, lst in hash_map.items():
            split_set = sorted({x.split for x in lst})
            if len(split_set) <= 1:
                continue
            exact_collisions.append(
                {
                    "type": kind,
                    "sha1": h,
                    "n_samples": len(lst),
                    "splits": ",".join(split_set),
                    "samples": ",".join(f"{x.split}:{x.stem}" for x in lst[:10]),
                }
            )

    # near-duplicate check across different splits by dhash
    near_dup_rows = []
    for i in range(len(recs)):
        ri = recs[i]
        for j in range(i + 1, len(recs)):
            rj = recs[j]
            if ri.split == rj.split:
                continue
            hd = _hamming(ri.dhash, rj.dhash)
            if hd <= args.near_dup_hamming:
                near_dup_rows.append(
                    {
                        "split_a": ri.split,
                        "stem_a": ri.stem,
                        "split_b": rj.split,
                        "stem_b": rj.stem,
                        "hamming": hd,
                        "same_donor": int(ri.donor == rj.donor),
                        "same_marker": int(ri.marker == rj.marker),
                    }
                )

    near_dup_rows = sorted(near_dup_rows, key=lambda x: (x["hamming"], -x["same_donor"]))

    # write outputs
    counts_path = out_dir / "split_counts.tsv"
    with open(counts_path, "w", newline="") as f:
        fields = ["split", "n_pairs", "n_donors", "n_markers"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for s in splits:
            subset = [r for r in recs if r.split == s]
            w.writerow(
                {
                    "split": s,
                    "n_pairs": len(subset),
                    "n_donors": len({r.donor for r in subset}),
                    "n_markers": len({r.marker for r in subset}),
                }
            )

    exact_path = out_dir / "exact_hash_collisions.tsv"
    with open(exact_path, "w", newline="") as f:
        fields = ["type", "sha1", "n_samples", "splits", "samples"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(exact_collisions)

    near_path = out_dir / "near_duplicates.tsv"
    with open(near_path, "w", newline="") as f:
        fields = ["split_a", "stem_a", "split_b", "stem_b", "hamming", "same_donor", "same_marker"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(near_dup_rows)

    summary = {
        "config": args.config,
        "n_total_pairs": len(recs),
        "split_counts_tsv": str(counts_path),
        "exact_hash_collisions_tsv": str(exact_path),
        "near_duplicates_tsv": str(near_path),
        "overlaps": overlaps,
        "n_exact_hash_collisions": len(exact_collisions),
        "n_near_duplicates": len(near_dup_rows),
        "near_dup_hamming": args.near_dup_hamming,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[leakage_audit] summary={summary_path}")
    print(f"[leakage_audit] exact={exact_path}")
    print(f"[leakage_audit] near={near_path}")
    print(f"[leakage_audit] n_pairs={len(recs)}")


if __name__ == "__main__":
    main()

