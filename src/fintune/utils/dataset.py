from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile as tiff

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def read_image(path: Path) -> np.ndarray:
    arr = tiff.imread(path)
    return arr


def read_mask(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return tiff.imread(path)


def list_pairs(dir_path: Path):
    pairs = []
    for p in sorted(dir_path.iterdir()):
        suf = p.suffix.lower()
        if suf not in IMG_EXTS:
            continue
        if p.stem.endswith("_mask") or p.stem.endswith("_pred"):
            continue
        mask_tif = p.with_name(f"{p.stem}_mask.tif")
        mask_tiff = p.with_name(f"{p.stem}_mask.tiff")
        if mask_tif.exists():
            pairs.append((p, mask_tif))
        elif mask_tiff.exists():
            pairs.append((p, mask_tiff))
    return pairs


def parse_marker_donor(stem: str):
    parts = stem.split("_")
    marker = parts[0].lower() if len(parts) > 0 else "unknown"
    donor = parts[1] if len(parts) > 1 else "unknown"
    return marker, donor
