from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import tifffile as tiff

from fintune.utils.dataset import parse_marker_donor


def _find_image(base: Path, stem: str) -> Path | None:
    for ext in [".tif", ".tiff", ".png"]:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _find_marker(base: Path, stem: str) -> Path | None:
    for ext in [".tif", ".tiff", ".png"]:
        p = base / f"{stem}_marker{ext}"
        if p.exists():
            return p
    return None


def _to_single_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] >= 1:
        return arr[..., 0]
    raise ValueError(f"Unsupported shape: {arr.shape}")


def prepare_cellpose_dataset(
    config_path: str,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 2024,
    out_format: str = "png",
) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["brain_data_dir"])
    splits_dir = Path(cfg["splits_dir"])
    out_dirs = {
        "train": Path(cfg["cellpose_train_dir"]),
        "val": Path(cfg["cellpose_val_dir"]),
        "test": Path(cfg["cellpose_test_dir"]),
    }
    for d in [splits_dir, *out_dirs.values()]:
        d.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(raw_dir.glob("*_cellbodies.npy"))
    stems = [m.name.replace("_cellbodies.npy", "") for m in mask_files]
    if len(stems) == 0:
        raise RuntimeError(f"No *_cellbodies.npy found in {raw_dir}")

    donors = {stem: parse_marker_donor(stem)[1] for stem in stems}
    unique_donors = np.array(sorted(set(donors.values())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_donors)

    n_train = int(len(unique_donors) * train_frac)
    n_val = int(len(unique_donors) * val_frac)
    donor_splits = {
        "train": set(unique_donors[:n_train].tolist()),
        "val": set(unique_donors[n_train : n_train + n_val].tolist()),
        "test": set(unique_donors[n_train + n_val :].tolist()),
    }
    with open(splits_dir / "donor_splits.yaml", "w") as f:
        yaml.safe_dump({k: sorted(v) for k, v in donor_splits.items()}, f)

    n_written = 0
    n_skipped = 0
    for stem in stems:
        donor = donors[stem]
        split = next((k for k, v in donor_splits.items() if donor in v), None)
        if split is None:
            n_skipped += 1
            continue

        img_path = _find_image(raw_dir, stem)
        marker_path = _find_marker(raw_dir, stem)
        mask_path = raw_dir / f"{stem}_cellbodies.npy"
        if img_path is None or not mask_path.exists():
            n_skipped += 1
            continue

        dapi = _to_single_channel(tiff.imread(img_path))
        if marker_path is not None:
            marker = _to_single_channel(tiff.imread(marker_path))
        else:
            # fallback: if marker file missing, use the same file as second channel
            marker = dapi.copy()

        zeros = np.zeros_like(dapi)
        three_ch = np.stack([dapi, marker, zeros], axis=-1)
        mask = np.load(mask_path).astype(np.uint16)

        out_img = out_dirs[split] / f"{stem}.{out_format}"
        out_mask = out_dirs[split] / f"{stem}_mask.tif"
        tiff.imwrite(out_img, three_ch)
        tiff.imwrite(out_mask, mask)
        n_written += 1

    print(f"[prepare_cellpose_data] written={n_written}, skipped={n_skipped}")
    print({k: str(v) for k, v in out_dirs.items()})


__all__ = ["prepare_cellpose_dataset"]
