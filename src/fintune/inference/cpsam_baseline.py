from __future__ import annotations

from pathlib import Path
import json
import yaml
import numpy as np
import tifffile as tiff
from cellpose import models

from fintune.utils.dataset import list_pairs, read_image, read_mask, parse_marker_donor
from fintune.utils.metrics import summarize_metrics


def run_baseline_inference(
    config_path: str,
    model: str = "cpsam",
    diameter: float | None = None,
    chan=(1, 2),
    save_vis: bool = False,
    split: str = "test",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    zero_dapi: bool = False,
):
    del save_vis  # not used in this minimal runner

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    split_to_dir = {
        "train": Path(cfg["cellpose_train_dir"]),
        "val": Path(cfg["cellpose_val_dir"]),
        "test": Path(cfg["cellpose_test_dir"]),
    }
    if split not in split_to_dir:
        raise ValueError(f"Unknown split={split}")
    data_dir = split_to_dir[split]

    out_dir = Path(cfg["predictions_dir"]) / Path(model).name / split
    out_dir.mkdir(parents=True, exist_ok=True)

    model_obj = models.CellposeModel(gpu=False, pretrained_model=model)
    pairs = list_pairs(data_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No image/mask pairs found in {data_dir}")

    overall_true = []
    overall_pred = []
    grouped_true = {}
    grouped_pred = {}

    for img_path, mask_path in pairs:
        img = read_image(img_path)
        if img.ndim == 2:
            img = np.stack([img, img, np.zeros_like(img)], axis=-1)
        if zero_dapi and img.ndim == 3 and img.shape[-1] >= 1:
            img = img.copy()
            img[..., 0] = 0
        pred, _, _ = model_obj.eval(
            img,
            channels=list(chan),
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
        pred = pred.astype(np.uint16)
        tiff.imwrite(out_dir / f"{img_path.stem}_pred.tif", pred)

        true = read_mask(mask_path).astype(np.int32)
        overall_true.append(true)
        overall_pred.append(pred)
        marker, _ = parse_marker_donor(img_path.stem)
        grouped_true.setdefault(marker, []).append(true)
        grouped_pred.setdefault(marker, []).append(pred)

    metrics_json = {"overall": summarize_metrics(overall_true, overall_pred)}
    for marker in sorted(grouped_true):
        metrics_json[marker] = summarize_metrics(grouped_true[marker], grouped_pred[marker])

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"[baseline] split={split}, n={len(pairs)}, model={model}, zero_dapi={zero_dapi}")
    print(f"[baseline] outputs={out_dir}")
    print(f"[baseline] overall={metrics_json['overall']}")


__all__ = ["run_baseline_inference"]
