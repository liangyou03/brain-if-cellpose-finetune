from __future__ import annotations

from pathlib import Path
import random
import yaml
import numpy as np
import torch
from cellpose import models, train as cp_train

from fintune.utils.dataset import list_pairs, read_image, read_mask, parse_marker_donor


def _load_data(
    dir_path: Path,
    marker_filter: str | None = None,
    marker_include: set[str] | None = None,
    marker_exclude: set[str] | None = None,
):
    imgs = []
    masks = []
    for img_path, mask_path in list_pairs(dir_path):
        marker, _ = parse_marker_donor(img_path.stem)
        if marker_filter is not None and marker != marker_filter.lower():
            continue
        if marker_include is not None and marker not in marker_include:
            continue
        if marker_exclude is not None and marker in marker_exclude:
            continue
        img = read_image(img_path)
        if img.ndim == 2:
            img = np.stack([img, img, np.zeros_like(img)], axis=-1)
        imgs.append(img)
        masks.append(read_mask(mask_path).astype(np.int32))
    return imgs, masks


def finetune_generic(
    config_path: str,
    epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 4,
    resume_ckpt: str | None = None,
    model_name: str = "generic_cpsam",
    marker_filter: str | None = None,
    marker_include: list[str] | None = None,
    marker_exclude: list[str] | None = None,
    nimg_per_epoch: int | None = None,
    nimg_test_per_epoch: int | None = None,
    zero_dapi: bool = False,
    seed: int = 2024,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_dir = Path(cfg["cellpose_train_dir"])
    val_dir = Path(cfg["cellpose_val_dir"])
    ckpt_dir = Path(cfg["checkpoints_dir"]) / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    include_set = set([m.lower() for m in marker_include]) if marker_include is not None else None
    exclude_set = set([m.lower() for m in marker_exclude]) if marker_exclude is not None else None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_imgs, train_masks = _load_data(
        train_dir,
        marker_filter=marker_filter,
        marker_include=include_set,
        marker_exclude=exclude_set,
    )
    val_imgs, val_masks = _load_data(
        val_dir,
        marker_filter=marker_filter,
        marker_include=include_set,
        marker_exclude=exclude_set,
    )
    if zero_dapi:
        for img in train_imgs:
            if img.ndim == 3 and img.shape[-1] >= 1:
                img[..., 0] = 0
        for img in val_imgs:
            if img.ndim == 3 and img.shape[-1] >= 1:
                img[..., 0] = 0
    if len(train_imgs) == 0:
        raise RuntimeError(f"No training pairs found in {train_dir} for marker={marker_filter}")
    if len(val_imgs) == 0:
        raise RuntimeError(f"No validation pairs found in {val_dir} for marker={marker_filter}")

    pretrained = resume_ckpt if resume_ckpt else "cpsam"
    model_obj = models.CellposeModel(gpu=True, pretrained_model=pretrained)

    model_path = cp_train.train_seg(
        model_obj.net,
        train_data=train_imgs,
        train_labels=train_masks,
        test_data=val_imgs,
        test_labels=val_masks,
        channel_axis=2,
        batch_size=batch_size,
        learning_rate=lr,
        n_epochs=epochs,
        save_path=str(ckpt_dir),
        model_name=model_name,
        rescale=False,
        compute_flows=False,
        nimg_per_epoch=nimg_per_epoch,
        nimg_test_per_epoch=nimg_test_per_epoch,
    )

    # train_seg may return path or tuple depending on version
    if isinstance(model_path, (tuple, list)):
        saved = model_path[0]
    else:
        saved = model_path

    print(
        f"[finetune] model_name={model_name}, train_n={len(train_imgs)}, val_n={len(val_imgs)}, zero_dapi={zero_dapi}, seed={seed}"
    )
    print(f"[finetune] saved={saved}")
    return saved


__all__ = ["finetune_generic"]
