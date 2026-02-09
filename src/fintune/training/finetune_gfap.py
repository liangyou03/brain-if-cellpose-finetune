from __future__ import annotations

from fintune.training.finetune_generic import finetune_generic


def finetune_gfap(
    config_path: str,
    epochs: int = 80,
    lr: float = 1e-4,
    batch_size: int = 4,
    resume_ckpt: str | None = None,
    nimg_per_epoch: int | None = None,
    nimg_test_per_epoch: int | None = None,
):
    return finetune_generic(
        config_path=config_path,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        resume_ckpt=resume_ckpt,
        model_name="gfap_cpsam",
        marker_filter="gfap",
        nimg_per_epoch=nimg_per_epoch,
        nimg_test_per_epoch=nimg_test_per_epoch,
    )


__all__ = ["finetune_gfap"]
