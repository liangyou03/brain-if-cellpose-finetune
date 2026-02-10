#!/usr/bin/env python
"""
Purpose: Train E1.2 generic finetune model using combined markers.
Output: finetuned checkpoint for downstream inference/evaluation.
"""
import argparse
from fintune.training.finetune_generic import finetune_generic


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--checkpoint", default=None, help="继续训练的 ckpt")
    ap.add_argument("--nimg-per-epoch", type=int, default=None)
    ap.add_argument("--nimg-test-per-epoch", type=int, default=None)
    ap.add_argument("--model-name", default="generic_cpsam")
    ap.add_argument("--marker-filter", default=None, help="only train one marker, e.g. gfap")
    ap.add_argument("--marker-include", nargs="+", default=None, help="train only these markers")
    ap.add_argument("--marker-exclude", nargs="+", default=None, help="exclude these markers from training")
    ap.add_argument("--zero-dapi", action="store_true", help="set DAPI channel to zero during training")
    ap.add_argument("--seed", type=int, default=2024)
    return ap.parse_args()


def main():
    args = parse_args()
    finetune_generic(
        config_path=args.config,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        resume_ckpt=args.checkpoint,
        model_name=args.model_name,
        marker_filter=args.marker_filter,
        marker_include=args.marker_include,
        marker_exclude=args.marker_exclude,
        nimg_per_epoch=args.nimg_per_epoch,
        nimg_test_per_epoch=args.nimg_test_per_epoch,
        zero_dapi=args.zero_dapi,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
