#!/usr/bin/env python
"""
Purpose: Train E1.3 GFAP-specific finetune model.
Output: GFAP-focused checkpoint for marker-specific evaluation.
"""
import argparse
from fintune.training.finetune_gfap import finetune_gfap


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--nimg-per-epoch", type=int, default=None)
    ap.add_argument("--nimg-test-per-epoch", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    finetune_gfap(
        config_path=args.config,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        resume_ckpt=args.checkpoint,
        nimg_per_epoch=args.nimg_per_epoch,
        nimg_test_per_epoch=args.nimg_test_per_epoch,
    )


if __name__ == "__main__":
    main()
