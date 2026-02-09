#!/usr/bin/env python
"""
E1.2: 四类 marker 合并的通用 fine-tune。
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
    return ap.parse_args()


def main():
    args = parse_args()
    finetune_generic(
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
