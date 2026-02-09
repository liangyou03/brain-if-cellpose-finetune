#!/usr/bin/env python
"""
E5.1: 全数据推理（14,749 tiles）并汇总 donor-level density。
"""
import argparse
from fintune.inference.full_inference import run_full_inference


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--model", default="fine_tuned", help="baseline or fine_tuned ckpt name/path")
    ap.add_argument("--batch", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    run_full_inference(
        config_path=args.config,
        model_name=args.model,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
