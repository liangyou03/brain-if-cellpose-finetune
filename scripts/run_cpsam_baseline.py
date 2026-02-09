#!/usr/bin/env python
"""
E1.1: off-the-shelf Cellpose-SAM 推理。
在 Test 集上输出 AP@0.5 / Precision / Recall / F1 并保存预测 mask。
"""
import argparse
from fintune.inference.cpsam_baseline import run_baseline_inference


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--model", default="cpsam", help="cellpose model name or checkpoint path")
    ap.add_argument("--diameter", type=float, default=None, help="optional: preset diameter")
    ap.add_argument("--chan", type=int, nargs=2, default=[1, 2], help="cellpose chan/chan2 order")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--save-vis", action="store_true", help="保存可视化 png")
    return ap.parse_args()


def main():
    args = parse_args()
    run_baseline_inference(
        config_path=args.config,
        model=args.model,
        diameter=args.diameter,
        chan=args.chan,
        save_vis=args.save_vis,
        split=args.split,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
    )


if __name__ == "__main__":
    main()
