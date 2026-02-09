#!/usr/bin/env python
"""
E3.1: Marker gating ablation。
"""
import argparse
from fintune.evaluation.gating_ablation import run_gating_ablation


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--model", default="generic_cpsam")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--intensity_thresh", type=float, nargs=2, default=[0.1, 0.3], help="min,max marker intensity percentiles")
    ap.add_argument("--min_area", type=int, default=50, help="min area in pixels")
    return ap.parse_args()


def main():
    args = parse_args()
    run_gating_ablation(
        config_path=args.config,
        model_name=args.model,
        intensity_thresh=tuple(args.intensity_thresh),
        min_area=args.min_area,
        split=args.split,
    )


if __name__ == "__main__":
    main()
