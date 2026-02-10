#!/usr/bin/env python
"""
Purpose: Run E4.1 micro-sam prompt sweep on sampled tiles.
Output: prompt-setting comparison metrics and summary tables.
"""
import argparse
from fintune.inference.microsam_sweep import run_microsam_sweep


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--points_per_side", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--num_tiles", type=int, default=10, help="per marker sampled tiles")
    ap.add_argument("--seed", type=int, default=2024)
    return ap.parse_args()


def main():
    args = parse_args()
    run_microsam_sweep(
        config_path=args.config,
        points_per_side=args.points_per_side,
        num_tiles=args.num_tiles,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
