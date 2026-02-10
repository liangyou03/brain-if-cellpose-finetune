#!/usr/bin/env python
"""
Purpose: Run E2.2 random-vs-hard sampling comparison under fixed label budgets.
Output: per-run records, difficulty scores, and strategy comparison summary.
"""
from __future__ import annotations

import argparse

from fintune.training.active_sampling import run_active_sampling_compare


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--budgets", nargs="+", type=int, default=[2, 5, 10, 20])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--marker", default="all", help="all or specific marker")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--eval-split", choices=["val", "test"], default="val")
    ap.add_argument("--nimg-per-epoch", type=int, default=72)
    ap.add_argument("--nimg-test-per-epoch", type=int, default=24)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--exp-name", default="e2_active_sampling_compare")
    ap.add_argument("--scoring-model", default="cpsam")
    ap.add_argument("--score-flow-a", type=float, default=0.4)
    ap.add_argument("--score-cellprob-a", type=float, default=0.0)
    ap.add_argument("--score-flow-b", type=float, default=0.2)
    ap.add_argument("--score-cellprob-b", type=float, default=-0.5)
    return ap.parse_args()


def main():
    args = parse_args()
    run_active_sampling_compare(
        config_path=args.config,
        budgets=args.budgets,
        repeats=args.repeats,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        marker=args.marker,
        seed=args.seed,
        eval_split=args.eval_split,
        nimg_per_epoch=args.nimg_per_epoch,
        nimg_test_per_epoch=args.nimg_test_per_epoch,
        skip_existing=args.skip_existing,
        exp_name=args.exp_name,
        scoring_model=args.scoring_model,
        flow_a=args.score_flow_a,
        cellprob_a=args.score_cellprob_a,
        flow_b=args.score_flow_b,
        cellprob_b=args.score_cellprob_b,
    )


if __name__ == "__main__":
    main()
