#!/usr/bin/env python
"""
Purpose: Run E2.1 label-efficiency budget-curve experiments with repeated sampling.
Output: per-run metrics and aggregated budget summaries.
"""
import argparse
from fintune.training.budget_curve import run_budget_curve


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--budgets", nargs="+", type=int, default=[2, 5, 10, 20, 30])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--marker", default="all", help="all or specific marker")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--eval-split", choices=["val", "test"], default="val")
    ap.add_argument("--nimg-per-epoch", type=int, default=24)
    ap.add_argument("--nimg-test-per-epoch", type=int, default=12)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--exp-name", default="budget_curve")
    return ap.parse_args()


def main():
    args = parse_args()
    run_budget_curve(
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
    )


if __name__ == "__main__":
    main()
