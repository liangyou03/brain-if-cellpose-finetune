#!/usr/bin/env python
"""
Purpose: Run E5.5 covariate-adjusted downstream models with bootstrap stability.
Output: adjusted effect tables, FDR results, and bootstrap stability summaries.
"""
from __future__ import annotations

import argparse

from fintune.evaluation.adjusted_models import run_adjusted_models


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--baseline-tile-csv", default=None)
    ap.add_argument("--finetuned-tile-csv", default=None)
    ap.add_argument("--clinical-csv", default=None)
    ap.add_argument("--outcomes", nargs="+", default=None)
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["ratio", "pos_density_mpx", "total_density_mpx"],
        choices=["ratio", "pos_density_mpx", "total_density_mpx"],
    )
    ap.add_argument("--covariates", nargs="+", default=["age_death", "msex", "pmi", "educ"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2024)
    return ap.parse_args()


def main():
    args = parse_args()
    run_adjusted_models(
        config_path=args.config,
        baseline_tile_csv=args.baseline_tile_csv,
        finetuned_tile_csv=args.finetuned_tile_csv,
        clinical_csv=args.clinical_csv,
        outcomes=args.outcomes,
        metrics=args.metrics,
        covariates=args.covariates,
        alpha=args.alpha,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
