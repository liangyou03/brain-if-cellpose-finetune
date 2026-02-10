#!/usr/bin/env python
"""
Purpose: Run E5.1 full-tile inference for baseline vs finetuned models.
Output: tile-level counts and donor-level density/ratio aggregation tables.
"""
import argparse
from fintune.inference.full_inference import run_full_inference


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--baseline-model", default="cpsam", help="initial model name/path")
    ap.add_argument("--finetuned-model", default="generic_cpsam", help="finetuned model name/path")
    ap.add_argument("--max-tiles", type=int, default=None, help="debug mode: cap number of tiles")
    ap.add_argument("--save-masks", action="store_true", help="save predicted masks")
    ap.add_argument("--no-resume", action="store_true", help="do not reuse existing tile counts")
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    return ap.parse_args()


def main():
    args = parse_args()
    run_full_inference(
        config_path=args.config,
        baseline_model=args.baseline_model,
        finetuned_model=args.finetuned_model,
        max_tiles=args.max_tiles,
        save_masks=args.save_masks,
        resume=(not args.no_resume),
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
    )


if __name__ == "__main__":
    main()
