#!/usr/bin/env python
"""
E5.2/5.3: 下游相关性 + 稳定性 (bootstrap)。
"""
import argparse
from fintune.evaluation.downstream import run_downstream_analysis


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--pred-dir", default="predictions/fine_tuned_full", help="输入预测目录（density 计算）")
    ap.add_argument("--bootstrap", type=int, default=200, help="tiles bootstrap 次数")
    ap.add_argument("--alpha", type=float, default=0.05, help="FDR 阈值")
    return ap.parse_args()


def main():
    args = parse_args()
    run_downstream_analysis(
        config_path=args.config,
        pred_dir=args.pred_dir,
        n_bootstrap=args.bootstrap,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
