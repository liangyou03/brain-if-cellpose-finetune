#!/usr/bin/env python
"""
E0.1/E0.2: donor-level split + 转换为 Cellpose 3 通道 (DAPI, marker, zeros)。
输入：configs/paths.yaml，原始 tiff/marker/npy。
输出：train_cells/, val_cells/, test_cells/ 下成对命名的图像与实例 mask。
"""
import argparse
from fintune.data_prep.prepare_cellpose_data import prepare_cellpose_dataset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml", help="路径配置文件")
    ap.add_argument("--train_frac", type=float, default=0.6, help="donor-level train 比例")
    ap.add_argument("--val_frac", type=float, default=0.2, help="donor-level val 比例")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--format", choices=["png", "tif"], default="png", help="输出图像格式")
    return ap.parse_args()


def main():
    args = parse_args()
    prepare_cellpose_dataset(
        config_path=args.config,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        out_format=args.format,
    )


if __name__ == "__main__":
    main()
