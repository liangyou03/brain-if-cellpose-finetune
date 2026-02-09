# Fine-tune Cellpose-SAM for ROSMAP Brain IF

本项目目标：基于 ROSMAP 人脑 IF 图像（DAPI + 单 marker）做 Cellpose-SAM 的 label-efficient 微调，并证明分割改进能提升下游病理/认知关联的稳定性与可靠性。

---

## 1. 项目目标与假设

1. 适配性：Cellpose-SAM 的默认模型在 ROSMAP 数据存在 domain shift（组织切片、弱边界、复杂形态）。
2. Label efficiency：少量标注也能显著提升分割性能。
3. 下游价值：分割更稳健会提升 donor-level 特征与病理/认知指标的相关性稳定性。

---

## 2. 数据概况

**数据来源与规模**
- ROSMAP 人脑 DLPFC IF 图像。
- 69 位 donor。
- 14,749 张 tiles（每张 1,040 × 1,388 像素）。
- 5 个 marker：GFAP / IBA1 / NeuN / OLIG2 / PECAM。
- 当前细胞体分割只使用 GFAP/IBA1/NeuN/OLIG2（PECAM 不做 cell body）。

**标注**
- 每 marker 30 张图像（四类共 120 张）细胞体实例标注。
- 每张 mask：实例级，0=背景，1..N=细胞 ID。

---

## 3. 输入格式与规范

**Cellpose-SAM 训练/推理输入**
- 3 通道图像：[DAPI, marker, zeros]。
- 形式：tif / png。
- mask：实例 mask tif（0 背景，1..N 细胞 id）。

**Marker-only 输入（新增实验）**
- 3 通道图像：[marker, zeros, zeros]（不使用 DAPI）。
- 与主实验保持同样的 mask、split、训练轮次与评估流程。
- 用于与 `[DAPI, marker, 0]` 方案做公平对照，评估 DAPI 对细胞体分割的贡献。

**命名规则**
- 图像：`<marker>_<donor>.<ext>`
- mask：`<marker>_<donor>_mask.tif`

**根路径**
- 默认数据根目录：`/ix/jbwang/liangyou/fintune/data`

---

## 4. 实验任务（E0–E5）

### E0. 数据准备（必须）

**E0.1 Donor-level split**
- 按 donor 划分 train/val/test。

**E0.2 转换为 Cellpose 训练格式**
- 输出目录：
  - `train_cells/`
  - `val_cells/`
  - `test_cells/`
- 每个目录内：图像 + mask 成对保存。

---

### E1. 分割主实验（必须）

**E1.1 Baseline**
- 使用 off-the-shelf Cellpose-SAM（`cpsam`）。
- 在 Test 标注集评估：
  - AP@0.5
  - Precision / Recall / F1
- 保存预测 mask 便于可视化。

**E1.2 Fine-tune 通用模型**
- Train：四类 marker 合并。
- Val：选择 checkpoint + 阈值（禁止使用 Test 调参）。
- Test：报告指标和提升幅度。

**E1.3 Fine-tune marker-specific（可选加分）**
- 仅 GFAP：Train/Val/Test。
- 目标：证明复杂形态细胞适配收益更大。

**E1.4 Fine-tune marker-only（新增，建议做）**
- 目标：验证“不使用 DAPI”时模型性能变化，量化 DAPI 通道价值。
- 训练输入：仅 marker 通道（3 通道打包为 `[marker, 0, 0]`）。
- 对照设置：
  - 对照组 A：`[DAPI, marker, 0]`（当前主方案）
  - 对照组 B：`[marker, 0, 0]`（新增方案）
- 其余条件保持一致：
  - donor-level split 不变
  - train/val/test 样本不变
  - 超参数不变（epochs/lr/batch/阈值选择规则）
- 输出：
  - 各 marker AP@0.5 / Precision / Recall / F1（Test）
  - `marker-only` 相对主方案的差值表（ΔAP/ΔP/ΔR/ΔF1）
  - 重点观察 GFAP/OLIG2 是否对 DAPI 依赖更高

---

### E2. Label Efficiency（必须）

**E2.1 标注预算曲线**
- Budget 方案：
  - 每 marker {2, 5, 10, 20, 30}
  - 或总张数 {8, 20, 40, 80, 120}
- 每个 budget 随机重复 5 次（仅从 Train donors 中抽）。
- 评估在 Val（建议）或 Test。
- 输出：
  - 各 marker AP@0.5 vs budget（均值±区间）
  - Recall vs budget（重点关注 GFAP/IBA1）

---

### E3. 推理后处理（建议）

**E3.1 Marker gating ablation**
- 同一模型输出：
  - 不加 gating
  - 加 marker 强度阈值 + 面积过滤
- 在 Test 对比 AP / Precision / Recall。
- 目标：减少 FP，提高 precision。

---

### E4. μSAM / micro-sam 诊断（可选）

**E4.1 Prompt sweep**
- 每 marker 抽 10 张（总 40 张）。
- 参数 sweep：points-per-side = 16/32/64。
- 评估 Recall + FP 或 AP@0.5。
- 目标：证明默认提示在密集小目标场景不稳定。

---

### E5. ROSMAP 下游价值（必须）

**E5.1 全数据推理 + donor-level 汇总**
- baseline + fine-tuned 各对 14,749 tiles 推理。
- 每 donor × marker 计算 cell density（建议面积标准化）。
- 输出两份 donor×marker 特征表。

**E5.2 下游相关性**
- Spearman：cell densities vs Braak / ADNC / cognition。
- 输出：每 marker 的 rho + FDR。

**E5.3 稳定性（强烈建议）**
- 对每 donor 进行 tiles bootstrap（80% tiles, 200 次）。
- 每次计算相关系数或回归系数。
- 对比 baseline vs fine-tuned：
  - 方差
  - sign consistency
  - 显著性通过率（FDR<0.05）
- 输出稳定性对比图：箱线图 / 条形图。

---

## 5. 目录结构

```
/ix/jbwang/liangyou/fintune
├── env/
│   └── fintune.yaml
├── configs/
│   ├── paths.example.yaml
│   └── paths.yaml
├── scripts/
│   ├── prepare_cellpose_data.py
│   ├── run_cpsam_baseline.py
│   ├── train_finetune_generic.py
│   ├── train_finetune_gfap.py
│   ├── run_budget_curve.py
│   ├── run_gating_ablation.py
│   ├── run_microsam_sweep.py
│   ├── run_full_inference.py
│   └── analyze_downstream.py
├── src/fintune/
│   ├── data_prep/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   └── utils/
├── notebooks/
└── reports/
```

---

## 6. 配置文件

`configs/paths.yaml` 核心字段：
- `data_root`
- `raw_tiles_dir`
- `brain_data_dir`
- `splits_dir`
- `cellpose_train_dir`
- `cellpose_val_dir`
- `cellpose_test_dir`
- `predictions_dir`
- `checkpoints_dir`
- `logs_dir`
- `downstream_dir`
- `reports_dir`

---

## 7. 环境与依赖

**环境**
- `finetune`（mamba/conda）

**核心依赖**
- python 3.10
- cellpose 4.x
- pytorch 2.2
- tifffile / numpy / scikit-image / pandas / matplotlib
- micro-sam（E4 可选）

---

## 8. 关键运行流程（建议顺序）

1. 生成 donor-level split + Cellpose 格式数据。
2. 跑 baseline（E1.1）。
3. 跑通用 fine-tune（E1.2）。
4. 评估并生成提升幅度。
5. 运行 marker-only 对照实验（E1.4）。
6. 运行 label efficiency（E2）。
7. 运行 gating ablation（E3）。
8. 运行全量推理 + downstream 分析（E5）。

---

## 9. 指标定义

- AP@0.5：Cellpose `average_precision`。
- Precision/Recall/F1：由 TP/FP/FN 汇总计算。
- 下游相关性：Spearman 相关 + FDR。
- 稳定性：bootstrap 方差、方向一致率、显著性通过率。

---

## 10. 风险与注意事项

1. Donor-level split 必须严格执行，避免泄漏。
2. Val 与 Test 用途分离，禁止在 Test 调参。
3. PECAM 不做细胞体分割。
4. 复杂形态 marker（GFAP）可能需更小 min_size 或更严格 gating。
5. micro-sam 在密集小目标场景很容易出现 FP 上升或 recall 不稳定。

---

## 11. 输出规范

- `predictions/<model>/<split>/metrics.json`
- `predictions/<model>/<split>/*_pred.tif`
- `checkpoints/<model>/`
- `reports/` 下保存曲线图和表格

---

## 12. 可扩展方向

- 引入更强的后处理规则（marker gating + size + shape prior）。
- 加入 semi-supervised 或 pseudo-label 改进 label efficiency。
- 将 donors 分类特征用于 cluster 与病理 subgroup 分析。

---

## 13. 一句话总结

这是一个围绕 “label-efficient 微调 + 下游稳定性提升” 的完整实验链路，Cellpose-SAM 是工具，ROSMAP 是验证场景，目标是把分割改进转化为更可靠的病理结论。
