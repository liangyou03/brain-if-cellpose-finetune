# Fintune Cellpose-SAM for ROSMAP IF

轻量但清晰的项目骨架，用 mamba 管理环境，围绕 E0–E5 任务设计。数据位于 `/ihome/jbwang/liy121/fintune/data`（DAPI + 单 marker 通道）。

## 快速开始
1. 创建并激活环境（推荐 `mamba`）  
   ```bash
   mamba env create -f env/fintune.yaml
   mamba activate fintune
   ```
2. 复制并编辑路径配置  
   ```bash
   cp configs/paths.example.yaml configs/paths.yaml
   # 根据实际数据/输出位置调整
   ```
3. （可选）在 `notebooks/` 下新建 exploratory notebook，使用 `configs/paths.yaml` 中的路径。

## 目录结构（简洁而分阶段）
- `env/`：mamba 环境定义。  
- `configs/`：路径/实验超参模板（目前仅 paths）。  
- `scripts/`：CLI 入口（数据预处理、训练、推理、评估）。  
- `src/fintune/`：按任务模块化的源码。  
  - `data_prep/`：donor-level split、转换成 Cellpose 3-channel（DAPI, marker, 0）。  
  - `training/`：baseline 推理包装、fine-tune、label-budget 实验。  
  - `inference/`：全量推理、后处理（marker gating）。  
  - `evaluation/`：分割指标、下游相关性与稳定性。  
  - `utils/`：通用工具（I/O、logging、metrics 封装）。
- `notebooks/`：探索/可视化。  
- `reports/`：图表与表格输出。

## 任务到代码的映射
- **E0 数据准备**  
  - `scripts/prepare_cellpose_data.py` → 调 `fintune.data_prep.prepare_cellpose_data`：donor-level split + 转 3 通道 tif/png + 生成 `train/val/test` 成对文件。  
- **E1 Baseline & Fine-tune**  
  - `scripts/run_cpsam_baseline.py`：在 Test 集跑 off-the-shelf Cellpose-SAM，输出指标+mask。  
  - `scripts/train_finetune_generic.py`：四类 marker 合并 fine-tune；Val 选 ckpt/阈值。  
  - （可选）`scripts/train_finetune_gfap.py`：GFAP 专用模型。  
- **E2 Label efficiency**  
  - `scripts/run_budget_curve.py`：按预算抽样→fine-tune→评估，输出 AP/Recall vs budget 图。  
- **E3 后处理/ablation**  
  - `scripts/run_gating_ablation.py`：同一模型输出，加/不加 marker 强度阈值+面积过滤，对比指标。  
- **E4 micro-sam/μSAM 诊断（可选）**  
  - `scripts/run_microsam_sweep.py`：points-per-side sweep，输出曲线/表。  
- **E5 下游分析**  
  - `scripts/run_full_inference.py`：对 14,749 tiles 全量推理（baseline & fine-tune）。  
  - `scripts/analyze_downstream.py`：donor×marker density，Spearman 与稳定性（bootstrap）。

## 最小工作流提示
1. 先跑 `prepare_cellpose_data.py` 生成 train/val/test。  
2. 用 `run_cpsam_baseline.py` 得到 E1.1 指标，保存预测。  
3. `train_finetune_generic.py` → `run_cpsam_baseline.py --model <ckpt>` 在 Test 上评估提升幅度。  
4. 依次扩展到 label budget、GFAP 专用、gating ablation、downstream 相关性。

## 约定
- 通道顺序：`[DAPI, marker, zeros]`，与官方 Cellpose-SAM 设置兼容。  
- 命名：图像与 mask 必须成对，示例 `sample_001.png` 与 `sample_001_mask.tif`。  
- 指标：默认计算 AP@0.5、Precision、Recall、F1（可在代码中改阈值）。  
- 日志/检查点：默认写到 `configs/paths.yaml` 中的目录。

如需补充或调整，请告诉我，我们可以在现有骨架上继续加功能而不让结构变复杂。
