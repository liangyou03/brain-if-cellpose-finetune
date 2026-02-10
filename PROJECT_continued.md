这些都是围绕你当前的主线：single-marker IF（每张 tile 只有 DAPI + 1 个 marker 通道；以及 marker-only 对照），并且能明显增加任务量、让故事更完整、审稿人更难挑毛病。

我先用你现有结果做一个客观定位：你已经把“domain shift → 微调显著提升分割 → label-efficient → 下游稳定性”这条链条跑通了。最关键的指标层面，baseline 在 test 上 AP50 很低但 recall 极高（典型过分割/假阳性多），微调后 precision/F1 大幅改善但 recall 降下来，说明你已经在“检测质量”上解决了主要矛盾，但还需要把“阈值/操作点选择”这件事做成系统化结果（PR 曲线、阈值 sweep、marker-specific calibration），故事会立刻更强。你目前的 baseline 与 generic finetune 的总体指标为：baseline test AP50=0.1062、P=0.0981、R=0.7864、F1=0.1744；generic finetune test AP50=0.2359、P=0.5009、R=0.5261、F1=0.5132。 

另外，一个必须直说的点：你现在的 marker-only 对照训练只跑了 8 epochs（而 generic 是 80 epochs），这在“公平对照”上会被审稿人抓住。你应该把 marker-only 也用同样 epochs/early-stopping 规则重跑，才能把性能差异归因于“是否有 DAPI”，而不是“训练不足”。目前 marker-only test 的 AP50=0.0899、P=0.4863、R=0.2659、F1=0.3438（并且 val 更低）。

---

## E1 补强：把“阈值选择/操作点”做成论文级证据（强烈建议加）

### E1.5 Threshold sweep + PR 曲线（必须加，能明显提升故事可信度）

目的：解释 baseline 的 “高 recall / 低 precision” 与 finetune 的 “precision 上来但 recall 下去”，证明你不是靠拍脑袋挑阈值，而是在验证集上系统选定操作点。

设计：

* 在 Val 上对每个模型（baseline cpsam、generic finetune、GFAP-specific、marker-only）做网格搜索：

  * flow_threshold ∈ {0.2, 0.4, 0.6}
  * cellprob_threshold ∈ {−1.0, −0.5, 0.0, 0.5, 1.0}
* 每个阈值组合计算 AP50、P/R/F1，并画：

  1. PR 曲线（或 P-R scatter）
  2. F1 vs threshold heatmap
* 选择规则（只用 Val）：以 F1 最大为主（也可以报一个 Recall≥某阈值下 Precision 最大的点，形成两种 operating points）。
* 固定阈值后再去 Test 只做一次评估。

输出：

* `reports/e1_threshold_sweep_<model>_val.tsv`
* `reports/fig_pr_curve_<model>.pdf`
* `predictions/<model>/test/metrics.json`（使用固定阈值）

预期结论：

* baseline 的高 recall 来自低阈值导致大量 FP；
* finetune 可以在相同 recall 水平下显著提高 precision（或在同等 precision 下显著提高 recall），把这个写成主结果之一。

### E1.6 训练稳定性：不同随机种子重复（建议加，成本不高）

目的：防止“你这一次跑得好只是运气”的质疑；也能跟你 E2 的重复采样逻辑形成统一风格。

设计：

* generic finetune 用 3–5 个 seed 重训（seed=2024/2025/2026…），完全同超参。
* 报告 test 上 AP50、F1 的均值±标准差。

输出：

* `reports/e1_seed_sweep_generic.tsv`
* 图：bar + errorbar（AP50/F1）

---

## E2 补强：把 label-efficiency 曲线做成“可解释”的结果（建议加）

你已有预算曲线的总体均值（val AP50 b2=0.0623、b5=0.1439、b10=0.2324、b20=0.3970、b30=0.4044）已经很有说服力，说明 20 张/marker 左右进入平台期。 现在建议补两类“让曲线更像论文图”的实验。

### E2.2 两种采样策略对比：random vs “困难样本优先”（主动学习 lite）

目的：同样预算下，证明“会挑样本”能更快逼近平台；这会让“label-efficient”更像方法贡献，而不是观察结论。

设计（不需要额外标注，只改变“选哪几张去标”）：

* 在 Train donors 内，对每个 marker 的未被选中训练图像，用 baseline 或 generic 模型先跑一次推理。
* 为每张图计算“难度分数”（任选一种即可）：

  * 预测实例数极端异常（过多或过少）
  * 平均 cellprob 接近阈值（不确定性高）
  * 预测 mask 平均面积/形状异常（碎片化）
* 预算 b={2,5,10,20}：对比

  * Random 选
  * Hard-first 选（难度分数最高的）
* 每个预算重复 5 次（hard-first 可以在 top-K 内随机抽，保留随机性）。

输出：

* `reports/e2_active_sampling_compare.tsv`
* 图：AP50 vs budget（两条曲线）

预期结论：

* 在低预算段（2/5/10）hard-first 曲线更高，说明标注资源更高效。

### E2.3 训练 compute 对等：budget curve 的 epochs 归一化

你现在的 budget curve 用 epochs=6（为了 25 次重复可接受），但审稿人可能问：是不是训练不够导致 b30 没完全饱和？
可以加一个很简单的补充实验：

* 只对 b10/b20/b30 各跑 1 次长训练（比如 epochs=40 或 80），看平台是否仍在 b20 附近。
* 作为补充材料图/表即可。

---

## E3 补强：把 gating 从“经验规则”变成“系统可复现”的消融（建议加）

你已经做了 3 组 gating 参数 sweep（q0_100/a50，q5_95/a80，q2_98/a60），并且 gating 的规则是“面积阈值 + marker 均值强度在分位数范围内”。
建议再加两步，使它更论文化。

### E3.2 Gating 参数选择只用 Val（并固定到 Test）

目的：避免“看了 Test 才挑 gating 参数”。

设计：

* 同样的 gating 网格（min_area × q_low/q_high）在 Val 上跑，选择 Val 最优（F1 或 AP50）。
* 固定参数跑 Test 并报告提升。

输出：

* `reports/e3_gating_val_grid.tsv`
* `reports/e3_gating_test_final.tsv`

### E3.3 分 marker 的 gating（GFAP vs IBA1/OLIG2 很可能需要不同 min_area）

目的：解释为什么某些 marker 更容易 FP（GFAP 纤维状背景、IBA1 小细胞碎片）。

设计：

* 对每个 marker 独立选 gating 参数（仍然只用 Val）。
* 在 Test 报 per-marker 指标变化，并强调“gating 主要提升 precision、对 recall 影响可控”。

---

## E4 补强：把 “proxy microsam sweep” 升级成真实 micro-sam 或更严谨的替代（可选但加分）

目前你做的是 proxy：用 cpsam 的 flow_threshold 映射模拟 pps=16/32/64。
如果你想把这块放进主文而不是补充材料，建议二选一：

### 方案 A：真正安装 micro-sam 跑一次（最干净）

* 同样抽样策略（每 marker 10 张），真实 sweep pps=16/32/64
* 输出 recall、FP、AP50
* 结论：密集小目标对 prompt 密度敏感；默认 prompt 不稳定。

### 方案 B：把它改写成“Cellpose-SAM 推理参数敏感性分析”（更诚实）

* 不叫 micro-sam，而是明确：我们系统测试了 flow_threshold / cellprob_threshold 对密集场景的敏感性（呼应 E1.5）
* 这会和你阈值 sweep 合并成一块“推理稳定性分析”。

---

## E5 补强：把下游分析做成“三层稳健性”证据链（必须加，最能讲故事）

你 E5 的目标设计很清楚：从 tile counts 汇总 donor-level ratio，做 Spearman + BH-FDR，并做 tile bootstrap（80% tiles, 200 次）。 这里建议加三个“审稿人常问的稳健性”。

### E5.4 表型定义敏感性（必须加，成本低）

你现在 marker-positive 的定义是：实例内 marker 均值 > tile 内 marker 的 75 分位数，然后 ratio = n_pos / n_total。
审稿人会问：为什么是 75 分位？换阈值会不会结论变？

设计：

* 在全量推理输出不变的情况下，只改“pos 判定阈值”：

  * percentile ∈ {60, 70, 75, 80, 90}
  * 或者用 Otsu / mixture model（每 tile 或每 donor）做强度阈值
* 对每个阈值重复下游 Spearman + FDR，并重复 bootstrap 稳定性（可以把 bootstrap 次数减半用于敏感性分析）。

输出：

* `reports/e5_threshold_sensitivity.tsv`
* 图：rho vs threshold、显著性通过率 vs threshold

预期结论：

* finetune 的下游相关更“阈值不敏感”（曲线更平、更稳），这是“可靠性提升”的强证据。

### E5.5 加入混杂调整（强烈建议加，尤其你是 biostat 背景）

目的：把“相关性更强”升级为“在合理混杂控制下仍更稳健”。

设计（按你现有临床表字段能做到什么就做什么）：

* 对每个 outcome，用线性/有序回归或 rank-based regression：

  * response: pathology/cognition
  * predictor: donor-level density/ratio
  * covariates: age、sex、PMI、batch/slide（若有）、brain weight（能拿到就加）
* 同样做 bootstrap，比较系数方差、符号一致性、显著性通过率。

输出：

* `reports/e5_adjusted_models.tsv`
* 图：adjusted beta 分布（baseline vs finetune）

预期结论：

* finetune 的效应估计更稳定，显著性更一致。

### E5.6 空间聚集/异质性特征（加分，单 marker 也能做）

目的：从“只看 cell density”扩展到“空间组织结构”，让下游更丰富。

设计：

* 每 tile 计算：

  * 邻近图（kNN）下的平均最近邻距离（NND）
  * Ripley’s K 或简化版聚集指数
  * 大小/形状分布（面积、圆度）作为细胞形态 proxy
* donor-level 聚合（均值、分位数、变异系数）。
* 与病理/认知做同样的相关与稳定性分析。

输出：

* `downstream/donor_marker_spatial_features.tsv`
* `reports/e5_spatial_assoc.tsv`
* 图：新增特征的相关热图 + 稳定性对比

预期结论：

* 分割更准确不仅提升“数量”，也提升“空间统计”的信噪比，从而在病理关联中更稳定。

---

## E6 新增：Cross-marker generalization（不需要新数据，任务量大，故事很强）

你现在主模型是“四类 marker 合并训练的 generic”。 你可以加一个很强的“泛化”模块：训练时故意不看某个 marker，测试时看它，证明模型学到的是“组织形态学 + 核结构”的可迁移知识，而不只是记住某个 marker 的外观。

### E6.1 Leave-one-marker-out (LOMO)

设计：

* 4 次训练：

  * Train 用 {IBA1, NeuN, OLIG2}，Test 在 GFAP
  * Train 用 {GFAP, NeuN, OLIG2}，Test 在 IBA1
  * …
* Val 仍用训练 marker 内部的 val（或留出一小部分被留出 marker 的 val 但不用于训练）。
* 输出 per-marker AP50/P/R/F1，与 fully-trained generic 对比。

输出：

* `reports/e6_lomo_metrics.tsv`
* 图：每个 marker 上 “full generic vs LOMO” 的差值条形图

预期结论：

* 如果 LOMO 仍能有 decent AP50，说明模型在 single-marker IF 场景学到可迁移结构信息（尤其 DAPI 帮助很大），这是你“foundation model + 少量适配”的强论点。

### E6.2 Marker-specific vs Generic 的“数据效率对比”

设计：

* 例如 GFAP：用相同标注数，比较

  * marker-specific 微调
  * generic 微调（共享数据）
  * LOMO→GFAP 再少量 GFAP 微调（两阶段适配）
* 这能讲一个更漂亮的故事：先用其他 marker 学通用形态，再用极少 GFAP 标注做轻量适配。

---

## E7 新增：Active learning（真正的“label-efficient 方法贡献”，不引入新数据）

你现在 E2 的 random budgets 已经证明“少标注能起效”。 下一步是证明“同样标注预算，用更聪明的挑样本策略更好”，这会把论文从“经验观察”拉到“方法学贡献”。

### E7.1 不确定性驱动的标注选择

设计：

* 用 baseline 或 generic 在 Train donors 上推理（未标注 tiles）。
* 计算 tile-level uncertainty（任选其一）：

  * 预测实例数对阈值敏感：在两组阈值下实例数差异大
  * cellprob 分布集中在阈值附近
  * 简单 ensemble（不同 seed 的模型）预测 IoU 分歧大
* 在每个预算 b 下选择 top-b uncertainty tiles 去标（你已经有标注池的话，就是从标注池里模拟选择；没有标注池就先用现有 30 张/marker 当 pool）。
* 与 random 相比。

输出：

* `reports/e7_active_learning_curve.tsv`
* 图：budget vs AP50（AL vs random）

---

## E8 新增：Pseudo-label / semi-supervised（可选但任务量大、也很像“下一篇”）

你有 14,749 tiles，但只有 120 张标注。用 pseudo-label 能把“数据规模优势”真正吃到。

### E8.1 Self-training（伪标签自训练）

设计：

1. 用 generic finetune 在未标注 tiles 推理，生成伪标签。
2. 伪标签筛选：

   * 通过 gating（面积、marker 强度）过滤低质量实例（你 E3 的 gating 规则可直接复用）。
   * 只保留 tile-level 质量高的（例如预测实例数不极端、平均 mask 形状合理）。
3. 用 “真标注 + 伪标注” 再训练一轮（可用较小 lr、冻结部分 backbone）。
4. 在 Test 标注集评估 AP50/P/R/F1；在 E5 下游评估稳定性是否进一步提升。

输出：

* `checkpoints/generic_selftrain_round1/`
* `reports/e8_selftrain_metrics.tsv`
* 下游：`reports/e8_downstream_compare.tsv`

预期结论：

* 分割指标再小幅提升，或在相同分割指标下下游稳定性提升（更可能）。

---

## E9 新增：PEFT/冻结策略（专门为“少标注下防过拟合”的审稿人问题准备）

你现在 generic 是直接从 cpsam 预训练继续训练。 可以加一个“参数高效微调”对照：少标注时只更新少量参数，性能更稳。

### E9.1 Full fine-tune vs Freeze-backbone vs LoRA/Adapter（3 组就够）

设计：

* b=2/5/10 这几个低预算最能体现差异：

  * Full fine-tune（你现有）
  * Freeze 大部分 encoder，只训 head / 少数层
  * LoRA/adapter（如果工程成本太高可跳过，至少做 freeze 对照）
* 比较：

  * Val AP50 的均值±方差（重复 5 次）
  * 最终 Test 指标（可选）

输出：

* `reports/e9_peft_low_budget.tsv`

预期结论：

* 低预算下 freeze/PEFT 方差更小、性能更稳（即使均值略低，也能作为“稳定性/可复现性”卖点）。

---

## E10 新增：利用 DAPI 做“核引导的胞体分裂/纠错”（非常贴 single-marker IF）

你的数据天然有 DAPI，这在 single-marker IF 场景是最强先验：核的位置应该是一细胞一核（至少对 NeuN/OLIG2/IBA1 常常成立），可用于拆分粘连或过滤假阳性。

### E10.1 Nuclear-seeded splitting（后处理升级版）

设计：

1. 先对 DAPI 做核分割（可以用 Cellpose nuclei 模型或简单阈值+watershed）。
2. 对胞体预测 mask：

   * 若一个胞体包含多个核，则按核种子做 watershed 拆分（纠正 under-segmentation）
   * 若一个胞体不包含核（或核强度极低），作为 FP 候选（尤其对 NeuN/OLIG2 很合理）
3. 在 Test 评估 AP50/P/R/F1；并看对 E5 下游是否更稳。

输出：

* `reports/e10_nucleus_guided_refine.tsv`
* `predictions/<model>/test/*_nucguided_pred.tif`

预期结论：

* Precision 上升或粘连情况减少，且对下游稳定性有额外增益。

---

# 最后给你一个“优先级/性价比”建议（不需要额外数据也能把故事讲大）

第一优先（我认为必须加，直接提升论文说服力）

1. E1.5 阈值 sweep + PR 曲线（把操作点选择规范化）
2. marker-only 重新按同样训练策略重跑（修掉公平性问题）
3. E5.4 表型定义敏感性（75 分位阈值的稳健性）

第二优先（任务量明显增加，审稿加分）
4) E6 LOMO 跨 marker 泛化（不引入新数据但贡献感强）
5) E5.5 混杂调整模型 + bootstrap 稳定性
6) E2.2 “困难样本优先” vs random（主动学习 lite）

第三优先（更像下一篇/补充材料，但能拉高上限）
7) E8 伪标签 self-training
8) E10 核引导纠错（非常贴 IF 数据特点）
9) E9 PEFT/冻结策略（低预算稳定性）