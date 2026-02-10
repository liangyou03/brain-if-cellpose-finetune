from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import random
import shutil

import numpy as np
import yaml
from cellpose import models

from fintune.inference.cpsam_baseline import run_baseline_inference
from fintune.training.finetune_generic import finetune_generic
from fintune.utils.dataset import list_pairs, parse_marker_donor, read_image


def _group_by_marker(pairs):
    grouped = {}
    for img_path, mask_path in pairs:
        marker, _ = parse_marker_donor(img_path.stem)
        grouped.setdefault(marker, []).append((img_path, mask_path))
    return grouped


def _write_subset_dir(subset_pairs, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path, mask_path in subset_pairs:
        shutil.copy2(img_path, out_dir / img_path.name)
        shutil.copy2(mask_path, out_dir / mask_path.name)


def _load_metrics(metrics_path: Path):
    with open(metrics_path, "r") as f:
        return json.load(f)


def _to_three_channel(img: np.ndarray):
    if img.ndim == 2:
        return np.stack([img, img, np.zeros_like(img)], axis=-1)
    return img


def _count_instances(mask: np.ndarray):
    ids = np.unique(mask)
    return int((ids > 0).sum())


def _difficulty_scores(
    pairs,
    model_name: str,
    flow_a: float,
    cellprob_a: float,
    flow_b: float,
    cellprob_b: float,
):
    model_obj = models.CellposeModel(gpu=True, pretrained_model=model_name)
    scores = {}
    for img_path, _ in pairs:
        img = _to_three_channel(read_image(img_path))
        pred_a, _, _ = model_obj.eval(
            img,
            channels=[1, 2],
            diameter=None,
            flow_threshold=flow_a,
            cellprob_threshold=cellprob_a,
        )
        pred_b, _, _ = model_obj.eval(
            img,
            channels=[1, 2],
            diameter=None,
            flow_threshold=flow_b,
            cellprob_threshold=cellprob_b,
        )
        n_a = _count_instances(pred_a)
        n_b = _count_instances(pred_b)
        # Sensitivity of instance count to inference thresholds:
        # larger value => less stable sample => harder sample.
        diff = abs(n_a - n_b)
        score = diff / float(n_a + n_b + 1)
        scores[str(img_path)] = {
            "score": float(score),
            "n_inst_a": int(n_a),
            "n_inst_b": int(n_b),
        }
    return scores


def _sample_marker_subset(
    cand_pairs,
    k: int,
    strategy: str,
    scores,
    rng: random.Random,
    hard_pool_factor: int = 3,
):
    if k <= 0 or len(cand_pairs) == 0:
        return []
    if strategy == "random":
        return rng.sample(cand_pairs, k=min(k, len(cand_pairs)))

    ranked = sorted(
        cand_pairs,
        key=lambda x: float(scores.get(str(x[0]), {}).get("score", 0.0)),
        reverse=True,
    )
    kk = min(k, len(ranked))
    pool_k = min(len(ranked), max(kk, kk * hard_pool_factor))
    pool = ranked[:pool_k]
    if len(pool) <= kk:
        return pool
    return rng.sample(pool, k=kk)


def _summarize(records, budgets, strategies):
    out = {}
    for budget in budgets:
        out[budget] = {}
        for strategy in strategies:
            subset = [r for r in records if r["budget"] == budget and r["strategy"] == strategy]
            if len(subset) == 0:
                continue
            out[budget][strategy] = {}
            for key in ["overall_ap50", "overall_precision", "overall_recall", "overall_f1"]:
                vals = [float(r[key]) for r in subset]
                out[budget][strategy][key] = {
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                }
        if "random" in out[budget] and "hard" in out[budget]:
            out[budget]["delta_hard_minus_random"] = {
                key: out[budget]["hard"][key]["mean"] - out[budget]["random"][key]["mean"]
                for key in ["overall_ap50", "overall_precision", "overall_recall", "overall_f1"]
            }
    return out


def run_active_sampling_compare(
    config_path: str,
    budgets,
    repeats: int = 5,
    epochs: int = 60,
    lr: float = 1e-4,
    batch_size: int = 2,
    marker: str = "all",
    seed: int = 2024,
    eval_split: str = "val",
    nimg_per_epoch: int | None = 24,
    nimg_test_per_epoch: int | None = 12,
    skip_existing: bool = True,
    exp_name: str = "active_sampling_compare",
    scoring_model: str = "cpsam",
    flow_a: float = 0.4,
    cellprob_a: float = 0.0,
    flow_b: float = 0.2,
    cellprob_b: float = -0.5,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_dir = Path(cfg["cellpose_train_dir"])
    val_dir = Path(cfg["cellpose_val_dir"])
    base_dir = Path(cfg["logs_dir"]) / exp_name
    base_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = list_pairs(train_dir)
    grouped = _group_by_marker(all_pairs)
    markers = sorted(grouped.keys()) if marker == "all" else [marker.lower()]

    print("[e2.2] scoring sample difficulty ...")
    scores = _difficulty_scores(
        pairs=all_pairs,
        model_name=scoring_model,
        flow_a=flow_a,
        cellprob_a=cellprob_a,
        flow_b=flow_b,
        cellprob_b=cellprob_b,
    )

    score_out = base_dir / "difficulty_scores.tsv"
    with open(score_out, "w") as f:
        f.write("img_path\tscore\tn_inst_a\tn_inst_b\n")
        for p in sorted(scores.keys()):
            s = scores[p]
            f.write(f"{p}\t{s['score']}\t{s['n_inst_a']}\t{s['n_inst_b']}\n")

    strategies = ["random", "hard"]
    records = []
    for budget in budgets:
        for rep in range(repeats):
            for strategy in strategies:
                run_seed = int(seed + budget * 1000 + rep * 100 + (1 if strategy == "hard" else 0))
                rng = random.Random(run_seed)
                run_id = f"b{budget}_r{rep}_{strategy}"
                run_dir = base_dir / run_id
                model_name = f"{exp_name}_{run_id}"
                metrics_path = run_dir / "predictions" / model_name / eval_split / "metrics.json"

                if skip_existing and metrics_path.exists():
                    metrics = _load_metrics(metrics_path)
                    rec = {
                        "budget": budget,
                        "repeat": rep,
                        "strategy": strategy,
                        "seed": run_seed,
                        "train_n": len(list_pairs(run_dir / "train_subset")) if (run_dir / "train_subset").exists() else "",
                        "eval_split": eval_split,
                        "model_path": str(run_dir / "checkpoints" / model_name / "models" / model_name),
                        "overall_ap50": metrics["overall"]["ap50"],
                        "overall_precision": metrics["overall"]["precision"],
                        "overall_recall": metrics["overall"]["recall"],
                        "overall_f1": metrics["overall"]["f1"],
                    }
                    for m in ["gfap", "iba1", "neun", "olig2"]:
                        if m in metrics:
                            rec[f"{m}_ap50"] = metrics[m]["ap50"]
                            rec[f"{m}_recall"] = metrics[m]["recall"]
                    records.append(rec)
                    print(f"[e2.2] skip existing budget={budget} repeat={rep} strategy={strategy}")
                    continue

                if run_dir.exists():
                    shutil.rmtree(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)

                subset_pairs = []
                for m in markers:
                    cand = grouped.get(m, [])
                    if len(cand) == 0:
                        continue
                    k = min(budget, len(cand))
                    picked = _sample_marker_subset(
                        cand_pairs=cand,
                        k=k,
                        strategy=strategy,
                        scores=scores,
                        rng=rng,
                    )
                    subset_pairs.extend(picked)

                subset_train = run_dir / "train_subset"
                _write_subset_dir(subset_pairs, subset_train)

                run_cfg = dict(cfg)
                run_cfg["cellpose_train_dir"] = str(subset_train)
                run_cfg["checkpoints_dir"] = str(run_dir / "checkpoints")
                run_cfg["predictions_dir"] = str(run_dir / "predictions")
                run_cfg_path = run_dir / "paths.yaml"
                with open(run_cfg_path, "w") as f:
                    yaml.safe_dump(run_cfg, f)

                local_nimg_per_epoch = None
                if nimg_per_epoch is not None:
                    local_nimg_per_epoch = min(int(nimg_per_epoch), len(subset_pairs))
                local_nimg_test_per_epoch = nimg_test_per_epoch
                if nimg_test_per_epoch is not None:
                    local_nimg_test_per_epoch = min(int(nimg_test_per_epoch), len(list_pairs(val_dir)))

                saved_model = finetune_generic(
                    config_path=str(run_cfg_path),
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                    resume_ckpt=None,
                    model_name=model_name,
                    marker_filter=None if marker == "all" else marker.lower(),
                    nimg_per_epoch=local_nimg_per_epoch,
                    nimg_test_per_epoch=local_nimg_test_per_epoch,
                    seed=run_seed,
                )

                run_baseline_inference(
                    config_path=str(run_cfg_path),
                    model=saved_model,
                    split=eval_split,
                    flow_threshold=0.4,
                    cellprob_threshold=0.0,
                )
                metrics = _load_metrics(metrics_path)

                rec = {
                    "budget": budget,
                    "repeat": rep,
                    "strategy": strategy,
                    "seed": run_seed,
                    "train_n": len(subset_pairs),
                    "eval_split": eval_split,
                    "model_path": str(saved_model),
                    "overall_ap50": metrics["overall"]["ap50"],
                    "overall_precision": metrics["overall"]["precision"],
                    "overall_recall": metrics["overall"]["recall"],
                    "overall_f1": metrics["overall"]["f1"],
                }
                for m in ["gfap", "iba1", "neun", "olig2"]:
                    if m in metrics:
                        rec[f"{m}_ap50"] = metrics[m]["ap50"]
                        rec[f"{m}_recall"] = metrics[m]["recall"]
                records.append(rec)
                print(f"[e2.2] done budget={budget} repeat={rep} strategy={strategy} train_n={len(subset_pairs)}")

    out_tsv = base_dir / "records.tsv"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tsv_tagged = base_dir / f"records_{run_tag}.tsv"
    if records:
        keys = sorted(records[0].keys())
        for path in [out_tsv, out_tsv_tagged]:
            with open(path, "w") as f:
                f.write("\t".join(keys) + "\n")
                for r in records:
                    f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")

    summary = _summarize(records=records, budgets=budgets, strategies=strategies)
    with open(base_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(base_dir / f"summary_{run_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[e2.2] records={len(records)}")
    print(f"[e2.2] saved={out_tsv}")
    print(f"[e2.2] saved_tagged={out_tsv_tagged}")
    print(f"[e2.2] difficulty_scores={score_out}")


__all__ = ["run_active_sampling_compare"]
