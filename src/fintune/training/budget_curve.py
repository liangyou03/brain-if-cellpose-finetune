from __future__ import annotations

from pathlib import Path
import json
import shutil
import random
from datetime import datetime
import yaml

from fintune.utils.dataset import list_pairs, parse_marker_donor
from fintune.training.finetune_generic import finetune_generic
from fintune.inference.cpsam_baseline import run_baseline_inference


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


def run_budget_curve(
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
    exp_name: str = "budget_curve",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    train_dir = Path(cfg["cellpose_train_dir"])
    val_dir = Path(cfg["cellpose_val_dir"])
    test_dir = Path(cfg["cellpose_test_dir"])
    base_dir = Path(cfg["logs_dir"]) / exp_name
    base_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = list_pairs(train_dir)
    grouped = _group_by_marker(all_pairs)
    markers = sorted(grouped.keys()) if marker == "all" else [marker.lower()]
    rng = random.Random(seed)

    records = []
    for budget in budgets:
        for rep in range(repeats):
            run_id = f"b{budget}_r{rep}"
            run_dir = base_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            model_name = f"budget_{run_id}"
            metrics_path = run_dir / "predictions" / model_name / eval_split / "metrics.json"
            if skip_existing and metrics_path.exists():
                metrics = _load_metrics(metrics_path)
                rec = {
                    "budget": budget,
                    "repeat": rep,
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
                print(f"[budget_curve] skip existing budget={budget} repeat={rep}")
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
                subset_pairs.extend(rng.sample(cand, k))

            subset_train = run_dir / "train_subset"
            _write_subset_dir(subset_pairs, subset_train)

            run_cfg = dict(cfg)
            run_cfg["cellpose_train_dir"] = str(subset_train)
            run_cfg["checkpoints_dir"] = str(run_dir / "checkpoints")
            run_cfg["predictions_dir"] = str(run_dir / "predictions")
            run_cfg_path = run_dir / "paths.yaml"
            with open(run_cfg_path, "w") as f:
                yaml.safe_dump(run_cfg, f)

            # Respect budget size: if cap is provided, do not exceed subset size.
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
            )

            run_baseline_inference(
                config_path=str(run_cfg_path),
                model=saved_model,
                split=eval_split,
                flow_threshold=0.4,
                cellprob_threshold=0.0,
            )
            metrics_path = run_dir / "predictions" / Path(saved_model).name / eval_split / "metrics.json"
            metrics = _load_metrics(metrics_path)

            rec = {
                "budget": budget,
                "repeat": rep,
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
            print(f"[budget_curve] done budget={budget} repeat={rep} train_n={len(subset_pairs)}")

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

    summary = {}
    for budget in budgets:
        subset = [r for r in records if r["budget"] == budget]
        if not subset:
            continue
        for key in ["overall_ap50", "overall_precision", "overall_recall", "overall_f1"]:
            vals = [float(r[key]) for r in subset]
            summary.setdefault(budget, {})[key] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }
    with open(base_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(base_dir / f"summary_{run_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[budget_curve] records={len(records)}")
    print(f"[budget_curve] saved={out_tsv}")
    print(f"[budget_curve] saved_tagged={out_tsv_tagged}")


__all__ = ["run_budget_curve"]
