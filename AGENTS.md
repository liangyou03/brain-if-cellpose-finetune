# Repository Guidelines

## Project Structure & Module Organization
- `src/fintune/` contains core code, organized by experiment stage:
  - `data_prep/` (E0 data conversion and split handling)
  - `training/` (E1/E2 fine-tuning and budget curve)
  - `inference/` (baseline/full inference, E4 sweep)
  - `evaluation/` (E3 gating, E5 downstream stats)
  - `utils/` (I/O, pair listing, metrics helpers)
- `scripts/` provides CLI entrypoints used locally and in Slurm.
- `slurm/` stores cluster job files (currently configured for `gpu` cluster, `l40s` partition).
- `configs/paths.yaml` is the active runtime path config; start from `configs/paths.example.yaml`.
- Outputs are written under `data/` (`predictions/`, `checkpoints/`, `logs/`, `reports/`, `downstream/`).

## Build, Test, and Development Commands
- Create env: `mamba env create -f env/fintune.yaml && mamba activate fintune`
- Baseline inference: `PYTHONPATH=src mamba run -n finetune python scripts/run_cpsam_baseline.py --config configs/paths.yaml --split test`
- Generic fine-tune: `PYTHONPATH=src mamba run -n finetune python scripts/train_finetune_generic.py --config configs/paths.yaml --epochs 10`
- Budget curve (E2): `PYTHONPATH=src mamba run -n finetune python scripts/run_budget_curve.py --config configs/paths.yaml --budgets 2 5 10`
- Slurm submit example: `sbatch slurm/e5_downstream_compare.sbatch`

## Coding Style & Naming Conventions
- Python style: PEP 8, 4-space indentation, type hints for public functions.
- Prefer small, stage-specific modules over large multipurpose files.
- Use snake_case for functions/files and explicit experiment tags in outputs (example: `e3_gating_*_q0_100_a50.tsv`).
- Keep paths configurable via YAML; do not hardcode new absolute paths.

## Testing Guidelines
- There is no full pytest suite yet; use fast runtime checks before long jobs:
  - `python -m py_compile src/fintune/**/*.py scripts/*.py`
  - Run a small-sample smoke command (example: E4 with `--num_tiles 1`).
- Validate outputs by checking generated `metrics.json`/TSV files and key columns.

## Commit & Pull Request Guidelines
- Commit style in this repo is concise, imperative, and scoped by purpose (example: `init project: E0-E3 pipeline...`).
- Recommended format: `<area>: <what changed>` (example: `e5: add bootstrap stability table`).
- PRs should include:
  - objective and affected experiment stage(s) (E0–E5)
  - exact commands run
  - output paths produced
  - any cluster settings used (`partition`, `time`, `mem`, `gpu`)
