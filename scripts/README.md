# Scripts

## Production Level-1

- `level1/train/`: validated training launchers and the earlier overfit
  pipeline.
- `level1/evaluate/`: checkpoint evaluation, prompt A/B, and comparison tools.
- `level1/dataset/`: deterministic dataset and manifest generation.
- `level1/report/`: trajectory auditing, checkpoint assembly, demo export, and
  report helpers.

Primary training entry point:

```bash
export OWNER_ROOT="$HOME"
export ENV_ROOT="${CONDA_PREFIX:?activate maapacman-rl first}"
export PYTHON="${ENV_ROOT}/bin/python"
export AREAL_ROOT=/path/to/AReaL
export MAAPACMAN_PACMAN_PYTHON_ROOT=/path/to/pacman-python
export MODEL_PATH=/path/to/Qwen3.5-9B
export CONFIG="$PWD/configs/level1/train/curriculum1.yaml"
export DATASET_MAX_STEPS=256
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ID="curriculum1-${RUN_TS}"
export DATASET_OUTPUT_ROOT="$PWD/artifacts/datasets/${RUN_ID}"
export ARTIFACT_ROOT="$PWD/run_artifacts/${RUN_ID}"

bash scripts/level1/train/run_level1_training.sh
```

Run a two-update smoke test without a separate config:

```bash
bash scripts/level1/train/run_level1_training.sh --smoke-updates 2
```

For Curriculum 2, select `configs/level1/train/curriculum2.yaml` and export
`CURRICULUM1_CHECKPOINT` as the complete loadable model checkpoint produced by
Curriculum 1.

Run this from the repository root after activating the `maapacman-rl` Conda
environment. Replace the three `/path/to/...` values with the fixed AReaL,
pacman-python, and local model checkouts described in the root README.

## Historical synthetic experiments

- `synthetic/`: multi-maze generation, rollout summaries, reports, and legacy
  remote supervisors.

The synthetic scripts are retained for reproducibility and are not the
production bundled-environment Level-1 launcher.
