# Scripts

## Production Level-1

- `level1/train/`: training launchers.
- `level1/dataset/`: deterministic dataset and manifest generation.
- `level1/evaluate/`: checkpoint evaluation and comparison tools.
- `level1/report/`: trajectory audit, checkpoint, demo, and report helpers.

Run Curriculum 1 from the repository root after activating the training
environment:

```bash
export PYTHON="${CONDA_PREFIX:?activate maapacman-rl first}/bin/python"
export AREAL_ROOT=/path/to/AReaL
export MAAPACMAN_PACMAN_PYTHON_ROOT=/path/to/pacman-python
export MODEL_PATH=/path/to/Qwen3.5-9B
export CONFIG="$PWD/configs/level1/train/level1_curriculum1_step256_100update_group12_8gpu.yaml"
export RUN_ID="curriculum1-$(date -u +%Y%m%dT%H%M%SZ)"
export DATASET_OUTPUT_ROOT="$PWD/artifacts/datasets/${RUN_ID}"
export ARTIFACT_ROOT="$PWD/run_artifacts/${RUN_ID}"
bash scripts/level1/train/run_level1_training.sh
```

For the two-update Curriculum 1 smoke test, add `SMOKE_TEST=1`. Curriculum 2
uses `level1_curriculum2_step256_100update_group12_8gpu.yaml` and requires
`CURRICULUM1_CHECKPOINT` to point to the complete checkpoint produced by
`level1/report/build_complete_vlm_checkpoint.py`.
Standalone evaluation uses `evaluate_level1.py --curriculum 1` or `2`.

The historical `synthetic/` scripts are retained for reproducibility and are
not the production bundled-environment path.
