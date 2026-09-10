#!/usr/bin/env bash
set -u -o pipefail

SNAPSHOT_ROOT=/home/h100-repro/maapacman-level1/source/curriculum2-overfit-source-sync-20260909T000000Z-v3
RUN_ROOT=/home/h100-repro/maapacman-level1/runs/curriculum2-overfit4seed-20update-20260910T112000Z

export PYTHON=/home/h100-repro/maapacman-level1/env/maapacman-rl/bin/python
export AREAL_ROOT="${SNAPSHOT_ROOT}/AReaL"
export MODEL_PATH=/home/nfs/models/Qwen3.5-9B
export CONFIG="${SNAPSHOT_ROOT}/areal-pacman/configs/level1/train/curriculum2_overfit.yaml"
export ARTIFACT_ROOT="${RUN_ROOT}"
export DATASET_OUTPUT_ROOT="${RUN_ROOT}/dataset"
export MAAPACMAN_PACMAN_PYTHON_ROOT="${SNAPSHOT_ROOT}/pacman-python"

cd "${SNAPSHOT_ROOT}/areal-pacman"
bash scripts/level1/train/run_level1_training.sh
status=$?
printf '%s\n' "${status}" > "${RUN_ROOT}/train.exit"
exit "${status}"
