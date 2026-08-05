#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/runs/generated/level1_overfit}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
BASELINE_URL="${BASELINE_URL:-}"
POST_TRAIN_URL="${POST_TRAIN_URL:-}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-}"

if [[ -z "${BASELINE_URL}" || -z "${POST_TRAIN_URL}" || -z "${BASE_MODEL_DIR}" ]]; then
  echo "Set BASELINE_URL, POST_TRAIN_URL, and BASE_MODEL_DIR before running." >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  ACTIVE_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [[ -n "${ACTIVE_PIDS}" ]]; then
    echo "GPU compute jobs are active; refusing to preempt them: ${ACTIVE_PIDS}" >&2
    exit 3
  fi
fi

mkdir -p "${ARTIFACT_ROOT}"
cd "${REPO_ROOT}"

python scripts/level1/dataset/prepare_level1_dataset.py --write-hf
python train_areal.py --config configs/level1/archive/level1_image_overfit_2epoch.yaml --dry-run
cp configs/level1/archive/level1_image_overfit_2epoch.yaml "${ARTIFACT_ROOT}/config.resolved.yaml"
python scripts/level1/dataset/write_level1_manifest.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --model-revision "${BASE_MODEL}" \
  --dataset-manifest "artifacts/datasets/level1_dataset/manifest.json"

python scripts/level1/evaluate/evaluate_level1.py \
  --model "${BASE_MODEL}" \
  --base-url "${BASELINE_URL}" \
  --output "${ARTIFACT_ROOT}/baseline/summary.json"

PACMAN_TRAJECTORY_DIR="${ARTIFACT_ROOT}/training/trajectories" \
python train_areal.py --config configs/level1/archive/level1_image_overfit_2epoch.yaml \
  "cluster.fileroot=${ARTIFACT_ROOT}/training" \
  "actor.path=${BASE_MODEL}"

TRAINED_CHECKPOINT="${TRAINED_CHECKPOINT:-}"
if [[ -z "${TRAINED_CHECKPOINT}" ]]; then
  TRAINED_CHECKPOINT="$(find "${ARTIFACT_ROOT}/training" -type f -name model.safetensors -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2-)"
fi
if [[ -z "${TRAINED_CHECKPOINT}" || ! -f "${TRAINED_CHECKPOINT}/model.safetensors" ]]; then
  echo "Could not locate the final actor checkpoint." >&2
  exit 4
fi

python scripts/level1/report/build_complete_vlm_checkpoint.py \
  --trained-dir "${TRAINED_CHECKPOINT}" \
  --base-dir "${BASE_MODEL_DIR}" \
  --output-dir "${ARTIFACT_ROOT}/complete_checkpoint"

python -c 'from transformers import AutoModelForImageTextToText; import sys; AutoModelForImageTextToText.from_pretrained(sys.argv[1], local_files_only=True)' \
  "${ARTIFACT_ROOT}/complete_checkpoint"

python scripts/level1/evaluate/evaluate_level1.py \
  --model "${ARTIFACT_ROOT}/complete_checkpoint" \
  --base-url "${POST_TRAIN_URL}" \
  --output "${ARTIFACT_ROOT}/post_training/summary.json"

python scripts/level1/report/audit_level1_rewards.py \
  "${ARTIFACT_ROOT}/baseline/trajectories" \
  "${ARTIFACT_ROOT}/training/trajectories" \
  "${ARTIFACT_ROOT}/post_training/trajectories"

python scripts/level1/evaluate/compare_level1_runs.py \
  --baseline "${ARTIFACT_ROOT}/baseline/summary.json" \
  --post-training "${ARTIFACT_ROOT}/post_training/summary.json" \
  --output "${ARTIFACT_ROOT}/comparison.json"
