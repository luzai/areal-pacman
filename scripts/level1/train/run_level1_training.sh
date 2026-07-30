#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OWNER_ROOT="${OWNER_ROOT:-/home/ubuntu/z00819216}"
ENV_ROOT="${ENV_ROOT:-${OWNER_ROOT}/miniconda/envs/maapacman-rl}"
PYTHON="${PYTHON:-${ENV_ROOT}/bin/python}"
AREAL_ROOT="${AREAL_ROOT:-${OWNER_ROOT}/xinglu/AReaL}"
MODEL_PATH="${MODEL_PATH:-${OWNER_ROOT}/models/Qwen3.5-9B}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${OWNER_ROOT}/run_artifacts/maapacman-rl/level1-overfit-${RUN_ID}}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/level1/archive/level1_image_overfit_4epoch_group12_8gpu.yaml}"
DATASET_OUTPUT_ROOT="${DATASET_OUTPUT_ROOT:-${REPO_ROOT}/run_artifacts/level1_dataset}"
TRAIN_EPISODES="${TRAIN_EPISODES:-8}"
VALIDATION_EPISODES="${VALIDATION_EPISODES:-2}"
DATASET_MAX_STEPS="${DATASET_MAX_STEPS:-287}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python is not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model checkout is incomplete: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Training config does not exist: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${AREAL_ROOT}/areal/__init__.py" ]]; then
  echo "Official AReaL checkout is incomplete: ${AREAL_ROOT}" >&2
  exit 2
fi

# Keep the MaaPacman recipe independent from the historical robotics fork.
# The official AReaL worktree must win import resolution even when the Conda
# environment still contains an older editable AReaL installation.
export PYTHONPATH="${AREAL_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
AREAL_IMPORT_PATH="$(
  "${PYTHON}" -c 'import pathlib, areal; print(pathlib.Path(areal.__file__).resolve())'
)"
AREAL_ROOT_REAL="$(
  "${PYTHON}" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).resolve())' \
    "${AREAL_ROOT}"
)"
case "${AREAL_IMPORT_PATH}" in
  "${AREAL_ROOT_REAL}"/*) ;;
  *)
    echo "AReaL import escaped official checkout: ${AREAL_IMPORT_PATH}" >&2
    exit 2
    ;;
esac
echo "areal_import=${AREAL_IMPORT_PATH}"

# AReaL's proxy workers bind the node-private interface so sibling worker
# processes can reach them. Use one unpredictable, process-lifetime key for
# both controller generations and never print or persist its resolved value.
if [[ -z "${AREAL_ADMIN_API_KEY:-}" ]]; then
  AREAL_ADMIN_API_KEY="$(
    "${PYTHON}" -c 'import secrets; print(secrets.token_urlsafe(32))'
  )"
fi
if [[ "${AREAL_ADMIN_API_KEY}" == "areal-admin-key" ]]; then
  echo "AREAL_ADMIN_API_KEY must not use AReaL's public default." >&2
  exit 2
fi
export AREAL_ADMIN_API_KEY

CONFIG_GPU_COUNT="$(
  awk '/^[[:space:]]+n_gpus_per_node:/ { print $2; exit }' "${CONFIG}"
)"
if [[ ! "${CONFIG_GPU_COUNT}" =~ ^(6|8)$ ]]; then
  echo "Config must request an accepted 6- or 8-GPU topology: ${CONFIG_GPU_COUNT}" >&2
  exit 2
fi

CONFIG_OFFLOAD="$(
  awk '/^enable_offload:/ { print $2; exit }' "${CONFIG}"
)"
if [[ "${CONFIG_OFFLOAD}" == "true" ]]; then
  "${PYTHON}" -c '
from pathlib import Path
import torch_memory_saver
from areal.utils.offload import get_tms_env_vars

preload = Path(get_tms_env_vars()["LD_PRELOAD"])
if not preload.is_file():
    raise SystemExit(f"missing torch-memory-saver preload library: {preload}")
print(f"torch_memory_saver_preload={preload}")
'
fi
if grep -q '^[[:space:]]*offload_params:[[:space:]]*true' "${CONFIG}"; then
  "${PYTHON}" -c '
import inspect
from areal.engine.fsdp_engine import FSDPEngine

source = inspect.getsource(FSDPEngine.initialize)
marker = "CPUOffloadPolicy has already moved the persistent FSDP parameter"
if marker not in source:
    raise SystemExit(
        "AReaL is missing patches/areal_fsdp_cpu_offload_empty_cache.patch"
    )
print("areal_fsdp_cpu_offload_empty_cache_patch=ok")
'
fi
if grep -q '^open_action_mask:[[:space:]]*true' "${CONFIG}"; then
  "${PYTHON}" - "${CONFIG}" <<'PY'
import inspect
import sys
from pathlib import Path

from areal.api.cli_args import vLLMConfig
from areal.engine.fsdp_engine import FSDPEngine

config_text = Path(sys.argv[1]).read_text(encoding="utf-8")
source = inspect.getsource(FSDPEngine)
if "_apply_pacman_action_mask" not in source:
    raise SystemExit(
        "AReaL is missing patches/areal_pacman_action_logprobs.patch"
    )
if not hasattr(vLLMConfig, "logprobs_mode"):
    raise SystemExit(
        "AReaL vLLMConfig cannot select processed rollout log-probabilities"
    )
if "logprobs_mode: processed_logprobs" not in config_text:
    raise SystemExit("open-action-mask training requires processed_logprobs")
if "top_p: 0.95" in config_text:
    raise SystemExit("open-action-mask training requires top_p=1.0")
print("areal_pacman_action_logprobs_patch=ok")
PY
fi
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_IDS="$(seq -s, 0 $((CONFIG_GPU_COUNT - 1)))"
fi

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne "${CONFIG_GPU_COUNT}" ]]; then
  echo "Config requires ${CONFIG_GPU_COUNT} unique GPUs, got ${GPU_IDS}." >&2
  exit 2
fi

declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "GPU_IDS must contain unique numeric GPU IDs: ${GPU_IDS}" >&2
    exit 2
  fi
  SEEN_GPUS[${gpu}]=1
  active_pids="$(
    nvidia-smi -i "${gpu}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true
  )"
  if [[ -n "${active_pids}" ]]; then
    echo "GPU ${gpu} is busy; refusing to preempt PIDs: ${active_pids}" >&2
    exit 3
  fi
done

mkdir -p "${ARTIFACT_ROOT}" "${ARTIFACT_ROOT}/training/trajectories"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export HF_HOME="${HF_HOME:-${OWNER_ROOT}/hf-cache}"
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TORCH_COMPILE_DISABLE=1
# PyTorch 2.11 + CUDA 13 can select cuDNN SDPA for Qwen3.5 shapes that
# cuDNN cannot plan. sitecustomize.py applies this before every controller
# and RPC worker imports AReaL/torch, preserving SDPA with another backend.
export MAAPACMAN_DISABLE_CUDNN_SDPA="${MAAPACMAN_DISABLE_CUDNN_SDPA:-1}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export MAAPACMAN_PACMAN_PYTHON_ROOT="${MAAPACMAN_PACMAN_PYTHON_ROOT:-${OWNER_ROOT}/maapacman-stack/pacman-python}"
export PACMAN_TRAJECTORY_DIR="${ARTIFACT_ROOT}/training/trajectories"

if [[ "$(command -v python3)" != "$(dirname "${PYTHON}")/python3" ]]; then
  echo "python3 does not resolve inside the selected Conda environment: $(command -v python3)" >&2
  exit 2
fi

"${PYTHON}" scripts/level1/dataset/prepare_level1_dataset.py \
  --output-root "${DATASET_OUTPUT_ROOT}" \
  --train-episodes "${TRAIN_EPISODES}" \
  --validation-episodes "${VALIDATION_EPISODES}" \
  --max-steps "${DATASET_MAX_STEPS}" \
  --write-hf
"${PYTHON}" train_areal.py --config "${CONFIG}" --dry-run --validate-areal
cp "${CONFIG}" "${ARTIFACT_ROOT}/config.yaml"
"${PYTHON}" scripts/level1/dataset/write_level1_manifest.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --model-revision "${MODEL_PATH}" \
  --dataset-manifest "${DATASET_OUTPUT_ROOT}/manifest.json"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight=ok"
  echo "areal_root=${AREAL_ROOT_REAL}"
  echo "gpu_ids=${GPU_IDS}"
  echo "artifacts=${ARTIFACT_ROOT}"
  exit 0
fi

echo "Starting AReaL level-1 training"
echo "  run_id=${RUN_ID}"
echo "  areal=${AREAL_ROOT_REAL}"
echo "  gpu_ids=${GPU_IDS}"
echo "  model=${MODEL_PATH}"
echo "  artifacts=${ARTIFACT_ROOT}"

exec "${PYTHON}" train_areal.py \
  --config "${CONFIG}" \
  "artifact_root=${ARTIFACT_ROOT}" \
  "cluster.fileroot=${ARTIFACT_ROOT}/training" \
  "cluster.name_resolve.nfs_record_root=${ARTIFACT_ROOT}/name_resolve" \
  "actor.path=${MODEL_PATH}" \
  "experiment_name=maapacman-level1" \
  "trial_name=overfit-${RUN_ID}"
