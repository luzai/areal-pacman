#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER_ROOT="${OWNER_ROOT:-/home/ubuntu/z00819216}"
ENV_ROOT="${ENV_ROOT:-${OWNER_ROOT}/miniconda/envs/maapacman-rl}"
PYTHON="${PYTHON:-${ENV_ROOT}/bin/python}"
PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT:-${OWNER_ROOT}/maapacman-stack/pacman-python}"
BASE_MODEL="${BASE_MODEL:-${OWNER_ROOT}/models/Qwen3.5-9B}"
SOURCE_RUN="${SOURCE_RUN:?SOURCE_RUN is required}"
TRIAL_NAME="${TRIAL_NAME:?TRIAL_NAME is required}"
EVAL_ROOT="${EVAL_ROOT:-${SOURCE_RUN}/sampled_test}"
GPU_ID="${GPU_ID:-0}"
PORT_BASE="${PORT_BASE:-18150}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SOURCE_RUN}/training/checkpoints/z00819216/maapacman-level1/${TRIAL_NAME}/default}"

if [[ "$(id -un)" != "z00819216" ]]; then
  echo "Run this evaluator as source owner z00819216." >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python is not executable: ${PYTHON}" >&2
  exit 2
fi
if [[ ! -d "${PACMAN_PYTHON_ROOT}/.git" ]]; then
  echo "Pinned pacman-python checkout is missing: ${PACMAN_PYTHON_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
  echo "Checkpoint root is missing: ${CHECKPOINT_ROOT}" >&2
  exit 2
fi
if [[ -e "${EVAL_ROOT}/comparison.json" ]]; then
  echo "Evaluation is already complete: ${EVAL_ROOT}/comparison.json" >&2
  exit 2
fi
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be numeric: ${GPU_ID}" >&2
  exit 2
fi

active="$(
  nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true
)"
if [[ -n "${active}" ]]; then
  echo "GPU ${GPU_ID} is busy; refusing to preempt PIDs: ${active}" >&2
  exit 3
fi

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MAAPACMAN_PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TORCH_COMPILE_DISABLE=1

mapfile -t TRAINED_DIRS < <(
  find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'epoch*' |
    sort -V
)
if [[ "${#TRAINED_DIRS[@]}" -ne 4 ]]; then
  echo "Expected four update checkpoints, found ${#TRAINED_DIRS[@]}." >&2
  exit 2
fi

mkdir -p "${EVAL_ROOT}/complete_checkpoints" "${EVAL_ROOT}/servers"
LABELS=(base)
MODEL_PATHS=("${BASE_MODEL}")
for index in "${!TRAINED_DIRS[@]}"; do
  label="$(printf 'update%02d' "$((index + 1))")"
  output="${EVAL_ROOT}/complete_checkpoints/${label}"
  if [[ ! -f "${output}/merge_manifest.json" ]]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/build_complete_vlm_checkpoint.py" \
      --trained-dir "${TRAINED_DIRS[${index}]}" \
      --base-dir "${BASE_MODEL}" \
      --output-dir "${output}"
  fi
  LABELS+=("${label}")
  MODEL_PATHS+=("${output}")
done

SERVER_PID=""
stop_server() {
  local count
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  for _ in $(seq 1 20); do
    count="$(ps -eo sid= | awk -v sid="${SERVER_PID}" '$1 == sid {n++} END {print n + 0}')"
    if [[ "${count}" -eq 0 ]]; then
      break
    fi
    sleep 1
  done
  count="$(ps -eo sid= | awk -v sid="${SERVER_PID}" '$1 == sid {n++} END {print n + 0}')"
  if [[ "${count}" -gt 0 ]]; then
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

start_server() {
  local label="$1"
  local model="$2"
  local port="$3"
  local log="${EVAL_ROOT}/servers/${label}.log"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" setsid "${PYTHON}" \
    -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${model}" \
    --served-model-name "${label}" \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    >"${log}" 2>&1 &
  SERVER_PID=$!
  printf '%s\n' "${SERVER_PID}" >"${EVAL_ROOT}/servers/${label}.pid"

  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -80 "${log}" >&2
      echo "vLLM server exited before readiness: ${label}" >&2
      exit 4
    fi
    sleep 1
  done
  echo "Timed out waiting for vLLM server: ${label}" >&2
  exit 4
}

for index in "${!LABELS[@]}"; do
  label="${LABELS[${index}]}"
  if [[ -f "${EVAL_ROOT}/${label}/sampled12.json" ]]; then
    echo "sampled12_already_complete=${label}"
    continue
  fi
  port=$((PORT_BASE + index))
  mkdir -p "${EVAL_ROOT}/${label}"
  start_server "${label}" "${MODEL_PATHS[${index}]}" "${port}"
  "${PYTHON}" "${REPO_ROOT}/scripts/evaluate_level1.py" \
    --model "${label}" \
    --base-url "http://127.0.0.1:${port}/v1" \
    --episodes 12 \
    --concurrency 4 \
    --temperature 0.7 \
    --top-p 0.95 \
    --max-completion-tokens 3 \
    --prompt-style live_state_v3 \
    --pacman-python-root "${PACMAN_PYTHON_ROOT}" \
    --output "${EVAL_ROOT}/${label}/sampled12.json" \
    >"${EVAL_ROOT}/${label}/sampled12.log" 2>&1
  stop_server
done

"${PYTHON}" "${REPO_ROOT}/scripts/compare_level1_sampled_run.py" \
  --eval-root "${EVAL_ROOT}" \
  --output "${EVAL_ROOT}/comparison.json" \
  --episodes 12

BEST_LABEL="$(
  "${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' \
    "${EVAL_ROOT}/comparison.json"
)"
OVERFIT_PASS="$(
  "${PYTHON}" -c \
    'import json,sys; print(str(json.load(open(sys.argv[1]))["overfit_pass"]).lower())' \
    "${EVAL_ROOT}/comparison.json"
)"

echo "sampled_run_evaluation=ok"
echo "best_checkpoint=${BEST_LABEL}"
echo "overfit_pass=${OVERFIT_PASS}"
echo "comparison=${EVAL_ROOT}/comparison.json"
