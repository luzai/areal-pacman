#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OWNER_ROOT="${OWNER_ROOT:-/home/ubuntu/z00819216}"
ENV_ROOT="${ENV_ROOT:-${OWNER_ROOT}/miniconda/envs/maapacman-rl}"
PYTHON="${PYTHON:-${ENV_ROOT}/bin/python}"
PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT:-${OWNER_ROOT}/maapacman-stack/pacman-python}"
BASE_MODEL="${BASE_MODEL:-${OWNER_ROOT}/models/Qwen3.5-9B}"
SOURCE_RUN="${SOURCE_RUN:-${OWNER_ROOT}/run_artifacts/maapacman-rl/level1-group12-20260723b}"
EVAL_ROOT="${EVAL_ROOT:-${SOURCE_RUN}/corrected_eval_20260723}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
PORT_BASE="${PORT_BASE:-18100}"

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
if [[ -e "${EVAL_ROOT}/comparison.json" ]]; then
  echo "Evaluation is already complete: ${EVAL_ROOT}/comparison.json" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne 6 ]]; then
  echo "Exactly six GPU IDs are required for base, epoch0-3, and final." >&2
  exit 2
fi
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]] || [[ -n "${SEEN[${gpu}]:-}" ]]; then
    echo "GPU_IDS must contain six unique numeric IDs: ${GPU_IDS}" >&2
    exit 2
  fi
  SEEN["${gpu}"]=1
  active="$(
    nvidia-smi -i "${gpu}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true
  )"
  if [[ -n "${active}" ]]; then
    echo "GPU ${gpu} is busy; refusing to preempt PIDs: ${active}" >&2
    exit 3
  fi
done

export PATH="$(dirname "${PYTHON}"):${PATH}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MAAPACMAN_PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT}"
export PYGAME_HIDE_SUPPORT_PROMPT=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TORCH_COMPILE_DISABLE=1

CHECKPOINT_ROOT="${SOURCE_RUN}/training/checkpoints/z00819216/maapacman-level1/overfit-group12-20260723b/default"
mapfile -t TRAINED_DIRS < <(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'epoch*' | sort)
if [[ "${#TRAINED_DIRS[@]}" -ne 4 ]]; then
  echo "Expected four epoch checkpoints, found ${#TRAINED_DIRS[@]}." >&2
  exit 2
fi

mkdir -p "${EVAL_ROOT}/complete_checkpoints" "${EVAL_ROOT}/servers"
LABELS=(base epoch0 epoch1 epoch2 epoch3 final)
MODEL_PATHS=("${BASE_MODEL}")
for index in 0 1 2 3; do
  output="${EVAL_ROOT}/complete_checkpoints/epoch${index}"
  if [[ ! -f "${output}/merge_manifest.json" ]]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/build_complete_vlm_checkpoint.py" \
      --trained-dir "${TRAINED_DIRS[${index}]}" \
      --base-dir "${BASE_MODEL}" \
      --output-dir "${output}"
  fi
  MODEL_PATHS+=("${output}")
done
MODEL_PATHS+=("${SOURCE_RUN}/complete_checkpoint_final")

SERVER_PIDS=()
cleanup() {
  local any_running pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done
  for _ in $(seq 1 10); do
    any_running=0
    for pid in "${SERVER_PIDS[@]:-}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        any_running=1
      fi
    done
    if [[ "${any_running}" -eq 0 ]]; then
      break
    fi
    sleep 1
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

for index in "${!LABELS[@]}"; do
  label="${LABELS[${index}]}"
  model="${MODEL_PATHS[${index}]}"
  port=$((PORT_BASE + index))
  log="${EVAL_ROOT}/servers/${label}.log"
  pid_file="${EVAL_ROOT}/servers/${label}.pid"
  CUDA_VISIBLE_DEVICES="${GPUS[${index}]}" nohup setsid "${PYTHON}" \
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
  pid=$!
  SERVER_PIDS+=("${pid}")
  printf '%s\n' "${pid}" >"${pid_file}"
done

for index in "${!LABELS[@]}"; do
  label="${LABELS[${index}]}"
  port=$((PORT_BASE + index))
  pid="${SERVER_PIDS[${index}]}"
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -80 "${EVAL_ROOT}/servers/${label}.log" >&2
      echo "vLLM server exited before readiness: ${label}" >&2
      exit 4
    fi
    sleep 1
  done
  if [[ "${ready}" -ne 1 ]]; then
    echo "Timed out waiting for vLLM server: ${label}" >&2
    exit 4
  fi
done

EVAL_PIDS=()
for index in "${!LABELS[@]}"; do
  label="${LABELS[${index}]}"
  port=$((PORT_BASE + index))
  mkdir -p "${EVAL_ROOT}/${label}"
  "${PYTHON}" "${REPO_ROOT}/scripts/evaluate_level1.py" \
    --model "${label}" \
    --base-url "http://127.0.0.1:${port}/v1" \
    --episodes 1 \
    --concurrency 1 \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-completion-tokens 3 \
    --pacman-python-root "${PACMAN_PYTHON_ROOT}" \
    --output "${EVAL_ROOT}/${label}/greedy.json" \
    >"${EVAL_ROOT}/${label}/greedy.log" 2>&1 &
  EVAL_PIDS+=("$!")
done
for pid in "${EVAL_PIDS[@]}"; do
  wait "${pid}"
done

"${PYTHON}" "${REPO_ROOT}/scripts/evaluate_level1.py" \
  --model final \
  --base-url "http://127.0.0.1:$((PORT_BASE + 5))/v1" \
  --episodes 24 \
  --concurrency 4 \
  --temperature 1.0 \
  --top-p 0.95 \
  --max-completion-tokens 3 \
  --pacman-python-root "${PACMAN_PYTHON_ROOT}" \
  --output "${EVAL_ROOT}/final/sampled24.json" \
  >"${EVAL_ROOT}/final/sampled24.log" 2>&1

"${PYTHON}" "${REPO_ROOT}/scripts/compare_level1_checkpoints.py" \
  --eval-root "${EVAL_ROOT}" \
  --output "${EVAL_ROOT}/comparison.json"

echo "corrected_evaluation=ok"
echo "comparison=${EVAL_ROOT}/comparison.json"
