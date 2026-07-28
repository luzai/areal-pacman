#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OWNER_ROOT="${OWNER_ROOT:-/home/ubuntu/z00819216}"
ENV_ROOT="${ENV_ROOT:-${OWNER_ROOT}/miniconda/envs/maapacman-rl}"
PYTHON="${PYTHON:-${ENV_ROOT}/bin/python}"
PACMAN_PYTHON_ROOT="${PACMAN_PYTHON_ROOT:-${OWNER_ROOT}/maapacman-stack/pacman-python}"
BASE_MODEL="${BASE_MODEL:-${OWNER_ROOT}/models/Qwen3.5-9B}"
SOURCE_RUN="${SOURCE_RUN:-${OWNER_ROOT}/run_artifacts/maapacman-rl/level1-group12-20260723b}"
FINAL_MODEL="${FINAL_MODEL:-${SOURCE_RUN}/complete_checkpoint_final}"
EVAL_ROOT="${EVAL_ROOT:-${SOURCE_RUN}/prompt_ab_20260723b}"
GPU_IDS="${GPU_IDS:-0,1}"
PORT_BASE="${PORT_BASE:-18200}"

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
if [[ ! -f "${BASE_MODEL}/config.json" || ! -f "${FINAL_MODEL}/config.json" ]]; then
  echo "Base or final model checkpoint is incomplete." >&2
  exit 2
fi
if [[ -e "${EVAL_ROOT}/comparison.json" ]]; then
  echo "Prompt A/B is already complete: ${EVAL_ROOT}/comparison.json" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne 2 || "${GPUS[0]}" == "${GPUS[1]}" ]]; then
  echo "GPU_IDS must contain two unique GPU IDs: ${GPU_IDS}" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "GPU_IDS must be numeric: ${GPU_IDS}" >&2
    exit 2
  fi
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

mkdir -p "${EVAL_ROOT}/servers"
MODELS=(base final)
MODEL_PATHS=("${BASE_MODEL}" "${FINAL_MODEL}")
SERVER_PIDS=()
EVAL_PIDS=()
cleanup() {
  local any_running pid
  for pid in "${EVAL_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
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

for index in "${!MODELS[@]}"; do
  label="${MODELS[${index}]}"
  port=$((PORT_BASE + index))
  CUDA_VISIBLE_DEVICES="${GPUS[${index}]}" nohup setsid "${PYTHON}" \
    -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port "${port}" \
    --model "${MODEL_PATHS[${index}]}" \
    --served-model-name "${label}" \
    --dtype bfloat16 \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    >"${EVAL_ROOT}/servers/${label}.log" 2>&1 &
  pid=$!
  SERVER_PIDS+=("${pid}")
  printf '%s\n' "${pid}" >"${EVAL_ROOT}/servers/${label}.pid"
done

for index in "${!MODELS[@]}"; do
  label="${MODELS[${index}]}"
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

for index in "${!MODELS[@]}"; do
  model="${MODELS[${index}]}"
  port=$((PORT_BASE + index))
  for prompt in minimal_v1 live_static_v2; do
    output_dir="${EVAL_ROOT}/${model}/${prompt}"
    mkdir -p "${output_dir}"
    "${PYTHON}" "${REPO_ROOT}/scripts/level1/evaluate/evaluate_level1.py" \
      --model "${model}" \
      --base-url "http://127.0.0.1:${port}/v1" \
      --episodes 1 \
      --concurrency 1 \
      --temperature 0.0 \
      --top-p 1.0 \
      --prompt-style "${prompt}" \
      --pacman-python-root "${PACMAN_PYTHON_ROOT}" \
      --output "${output_dir}/greedy.json" \
      >"${output_dir}/greedy.log" 2>&1 &
    EVAL_PIDS+=("$!")
  done
done
for pid in "${EVAL_PIDS[@]}"; do
  wait "${pid}"
done

for prompt in minimal_v1 live_static_v2; do
  EVAL_PIDS=()
  for index in "${!MODELS[@]}"; do
    model="${MODELS[${index}]}"
    port=$((PORT_BASE + index))
    output_dir="${EVAL_ROOT}/${model}/${prompt}"
    "${PYTHON}" "${REPO_ROOT}/scripts/level1/evaluate/evaluate_level1.py" \
      --model "${model}" \
      --base-url "http://127.0.0.1:${port}/v1" \
      --episodes 12 \
      --concurrency 4 \
      --temperature 1.0 \
      --top-p 0.95 \
      --prompt-style "${prompt}" \
      --pacman-python-root "${PACMAN_PYTHON_ROOT}" \
      --output "${output_dir}/sampled12.json" \
      >"${output_dir}/sampled12.log" 2>&1 &
    EVAL_PIDS+=("$!")
  done
  for pid in "${EVAL_PIDS[@]}"; do
    wait "${pid}"
  done
done

"${PYTHON}" "${REPO_ROOT}/scripts/level1/evaluate/compare_level1_prompt_ab.py" \
  --eval-root "${EVAL_ROOT}" \
  --output "${EVAL_ROOT}/comparison.json"

echo "prompt_ab=ok"
echo "comparison=${EVAL_ROOT}/comparison.json"
