#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/z00819216/vla-research/areal-pacman
PYTHON=/home/z00819216/miniconda/envs/areal-vla/bin/python
TRAIN_CONFIG=configs/archive/vision/config_vision_rollout_qwen3p5_9b_imageonly_multimaze_v1_grpo_3epoch_4sample_8gpu.yaml
EVAL_CONFIG=configs/archive/vision/config_vision_rollout_qwen3p5_9b_imageonly_multimaze_v1_notrain_4sample_8gpu.yaml
ARTIFACT_ROOT="$ROOT/run_artifacts/multi_maze_v1_20260715"
TRAIN_FILEROOT="$ROOT/run_artifacts/areal_vision_imageonly_multimaize_v1_grpo"
BASE_MODEL=/mnt/data/z00819216/hf-cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a

source /home/z00819216/miniconda/etc/profile.d/conda.sh
conda activate /home/z00819216/miniconda/envs/areal-vla
export PATH=/home/z00819216/miniconda/envs/areal-vla/bin:$PATH
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export AREAL_FORCE_HOST_IP=10.0.12.183
cd "$ROOT"

mkdir -p "$ARTIFACT_ROOT"/{baseline,training,post_rl}

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
}

require_gpu_empty() {
    local pids
    pids=$(gpu_pids)
    if [[ -n "$pids" ]]; then
        echo "ABORT: GPU compute processes are active: $pids" >&2
        exit 70
    fi
}

run_areal() {
    local config=$1
    local trial=$2
    local output_dir=$3
    local fileroot=$4
    shift 4
    require_gpu_empty
    mkdir -p "$output_dir/trajectories"
    export PACMAN_TRAJECTORY_DIR="$output_dir/trajectories"
    export PACMAN_WORKFLOW_TRACE="$output_dir/workflow_trace.jsonl"
    set +e
    timeout 86400 "$PYTHON" train_areal.py --config "$config" \
        "trial_name=$trial" \
        "cluster.fileroot=$fileroot" \
        "cluster.name_resolve.nfs_record_root=$ROOT/run_artifacts/name_resolve_$trial" \
        "$@" 2>&1 | tee "$output_dir/run.log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "EXIT=$rc" >> "$output_dir/run.log"
    if [[ $rc -ne 0 ]]; then
        echo "AReaL trial failed: $trial rc=$rc" >&2
        exit "$rc"
    fi
}

summarize_split() {
    local trajectory_dir=$1
    local split=$2
    local output=$3
    "$PYTHON" scripts/synthetic/summarize_multi_maze_rollouts.py \
        --trajectory-dir "$trajectory_dir" \
        --include-split "$split" \
        --require-complete \
        --output "$output"
}

latest_checkpoint() {
    find "$1" -mindepth 1 -maxdepth 1 -type d -name 'epoch*' | sort -V | tail -1
}

BASE_TRAINVAL="$ARTIFACT_ROOT/baseline/train_validation"
run_areal "$EVAL_CONFIG" \
    vision-qwen3p5-vlm-imageonly-multimaze-v1-base-trainval-20260715 \
    "$BASE_TRAINVAL" \
    "$ROOT/run_artifacts/areal_vision_imageonly_multimaze_v1_base_trainval"
summarize_split "$BASE_TRAINVAL/trajectories" train "$ARTIFACT_ROOT/baseline/summary_train.json"
summarize_split "$BASE_TRAINVAL/trajectories" validation "$ARTIFACT_ROOT/baseline/summary_validation.json"

BASE_TEST="$ARTIFACT_ROOT/baseline/test"
run_areal "$EVAL_CONFIG" \
    vision-qwen3p5-vlm-imageonly-multimaze-v1-base-test-20260715 \
    "$BASE_TEST" \
    "$ROOT/run_artifacts/areal_vision_imageonly_multimaze_v1_base_test" \
    "train_dataset.path=$ROOT/run_artifacts/multi_maze_v1/test_hf_dataset"
summarize_split "$BASE_TEST/trajectories" test "$ARTIFACT_ROOT/baseline/summary_test.json"

TRAIN_TRIAL=vision-qwen3p5-vlm-imageonly-multimaze-v1-grpo3epoch4sample-20260715
TRAIN_OUTPUT="$ARTIFACT_ROOT/training"
run_areal "$TRAIN_CONFIG" "$TRAIN_TRIAL" "$TRAIN_OUTPUT" "$TRAIN_FILEROOT"
grep -q 'Train step 24/24 done' "$TRAIN_OUTPUT/run.log"
grep -q 'Training completes' "$TRAIN_OUTPUT/run.log"

CHECKPOINT_ROOT="$TRAIN_FILEROOT/checkpoints/z00819216/pacman-agentic-rl/$TRAIN_TRIAL/default"
FINAL_CKPT=$(latest_checkpoint "$CHECKPOINT_ROOT")
test -f "$FINAL_CKPT/model.safetensors"
COMPLETE_CKPT="$ARTIFACT_ROOT/complete_checkpoint"
"$PYTHON" scripts/level1/report/build_complete_vlm_checkpoint.py \
    --trained-dir "$FINAL_CKPT" \
    --base-dir "$BASE_MODEL" \
    --output-dir "$COMPLETE_CKPT" | tee "$ARTIFACT_ROOT/complete_checkpoint_build.log"
test -f "$COMPLETE_CKPT/model.safetensors.index.json"
test -f "$COMPLETE_CKPT/visual-model.safetensors"

POST_TRAINVAL="$ARTIFACT_ROOT/post_rl/train_validation"
run_areal "$EVAL_CONFIG" \
    vision-qwen3p5-vlm-imageonly-multimaze-v1-postrl-trainval-20260715 \
    "$POST_TRAINVAL" \
    "$ROOT/run_artifacts/areal_vision_imageonly_multimaze_v1_postrl_trainval" \
    "actor.path=$COMPLETE_CKPT" \
    "ref.path=$COMPLETE_CKPT"
summarize_split "$POST_TRAINVAL/trajectories" train "$ARTIFACT_ROOT/post_rl/summary_train.json"
summarize_split "$POST_TRAINVAL/trajectories" validation "$ARTIFACT_ROOT/post_rl/summary_validation.json"

POST_TEST="$ARTIFACT_ROOT/post_rl/test"
run_areal "$EVAL_CONFIG" \
    vision-qwen3p5-vlm-imageonly-multimaze-v1-postrl-test-20260715 \
    "$POST_TEST" \
    "$ROOT/run_artifacts/areal_vision_imageonly_multimaze_v1_postrl_test" \
    "actor.path=$COMPLETE_CKPT" \
    "ref.path=$COMPLETE_CKPT" \
    "train_dataset.path=$ROOT/run_artifacts/multi_maze_v1/test_hf_dataset"
summarize_split "$POST_TEST/trajectories" test "$ARTIFACT_ROOT/post_rl/summary_test.json"

"$PYTHON" scripts/synthetic/build_multi_maze_comparison.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --checkpoint "$COMPLETE_CKPT" \
    --output "$ARTIFACT_ROOT/comparison.json" | tee "$ARTIFACT_ROOT/comparison.txt"

require_gpu_empty
echo "MULTI_MAZE_V1_COMPLETE artifact_root=$ARTIFACT_ROOT checkpoint=$COMPLETE_CKPT"
