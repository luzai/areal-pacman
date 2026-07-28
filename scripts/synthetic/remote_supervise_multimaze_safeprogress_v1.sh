#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/z00819216/vla-research/areal-pacman
PYTHON=/home/z00819216/miniconda/envs/areal-vla/bin/python
TRAIN_CONFIG=configs/archive/vision/config_vision_rollout_qwen3p5_9b_imageonly_multimaze_v1_safeprogress_grpo_3epoch_4sample_8gpu.yaml
EVAL_CONFIG=configs/archive/vision/config_vision_rollout_qwen3p5_9b_imageonly_multimaze_v1_notrain_4sample_8gpu.yaml
ARTIFACT_ROOT="$ROOT/run_artifacts/multi_maze_v1_safeprogress_alpha4_20260715"
TRAIN_FILEROOT="$ROOT/run_artifacts/areal_vision_imageonly_multimaize_v1_safeprogress_alpha4_grpo"
BASELINE_ROOT="$ROOT/run_artifacts/multi_maze_v1_20260715/baseline"
BASE_MODEL=/mnt/data/z00819216/hf-cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a

source /home/z00819216/miniconda/etc/profile.d/conda.sh
conda activate /home/z00819216/miniconda/envs/areal-vla
export PATH=/home/z00819216/miniconda/envs/areal-vla/bin:$PATH
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export AREAL_FORCE_HOST_IP=10.0.12.183
cd "$ROOT"

mkdir -p "$ARTIFACT_ROOT"/{baseline,training,epoch_validation,post_rl,complete_checkpoints}

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

for split in train validation test; do
    cp "$BASELINE_ROOT/summary_${split}.json" "$ARTIFACT_ROOT/baseline/summary_${split}.json"
done

TRAIN_TRIAL=vision-qwen3p5-vlm-imageonly-multimaze-v1-safeprogress-alpha4-grpo3epoch4sample-20260715
run_areal "$TRAIN_CONFIG" "$TRAIN_TRIAL" "$ARTIFACT_ROOT/training" "$TRAIN_FILEROOT"
grep -q 'Train step 24/24 done' "$ARTIFACT_ROOT/training/run.log"
grep -q 'Training completes' "$ARTIFACT_ROOT/training/run.log"

CHECKPOINT_ROOT="$TRAIN_FILEROOT/checkpoints/z00819216/pacman-agentic-rl/$TRAIN_TRIAL/default"
mapfile -t TRAINED_CHECKPOINTS < <(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'epoch*' | sort -V)
test "${#TRAINED_CHECKPOINTS[@]}" -eq 3

for index in "${!TRAINED_CHECKPOINTS[@]}"; do
    complete="$ARTIFACT_ROOT/complete_checkpoints/epoch${index}"
    "$PYTHON" scripts/level1/report/build_complete_vlm_checkpoint.py \
        --trained-dir "${TRAINED_CHECKPOINTS[$index]}" \
        --base-dir "$BASE_MODEL" \
        --output-dir "$complete" | tee "$ARTIFACT_ROOT/complete_checkpoints/epoch${index}.log"
    output="$ARTIFACT_ROOT/epoch_validation/epoch${index}"
    run_areal "$EVAL_CONFIG" \
        "vision-qwen3p5-vlm-imageonly-multimaze-v1-safeprogress-alpha4-epoch${index}-validation-20260715" \
        "$output" \
        "$ROOT/run_artifacts/areal_vision_imageonly_multimaze_v1_safeprogress_alpha4_epoch${index}_validation" \
        "actor.path=$complete" \
        "ref.path=$complete" \
        "train_dataset.path=$ROOT/run_artifacts/multi_maze_v1/validation_hf_dataset"
    summarize_split "$output/trajectories" validation "$ARTIFACT_ROOT/epoch_validation/summary_epoch${index}.json"
done

SELECTED_INDEX=$(
    "$PYTHON" - "$ARTIFACT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "epoch_validation").glob("summary_epoch*.json")):
    index = int(path.stem.removeprefix("summary_epoch"))
    summary = json.loads(path.read_text())
    rows.append((float(summary["macro_layout_pass_rate"]), -index, index, summary))
selected = max(rows)
payload = {
    "selection_metric": "validation macro_layout_pass_rate",
    "selected_epoch_index": selected[2],
    "epochs": {
        str(index): {
            "macro_layout_pass_rate": score,
            "raw_pass_rate": float(summary["pass_rate"]),
        }
        for score, _, index, summary in rows
    },
}
(root / "checkpoint_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
print(selected[2])
PY
)
SELECTED_CKPT="$ARTIFACT_ROOT/complete_checkpoints/epoch${SELECTED_INDEX}"
cp "$ARTIFACT_ROOT/epoch_validation/summary_epoch${SELECTED_INDEX}.json" "$ARTIFACT_ROOT/post_rl/summary_validation.json"

POST_TRAIN="$ARTIFACT_ROOT/post_rl/train"
run_areal "$EVAL_CONFIG" \
    "vision-qwen3p5-vlm-imageonly-multimaze-v1-safeprogress-alpha4-selected-train-20260715" \
    "$POST_TRAIN" \
    "$ROOT/run_artifacts/areal_vision_imageonly_multimaze_v1_safeprogress_alpha4_selected_train" \
    "actor.path=$SELECTED_CKPT" \
    "ref.path=$SELECTED_CKPT"
summarize_split "$POST_TRAIN/trajectories" train "$ARTIFACT_ROOT/post_rl/summary_train.json"

POST_TEST="$ARTIFACT_ROOT/post_rl/test"
run_areal "$EVAL_CONFIG" \
    "vision-qwen3p5-vlm-imageonly-multimaze-v1-safeprogress-alpha4-selected-test-20260715" \
    "$POST_TEST" \
    "$ROOT/run_artifacts/areal_vision_imageonly_multimaze_v1_safeprogress_alpha4_selected_test" \
    "actor.path=$SELECTED_CKPT" \
    "ref.path=$SELECTED_CKPT" \
    "train_dataset.path=$ROOT/run_artifacts/multi_maze_v1/test_hf_dataset"
summarize_split "$POST_TEST/trajectories" test "$ARTIFACT_ROOT/post_rl/summary_test.json"

"$PYTHON" scripts/synthetic/build_multi_maze_comparison.py \
    --artifact-root "$ARTIFACT_ROOT" \
    --checkpoint "$SELECTED_CKPT" \
    --output "$ARTIFACT_ROOT/comparison.json" | tee "$ARTIFACT_ROOT/comparison.txt"

require_gpu_empty
echo "MULTI_MAZE_V1_SAFE_PROGRESS_ALPHA4_COMPLETE artifact_root=$ARTIFACT_ROOT selected_epoch=$SELECTED_INDEX checkpoint=$SELECTED_CKPT"
