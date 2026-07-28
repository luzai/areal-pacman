#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/z00819216/vla-research/areal-pacman
BASE_TRIAL=vision-qwen3p5-vlm-imageonly-rules-activationgate-overfit6epoch-20260713
BASE_LOG="$ROOT/run_artifacts/$BASE_TRIAL.log"
BASE_PID=3325229
BASE_CKPT_ROOT="$ROOT/run_artifacts/areal_vision_imageonly_rules_activationgate_overfit6epoch/checkpoints/z00819216/pacman-agentic-rl/$BASE_TRIAL/default"
CONFIG=configs/archive/vision/config_vision_rollout_qwen3p5_9b_imageonly_rules_medium_overfit_grpo_6epoch_8tok_8gpu.yaml

source /home/z00819216/miniconda/etc/profile.d/conda.sh
conda activate /home/z00819216/miniconda/envs/areal-vla
export PATH=/home/z00819216/miniconda/envs/areal-vla/bin:$PATH
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export AREAL_FORCE_HOST_IP=10.0.12.183
cd "$ROOT"

latest_checkpoint() {
    find "$1" -mindepth 1 -maxdepth 1 -type d -name 'epoch*' | sort -V | tail -1
}

run_areal() {
    local trial=$1
    local log=$2
    shift 2
    export PACMAN_TRAJECTORY_DIR="run_artifacts/trajectories_$trial"
    set +e
    timeout 14400 python train_areal.py --config "$CONFIG" "trial_name=$trial" "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "EXIT=$rc" >> "$log"
    return "$rc"
}

while kill -0 "$BASE_PID" 2>/dev/null; do
    sleep 30
done

FINAL_CKPT=
if grep -q 'Train step 24/24 done' "$BASE_LOG" && grep -q 'Training completes' "$BASE_LOG"; then
    FINAL_CKPT=$(latest_checkpoint "$BASE_CKPT_ROOT")
    echo "base run complete: $FINAL_CKPT"
else
    echo "base run interrupted; waiting for its GPU workers to exit"
    for _ in $(seq 1 20); do
        if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
            break
        fi
        sleep 15
    done
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
        echo "ABORT: GPU processes remain after interrupted base run"
        exit 70
    fi

    CKPT=$(latest_checkpoint "$BASE_CKPT_ROOT")
    test -f "$CKPT/config.json"
    test -f "$CKPT/model.safetensors"
    GLOBAL_STEP=$(basename "$CKPT" | sed -E 's/.*globalstep([0-9]+).*/\1/')
    COMPLETED_STEPS=$((GLOBAL_STEP + 1))
    REMAINING_STEPS=$((24 - COMPLETED_STEPS))
    REMAINING_EPOCHS=$(((REMAINING_STEPS + 3) / 4))
    if (( REMAINING_EPOCHS < 1 )); then
        REMAINING_EPOCHS=1
    fi

    CONT_TRIAL="$BASE_TRIAL-detached-continuation"
    CONT_ROOT="$ROOT/run_artifacts/areal_vision_imageonly_rules_activationgate_detached_continuation"
    CONT_LOG="$ROOT/run_artifacts/$CONT_TRIAL.log"
    run_areal "$CONT_TRIAL" "$CONT_LOG" \
        "actor.path=$CKPT" \
        "ref.path=$CKPT" \
        "total_train_epochs=$REMAINING_EPOCHS" \
        "cluster.fileroot=$CONT_ROOT" \
        "cluster.name_resolve.nfs_record_root=$ROOT/run_artifacts/name_resolve_vision_imageonly_rules_activationgate_detached_continuation"
    CONT_CKPT_ROOT="$CONT_ROOT/checkpoints/z00819216/pacman-agentic-rl/$CONT_TRIAL/default"
    FINAL_CKPT=$(latest_checkpoint "$CONT_CKPT_ROOT")
    echo "continuation complete: $FINAL_CKPT"
fi

test -f "$FINAL_CKPT/config.json"
test -f "$FINAL_CKPT/model.safetensors"
if [[ -f "$ROOT/run_artifacts/activationgate_rl_stop_after_train" ]]; then
    echo "SUPERVISOR_TRAIN_ONLY_COMPLETE FINAL_CKPT=$FINAL_CKPT"
    exit 0
fi

EVAL_MODEL="$ROOT/run_artifacts/vision-qwen3p5-vlm-imageonly-rules-activationgate-overfit6epoch-complete-checkpoint-20260714"
test -f "$EVAL_MODEL/model.safetensors.index.json"
test -f "$EVAL_MODEL/visual-model.safetensors"
EVAL_TRIAL=vision-qwen3p5-vlm-imageonly-rules-activationgate-posttrain-notrain16-completeckpt-20260714
EVAL_ROOT="$ROOT/run_artifacts/areal_vision_imageonly_rules_activationgate_posttrain_notrain16_completeckpt_20260714"
EVAL_LOG="$ROOT/run_artifacts/$EVAL_TRIAL.log"
run_areal "$EVAL_TRIAL" "$EVAL_LOG" \
    "actor.path=$EVAL_MODEL" \
    "ref.path=$EVAL_MODEL" \
    "total_train_epochs=1" \
    "gconfig.n_samples=1" \
    "actor.optimizer.lr=0.0" \
    "actor.eps_clip=0.0" \
    "actor.kl_ctl=0.0" \
    "actor.recompute_logprob=false" \
    "actor.use_decoupled_loss=false" \
    "cluster.fileroot=$EVAL_ROOT" \
    "cluster.name_resolve.nfs_record_root=$ROOT/run_artifacts/name_resolve_vision_imageonly_rules_activationgate_posttrain_notrain16_completeckpt_20260714"

echo "SUPERVISOR_COMPLETE FINAL_CKPT=$FINAL_CKPT"
