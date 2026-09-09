#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/repro/run_c1_iter25_lineage.sh [--preflight-only|--smoke]

Required environment:
  C1_REPRO_PYTHON
  C1_REPRO_RUN_ROOT
  C1_REPRO_DATA_ROOT
  C1_REPRO_BASE_MODEL
  C1_REPRO_GAME_ROOT
  C1_REPRO_STAGE{1,2,3}_RECIPE_ROOT
  C1_REPRO_STAGE{1,2,3}_AREAL_ROOT
  C1_REPRO_STAGE{1,2,3}_MAAPACMAN_ROOT
  AREAL_ADMIN_API_KEY

The default run performs 16 + 1 + 8 updates. --smoke performs one update per
stage while retaining the 256/700/512 horizons and exercising both handoffs.
EOF
}

mode=full
case "${1:-}" in
  "") ;;
  --preflight-only) mode=preflight ;;
  --smoke) mode=smoke ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

required_vars=(
  C1_REPRO_PYTHON C1_REPRO_RUN_ROOT C1_REPRO_DATA_ROOT
  C1_REPRO_BASE_MODEL C1_REPRO_GAME_ROOT AREAL_ADMIN_API_KEY
  C1_REPRO_STAGE1_RECIPE_ROOT C1_REPRO_STAGE2_RECIPE_ROOT
  C1_REPRO_STAGE3_RECIPE_ROOT C1_REPRO_STAGE1_AREAL_ROOT
  C1_REPRO_STAGE2_AREAL_ROOT C1_REPRO_STAGE3_AREAL_ROOT
  C1_REPRO_STAGE1_MAAPACMAN_ROOT C1_REPRO_STAGE2_MAAPACMAN_ROOT
  C1_REPRO_STAGE3_MAAPACMAN_ROOT
)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

orchestrator_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
verify="$orchestrator_root/scripts/repro/verify_c1_checkpoint.py"
user_name=$(id -un)
experiment=pacman-agentic-rl
handoff_trial=c1-iter25-stage23-handoff
handoff_root="$C1_REPRO_RUN_ROOT/stage23-handoff"

require_file() {
  [[ -f "$1" ]] || { echo "required file not found: $1" >&2; exit 2; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "required directory not found: $1" >&2; exit 2; }
}
require_dir "$C1_REPRO_DATA_ROOT"
require_dir "$C1_REPRO_BASE_MODEL"
require_dir "$C1_REPRO_GAME_ROOT"
require_file "$C1_REPRO_PYTHON"
require_file "$verify"
for stage in 1 2 3; do
  recipe_var="C1_REPRO_STAGE${stage}_RECIPE_ROOT"
  areal_var="C1_REPRO_STAGE${stage}_AREAL_ROOT"
  maapacman_var="C1_REPRO_STAGE${stage}_MAAPACMAN_ROOT"
  require_file "${!recipe_var}/train_areal.py"
  require_file "${!areal_var}/areal/utils/recover.py"
  require_file "${!areal_var}/areal/engine/fsdp_utils/checkpoint.py"
  require_file "${!maapacman_var}/maapacman/env/__init__.py"
  grep -q 'with_optim = not self.config.no_save_optim' \
    "${!areal_var}/areal/utils/recover.py"
  grep -q 'with_optim = not self.config.no_load_optim' \
    "${!areal_var}/areal/utils/recover.py"
  grep -q 'state_dict = {"model": model_state_dict, "optim": optimizer_state_dict}' \
    "${!areal_var}/areal/engine/fsdp_utils/checkpoint.py"
done

for spec in stage1-256/train_hf stage2-700/train_hf stage3-512/train_hf; do
  require_dir "$C1_REPRO_DATA_ROOT/$spec"
done

run_parent=$(dirname "$C1_REPRO_RUN_ROOT")
require_dir "$run_parent"
available_kib=$(df -Pk "$run_parent" 2>/dev/null | awk 'NR==2 {print $4}')
min_free_gib=${C1_REPRO_MIN_FREE_GIB:-300}
if [[ -z "$available_kib" || "$available_kib" -lt $((min_free_gib * 1024 * 1024)) ]]; then
  echo "insufficient disk: require at least ${min_free_gib} GiB free" >&2
  exit 2
fi

echo "C1_LINEAGE_PREFLIGHT_OK mode=$mode run_root=$C1_REPRO_RUN_ROOT"
[[ "$mode" == preflight ]] && exit 0

if [[ "$mode" == smoke ]]; then
  stage1_stop=1
  stage2_start=1
  stage2_stop=2
  stage3_start=2
  stage3_stop=3
else
  stage1_stop=16
  stage2_start=16
  stage2_stop=17
  stage3_start=17
  stage3_stop=25
fi

mkdir -p "$C1_REPRO_RUN_ROOT/logs" "$handoff_root"

run_stage() {
  local stage=$1 recipe_root=$2 areal_root=$3 maapacman_root=$4
  local config=$5 resume_step=$6
  shift 6
  local log="$C1_REPRO_RUN_ROOT/logs/stage${stage}.log"
  export PYTHONPATH="$recipe_root:$maapacman_root:$areal_root${PYTHONPATH:+:$PYTHONPATH}"
  export MAAPACMAN_PACMAN_ROOT="$C1_REPRO_GAME_ROOT/pacman"
  export MAAPACMAN_EXPLICIT_RESUME_GLOBAL_STEP="$resume_step"
  (
    cd "$recipe_root"
    "$C1_REPRO_PYTHON" train_areal.py --config "$config" "$@"
  ) 2>&1 | tee "$log"
}

find_one_checkpoint() {
  local root=$1 global_step=$2
  mapfile -t matches < <(find "$root" -type d -name "*globalstep${global_step}" -print)
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "expected one globalstep${global_step} checkpoint under $root, found ${#matches[@]}" >&2
    printf '%s\n' "${matches[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

stage1_config="$orchestrator_root/configs/repro/c1_stage1_256.yaml"
stage2_config="$orchestrator_root/configs/repro/c1_stage2_700.yaml"
stage3_config="$orchestrator_root/configs/repro/c1_stage3_512.yaml"

run_stage 1 "$C1_REPRO_STAGE1_RECIPE_ROOT" "$C1_REPRO_STAGE1_AREAL_ROOT" \
  "$C1_REPRO_STAGE1_MAAPACMAN_ROOT" "$stage1_config" 0 \
  "total_train_steps=$stage1_stop" \
  "saver.freq_steps=$stage1_stop"
iter16=$(find_one_checkpoint "$C1_REPRO_RUN_ROOT/stage1-256" "$((stage1_stop - 1))")
"$C1_REPRO_PYTHON" "$verify" hf --path "$iter16" \
  --expected-global-step "$((stage1_stop - 1))"
export C1_REPRO_ITER16="$iter16"

recover_overrides=(
  "recover.fileroot=$handoff_root"
  "recover.experiment_name=$experiment"
  "recover.trial_name=$handoff_trial"
)
run_stage 2 "$C1_REPRO_STAGE2_RECIPE_ROOT" "$C1_REPRO_STAGE2_AREAL_ROOT" \
  "$C1_REPRO_STAGE2_MAAPACMAN_ROOT" "$stage2_config" "$stage2_start" \
  "total_train_steps=$stage2_stop" \
  "${recover_overrides[@]}"

recover_base="$handoff_root/checkpoints/$user_name/$experiment/$handoff_trial"
recover_dcp="$recover_base/default/recover_checkpoint"
recover_info="$recover_base/recover_info"
"$C1_REPRO_PYTHON" "$verify" dcp --path "$recover_dcp" \
  --recover-info "$recover_info" --expected-global-step "$((stage2_stop - 1))"

run_stage 3 "$C1_REPRO_STAGE3_RECIPE_ROOT" "$C1_REPRO_STAGE3_AREAL_ROOT" \
  "$C1_REPRO_STAGE3_MAAPACMAN_ROOT" "$stage3_config" "$stage3_start" \
  "total_train_steps=$stage3_stop" \
  "${recover_overrides[@]}"
grep -Fq 'new_branch_dcp_model_optimizer_plus_reconstructed_scheduler_rng' \
  "$C1_REPRO_RUN_ROOT/logs/stage3.log" || {
    echo "stage 3 did not report the DCP model+optimizer resume path" >&2
    exit 1
  }

final_checkpoint=$(find_one_checkpoint "$C1_REPRO_RUN_ROOT/stage3-512" "$((stage3_stop - 1))")
"$C1_REPRO_PYTHON" "$verify" hf --path "$final_checkpoint" \
  --expected-global-step "$((stage3_stop - 1))"
echo "C1_LINEAGE_COMPLETE final_checkpoint=$final_checkpoint"
