# AReaL PacMan Scaffold

## Architecture design

- [AReaL Pacman RL Recipe Design](docs/architecture/AREAL_RECIPE_DESIGN.md)
- [Companion MaaPacman Environment Interface Design](../MaaPacman/PACMAN_ENV_DESIGN.md)

## Current Level-1 run

The latest completed production run is
`openmask16-step32-r12-20260728T160000Z`: 16/16 optimizer updates with
thinking disabled and a dynamic open-action mask. Its local presentation,
videos, audit helpers, representative trajectory, and retention policy are
described in [RUN_ARTIFACTS.md](RUN_ARTIFACTS.md). The complete remote
artifacts and all 16 checkpoints were written under:

```text
/home/ubuntu/z00819216/run_artifacts/maapacman-rl/openmask16-step32-r12-20260728T160000Z
```

## Quick start

This repository is the AReaL recipe layer. The real Level-1 game environment
comes from the separate `maapacman` Python package; install that package before
running the production workflow.

Local development from PowerShell:

```powershell
Set-Location C:\Users\<user>\path\to\areal-pacman
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dataset,agent]"
python -m pytest
```

If `maapacman` is not available from your package index, install the companion
MaaPacman checkout first:

```powershell
python -m pip install -e C:\path\to\MaaPacman
python -m pip install -e ".[dev,dataset,agent]"
```

Prepare the short-horizon Level-1 dataset:

```powershell
python scripts\level1\dataset\prepare_level1_dataset.py `
  --output-root run_artifacts\level1_dataset_step32 `
  --train-episodes 8 `
  --validation-episodes 2 `
  --max-steps 32
```

Production training is Linux/H100-oriented and expects an official AReaL
checkout, the MaaPacman environment, and a local model checkpoint:

```bash
export AREAL_ROOT=/home/ubuntu/z00819216/xinglu/AReaL
export MODEL_PATH=/home/ubuntu/z00819216/models/Qwen3.5-9B
export CONFIG="$PWD/configs/level1/train/level1_live_state_step32_16update_group12_8gpu.yaml"
export DATASET_MAX_STEPS=32
bash scripts/level1/train/run_level1_training.sh
```

The launcher verifies that `areal` resolves from `AREAL_ROOT`, prepares the
dataset, checks the configured GPU topology, and writes checkpoints,
trajectories, and logs beneath `ARTIFACT_ROOT`. Set `ARTIFACT_ROOT` explicitly
for durable or shared storage.

### Reproduce the 200-update Level-1 run on H100

Before launching, verify that all eight GPUs on the selected node are free.
The following detached command creates a unique UTC-stamped trial and artifact
directory, so it does not overwrite the original run:

```bash
export AREAL_PACMAN_ROOT=/mnt/data/z00819216/maapacman-stack/areal-pacman-r13-20260728T221500Z
export MODEL_PATH=/mnt/data/z00819216/models/Qwen3.5-9B

cd "${AREAL_PACMAN_ROOT}"

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
TRIAL_NAME="overfit-openmask200-step32-r14-repro-${RUN_TS}"
ARTIFACT_ROOT="/mnt/data/z00819216/run_artifacts/maapacman-rl/${TRIAL_NAME}"

mkdir -p "${ARTIFACT_ROOT}"

nohup /mnt/data/z00819216/conda_env/maapacman-rl/bin/python train_areal.py \
  --config "${AREAL_PACMAN_ROOT}/configs/level1/train/level1_live_state_step32_200update_group12_8gpu.yaml" \
  artifact_root="${ARTIFACT_ROOT}" \
  cluster.fileroot="${ARTIFACT_ROOT}/training" \
  cluster.name_resolve.nfs_record_root="${ARTIFACT_ROOT}/name_resolve" \
  actor.path="${MODEL_PATH}" \
  experiment_name=maapacman-level1 \
  trial_name="${TRIAL_NAME}" \
  > "${ARTIFACT_ROOT}/launcher.log" 2>&1 < /dev/null &

PID=$!
echo "PID=${PID}"
echo "TRIAL_NAME=${TRIAL_NAME}"
echo "ARTIFACT_ROOT=${ARTIFACT_ROOT}"
echo "LOG=${ARTIFACT_ROOT}/launcher.log"
```

## Environment and code layout

```text
MaaPacman repository / maapacman package
  maapacman.env.PygamePacmanEnv
    Owns the real pygame Level-1 state transition, RGB rendering,
    legal/open actions, rewards, termination, and gameplay metrics.

areal-pacman repository (this repository)
  areal_pacman/level1/workflow.py
    Connects image plus live game state to AReaL rollout generation.
    PacmanNativeVisionWorkflow applies no-thinking decoding and the dynamic
    open-action mask used by production training.
  areal_pacman/level1/level1_dataset.py
    Builds deterministic train/validation rows backed by PygamePacmanEnv.
  areal_pacman/level1/rewards.py
    Defines recipe-side reward composition and reporting.
  areal_pacman/synthetic/env.py
    Historical synthetic text/maze environment only; not production Level-1.
  configs/
    Production Level-1 train/eval configs plus archived experiment configs.
    See configs/README.md.
  scripts/
    Level-1 train/evaluate/dataset/report tools and historical synthetic tools.
    See scripts/README.md.
  scripts/level1/train/run_level1_training.sh
    Validated Linux/H100 training entry point.
  scripts/level1/evaluate/evaluate_level1.py
    Independent checkpoint evaluator.
  scripts/level1/report/summarize_level1_trajectories.py
    Audits parsing, action-mask compliance, collisions, and rollout metrics.
  tests/
    CPU contract tests for dataset, workflow, masking, rewards, and utilities.

Official AReaL checkout
  Supplies trainer, rollout workers, vLLM integration, FSDP actor/reference
  engines, scheduling, and checkpoint publication.
```

The boundaries are intentional: MaaPacman owns the game, this repository owns
the RL recipe, and official AReaL owns distributed training. Root modules such
as `areal_pacman.workflow` and `areal_pacman.env` are backward-compatible
import shims. New code should use `areal_pacman.level1.*` or
`areal_pacman.synthetic.*`. Do not run production Level-1 through the
historical `areal_pacman.synthetic.env.PacmanEnv`.

## Official AReaL checkout policy

On both `h100-node1` and `h100-node5`, framework and robotics development are
now deliberately separated:

```text
/home/ubuntu/z00819216/xinglu/AReaL
  branch: areal-main
  tracks: https://github.com/inclusionAI/AReaL.git main

/home/ubuntu/z00819216/xinglu/AReaL-VLA
  branch: robotics/morgan-vla
  keeps the existing Morgan/robotics work and dirty state
```

The `maapacman-rl` Conda environment and
`scripts/level1/train/run_level1_training.sh` resolve `areal` from the clean official
`AReaL` worktree. Do not run Pacman training from `AReaL-VLA`.

The 2026-07-23 official-main GPU compatibility probe reached real
`PygamePacmanEnv` RGB rollout and produced four trajectory streams, then
failed before the first optimizer update at reference log-prob computation:
the OpenAI proxy trajectory omitted `mm_token_type_ids` and
`multi_modal_input`, which official Qwen-VL FSDP requires. The optional vLLM
`awex_adapter`/Megatron warning was not the failure.

`areal_pacman.level1.workflow.PacmanNativeVisionWorkflow` now implements the
recipe-owned native AReaL `RolloutWorkflow` contract. Each action request is
processed once and returns aligned `input_ids`, `mm_token_type_ids`,
`pixel_values`, `image_grid_thw`, rollout log-probs, versions, masks, and the
environment reward. The official 3B smoke config selects this workflow; the
legacy OpenAI-proxy workflow remains available for live/evaluation clients.
CPU contract tests and the real cached Qwen2.5-VL processor pass. A real
reference log-prob, optimizer step, and checkpoint reload are still required
before official-main training is accepted.

The production level-1 recipe consumes MaaPacman's public
`maapacman.env.PygamePacmanEnv`. The separate
`areal_pacman.synthetic.env.PacmanEnv`
class remains only for historical synthetic/multi-maze research. It is not the
original pygame environment and must not be used by the production level-1
recipe. The design document records its future rename to `SyntheticMazeEnv`.

这是一个最小 PacMan-style text environment, 用来准备 AReaL agentic RL workflow.

## 文件

- [pyproject.toml](pyproject.toml): package metadata 和 pytest config.
- [areal_pacman/synthetic/env.py](areal_pacman/synthetic/env.py): deterministic text PacMan environment.
- [areal_pacman/synthetic/baselines.py](areal_pacman/synthetic/baselines.py): random 和 greedy baseline agents.
- [areal_pacman/synthetic/evaluate.py](areal_pacman/synthetic/evaluate.py): episode evaluator.
- [areal_pacman/synthetic/dataset.py](areal_pacman/synthetic/dataset.py): JSONL prompt dataset generator.
- [areal_pacman/synthetic/maze_suite.py](areal_pacman/synthetic/maze_suite.py): `multi_maze_v1` manifest loader, split/hash validation, and registry.
- [multi_maze_v1.json](areal_pacman/synthetic/maze_suites/multi_maze_v1.json): deterministic `64/16/32` train/validation/test maze manifest with oracle solutions.
- [areal_pacman/prepare_hf_dataset.py](areal_pacman/prepare_hf_dataset.py): JSONL 到 Hugging Face dataset 的 export.
- [areal_pacman/areal_workflow.py](areal_pacman/areal_workflow.py): AReaL-compatible workflow adapter skeleton.
- [train_areal.py](train_areal.py): AReaL trainer entry, 带 dry-run mode.
- [remote_supervise_multimaze_v1.sh](scripts/synthetic/remote_supervise_multimaze_v1.sh): GPU-empty-gated base evaluation, multi-maze RL, complete checkpoint, and held-out evaluation pipeline.
- [remote_supervise_multimaze_safeprogress_v1.sh](scripts/synthetic/remote_supervise_multimaze_safeprogress_v1.sh): alpha-4 safe-progress GRPO, per-epoch validation selection, and selected-checkpoint train/test evaluation.
- [configs/archive/text/config_pacman.yaml](configs/archive/text/config_pacman.yaml): minimal local-scheduler AReaL config.
- [reports/](../reports/): grouped experiment reports and status updates.
- [trajectories/](../trajectories/): grouped rollout and failure/success trajectory samples.
- [configs/archive/text/config_text_rollout.yaml](configs/archive/text/config_text_rollout.yaml): minimal pure text episode rollout smoke.
- [configs/archive/text/config_text_rollout_longer.yaml](configs/archive/text/config_text_rollout_longer.yaml): slightly larger pure text rollout, for short reward and trajectory checks.
- [configs/archive/text/config_text_rollout_1p5b.yaml](configs/archive/text/config_text_rollout_1p5b.yaml): same rollout as longer, but with `Qwen/Qwen2.5-1.5B-Instruct`.
- [configs/archive/text/config_text_rollout_1p5b_legal.yaml](configs/archive/text/config_text_rollout_1p5b_legal.yaml): 1.5B legal-action prompt and illegal-action penalty ablation.
- [configs/archive/text/config_text_rollout_1p5b_rules.yaml](configs/archive/text/config_text_rollout_1p5b_rules.yaml): 1.5B explicit game-rules prompt PPO baseline.
- [configs/archive/text/config_text_rollout_1p5b_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_grpo.yaml): 1.5B GRPO-style config with `n_samples=4` and group reward normalization.
- [configs/archive/text/config_text_rollout_1p5b_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_overfit_grpo.yaml): 1.5B 4-episode overfit gate with `prompt_style=ghost_legal`.
- [configs/archive/text/config_text_rollout_1p5b_tiny_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_tiny_overfit_grpo.yaml): 1.5B tiny sparse curriculum overfit gate, no teacher hint.
- [configs/archive/text/config_text_rollout_1p5b_teacher_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_teacher_overfit_grpo.yaml): 1.5B teacher-hint overfit warm-start gate.
- [configs/archive/text/config_text_rollout_1p5b_teacher_pure_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_teacher_pure_overfit_grpo.yaml): 1.5B pure hint-only overfit gate, no execution-time teacher override.
- [configs/archive/text/config_text_rollout_1p5b_long_debug_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_long_debug_grpo.yaml): larger debug config to use only after the overfit gate passes.
- [configs/archive/text/config_text_rollout_7b_overfit_grpo.yaml](configs/archive/text/config_text_rollout_7b_overfit_grpo.yaml): 7B default-map sparse overfit config, single-GPU attempt OOMed during optimizer state allocation.
- [configs/archive/text/config_text_rollout_7b_overfit_grpo_8gpu.yaml](configs/archive/text/config_text_rollout_7b_overfit_grpo_8gpu.yaml): 7B default-map sparse overfit config using 8 GPUs with `vllm:d4p1t1` rollout and `fsdp:d4p1t1` actor.
- [configs/archive/text/config_text_rollout_7b_medium_default_overfit_grpo_8gpu.yaml](configs/archive/text/config_text_rollout_7b_medium_default_overfit_grpo_8gpu.yaml): 7B medium-map sparse overfit config using `layout_name=medium_default` and `prompt_style=ghost_legal_strict`.
- [configs/archive/text/config_text_rollout_7b_medium_default_reason_debug_8gpu.yaml](configs/archive/text/config_text_rollout_7b_medium_default_reason_debug_8gpu.yaml): 7B medium-map debug config using `prompt_style=ghost_legal_reason` and `max_new_tokens=96` so trajectories can include short reasons plus `Action: <token>`.
- [configs/archive/text/config_text_rollout_qwen3p5_9b_json_medium_overfit_grpo_6epoch_8gpu.yaml](configs/archive/text/config_text_rollout_qwen3p5_9b_json_medium_overfit_grpo_6epoch_8gpu.yaml): Qwen3.5-9B strict no-BFS `ghost_legal_json` 6epoch overfit config using the current `2/12` no-training baseline setting.
- [configs/archive/text/config_text_rollout_qwen3p5_9b_jsonfirst_medium_overfit_grpo_6epoch_1024_8gpu.yaml](configs/archive/text/config_text_rollout_qwen3p5_9b_jsonfirst_medium_overfit_grpo_6epoch_1024_8gpu.yaml): Qwen3.5-9B no-BFS `ghost_legal_json_first` 6epoch overfit config, using JSON-first prompt and `max_new_tokens=1024` to test whether the `256` run was bottlenecked by verbose pre-JSON output.
- [configs/archive/text/config_text_rollout_qwen3p5_9b_jsonfirst_medium_overfit_grpo_6epoch_256_8gpu.yaml](configs/archive/text/config_text_rollout_qwen3p5_9b_jsonfirst_medium_overfit_grpo_6epoch_256_8gpu.yaml): Qwen3.5-9B no-BFS JSON-first short-output diagnostic config.
- [configs/archive/text/config_text_rollout_qwen3p5_9b_actiononly_medium_overfit_grpo_6epoch_8tok_8gpu.yaml](configs/archive/text/config_text_rollout_qwen3p5_9b_actiononly_medium_overfit_grpo_6epoch_8tok_8gpu.yaml): Qwen3.5-9B no-BFS action-only diagnostic config with `max_new_tokens=8`.
- [configs/archive/text/config_text_rollout_qwen3p5_9b_actiononly_guided_medium_overfit_grpo_6epoch_8192_8gpu.yaml](configs/archive/text/config_text_rollout_qwen3p5_9b_actiononly_guided_medium_overfit_grpo_6epoch_8192_8gpu.yaml): Qwen3.5-9B no-BFS action-only guided-choice overfit config with `max_new_tokens=8192`.
- [configs/archive/text/config_text_rollout_qwen3p5_9b_actiononly_medium_overfit_grpo_6epoch_8192_parserfix_8gpu.yaml](configs/archive/text/config_text_rollout_qwen3p5_9b_actiononly_medium_overfit_grpo_6epoch_8192_parserfix_8gpu.yaml): Qwen3.5-9B no-BFS action-only parser-fix config with `max_new_tokens=8192`.
- [level1_image_anticollapse_4update_group12_8gpu.yaml](configs/level1/archive/level1_image_anticollapse_4update_group12_8gpu.yaml): production original-pygame image-only gate with group size 12, exactly four updates, temperature `0.7`, LR `1.5e-6`, KL `0.01`, and separate greedy validation.
- [level1_image_progress_4update_group12_8gpu.yaml](configs/level1/archive/level1_image_progress_4update_group12_8gpu.yaml): isolated nearest-pellet alpha-1 follow-up with matched sampled12 in-training validation and separately reported greedy1 validation.
- [level1_live_state_4update_group12_8gpu.yaml](configs/level1/archive/level1_live_state_4update_group12_8gpu.yaml): live-state follow-up with `open_action_mask: true`; every model request is dynamically constrained to the current engine-reported open movement actions, and [areal_pacman_action_logprobs.patch](patches/areal_pacman_action_logprobs.patch) applies the identical mask and temperature to actor/reference log-probabilities. This config uses `top_p=1.0` and vLLM `processed_logprobs` so cached old and recomputed new probabilities describe the same policy.
- [evaluate_level1.py](scripts/level1/evaluate/evaluate_level1.py): thinking-disabled greedy or sampled level-1 evaluator with independent decoding, prompt selection, parallel episodes, and collision-free trajectories.
- [reevaluate_level1_group12.sh](scripts/level1/evaluate/reevaluate_level1_group12.sh): corrected base/all-checkpoint greedy evaluation plus 24 matched sampled final episodes.
- [run_level1_prompt_ab.sh](scripts/level1/evaluate/run_level1_prompt_ab.sh): base/final `minimal_v1` versus static live-inspired image-only prompt A/B.
- [summarize_level1_trajectories.py](scripts/level1/report/summarize_level1_trajectories.py): collision-free chronological train/validation batch audit.
- [evaluate_level1_run.sh](scripts/level1/evaluate/evaluate_level1_run.sh): sequential one-GPU dual validation for base plus all four updates: greedy1 at `0/1.0` and sampled12 at the training `0.7/0.95` decoding.
- [compare_level1_run.py](scripts/level1/evaluate/compare_level1_run.py): deterministic dual-validation report builder with separate sampled-primary and greedy checkpoint labels.
- [smoke_tms_offload.py](scripts/level1/report/smoke_tms_offload.py): H100 torch-memory-saver pause/resume integration smoke.
- [sitecustomize.py](sitecustomize.py): optional forced-host-IP patch for node5, used with `AREAL_FORCE_HOST_IP=10.0.12.183` to avoid AReaL advertising `169.254.100.1`.
- [example_trajectory_1p5b.md](../trajectories/example_trajectory_1p5b.md): readable first trajectory from the 1.5B H100 rollout.
- [failed_trajectory_7b_medium_default.md](../trajectories/failed_trajectory_7b_medium_default.md): readable failed 7B medium-map sparse trajectory, showing the `down/up` loop.
- [failed_trajectory_7b_reason_debug.md](../trajectories/failed_trajectory_7b_reason_debug.md): readable 7B debug trajectory with short reasons plus `Action: <token>`.
- [2026-07-09_clean_report_index.md](../reports/2026-07-09_clean_report_index.md): clean start point for the report set.
- [2026-07-09_thinking_disable_mistake_and_new_goal.md](../reports/2026-07-09_thinking_disable_mistake_and_new_goal.md): current correction note about `enable_thinking=false` and the new decoding-control goal.
- [2026-07-09_qwen3p5_9b_nohint_overfit_blocker_report.md](../reports/2026-07-09_qwen3p5_9b_nohint_overfit_blocker_report.md): main no-hint overfit blocker report.
- [2026-07-09_qwen3p5_9b_medium_default_ghost_legal_json_structured_choice_report.md](../reports/2026-07-09_qwen3p5_9b_medium_default_ghost_legal_json_structured_choice_report.md): structured-choice evidence for `ghost_legal_json`.
- [trajectory_qwen3p5_9b_best6_ckpt_greedy_success.md](../trajectories/trajectory_qwen3p5_9b_best6_ckpt_greedy_success.md): readable greedy success trajectory from the best 6epoch checkpoint final exam.
- [2026-07-07_qwen3p5_qwen3p6_rollout_training_report.md](../reports/2026-07-07_qwen3p5_qwen3p6_rollout_training_report.md): report for Qwen3.5-9B, Qwen3.6-27B, and Qwen3.6-35B-A3B no-training rollouts plus 9B/27B training attempts.
- [multi_maze_vlm_rl_report_2026-07-15.md](../reports/multi_maze_vlm_rl_report_2026-07-15.md): completed `64/16/32` topology-disjoint image-only VLM baseline, GRPO, and held-out generalization report, with matching [PPTX](../reports/multi_maze_vlm_rl_report_2026-07-15.pptx) and [PDF](../reports/multi_maze_vlm_rl_report_2026-07-15.pdf).
- [no_training_trajectory_qwen3p5_9b.md](../trajectories/no_training_trajectory_qwen3p5_9b.md): no-training trajectory for `Qwen/Qwen3.5-9B`.
- [no_training_trajectory_qwen3p5_9b_distance_json_medium_temp12_seed3_failed.md](../trajectories/no_training_trajectory_qwen3p5_9b_distance_json_medium_temp12_seed3_failed.md): readable failed seed from the `medium_default`, distance-feature, `temperature=1.2` no-training pass-rate run.
- [no_training_trajectory_qwen3p6_27b.md](../trajectories/no_training_trajectory_qwen3p6_27b.md): no-training trajectory for `Qwen/Qwen3.6-27B`.
- [no_training_trajectory_qwen3p6_35b_a3b.md](../trajectories/no_training_trajectory_qwen3p6_35b_a3b.md): no-training trajectory for `Qwen/Qwen3.6-35B-A3B`.
- [training_trajectory_qwen3p5_9b_reason_debug.md](../trajectories/training_trajectory_qwen3p5_9b_reason_debug.md): rollout sample from the `Qwen/Qwen3.5-9B` training attempt.
- [tests](tests/): local smoke tests.

## 本地 Smoke

```bash
cd /home/xinglu/life/vla-research/areal-pacman
python3 -m pytest -q
python3 -m areal_pacman.synthetic.evaluate --agent greedy --episodes 20
python3 -m areal_pacman.synthetic.dataset --output run_artifacts/pacman_prompts.jsonl --episodes 5
python3 -m areal_pacman.synthetic.dataset --mode episode --output run_artifacts/pacman_episode_specs.jsonl --episodes 5
```

## Trajectory Logging

Set `PACMAN_TRAJECTORY_DIR` to save one JSON file per rollout episode:

```bash
PACMAN_TRAJECTORY_DIR=run_artifacts/trajectories python3 train_areal.py --config configs/archive/text/config_text_rollout.yaml
```

Production level-1 filenames include both the dataset row ID and a random
`trajectory_sample_id`. Repeated GRPO samples therefore never overwrite one
another. Each JSON records environment revisions, prompt/decoding contract,
verbatim model responses, base and shaped rewards, score, collectibles, wall
collisions, and terminal state.

The production image-only workflow always sends
`chat_template_kwargs.enable_thinking=false`. Training rollout decoding comes
from `gconfig`; validation decoding independently comes from `eval_gconfig`.
The current progress follow-up uses sampled group-12 training and matched
sampled12 in-training validation at `temperature=0.7`, `top_p=0.95`. Its
post-run evaluator separately reports greedy1 at `temperature=0`,
`top_p=1.0` for base and every checkpoint. Both validation modes require
`enable_thinking=false` and zero observed reasoning turns.

Summarize saved trajectories:

```bash
python3 -m areal_pacman.analyze_trajectories run_artifacts/trajectories
python3 -m areal_pacman.analyze_trajectories run_artifacts/trajectories --show-first --steps 8
python3 -m areal_pacman.synthetic.dataset --mode episode --output run_artifacts/pacman_episode_legal_specs.jsonl --episodes 8 --max-steps 30 --prompt-style legal --illegal-action-penalty -4
```

## Multi-Maze V1

`multi_maze_v1` contains `64` train, `16` validation, and `32` held-out test layouts. All `112` full-layout hashes and all `112` wall-topology hashes are unique across splits. Every maze has exactly five pellets and a successful oracle replay under the real ghost dynamics, legal non-stay decoding, and no-immediate-backtracking rule within `30` steps.

Completed result: the `24/24`-step GRPO run raised train macro pass rate from `3.9%` to `5.5%`, but validation fell from `12.5%` to `4.7%` and held-out test fell from `5.1%` to `3.1%`. The pipeline is operational, but this short sparse-GRPO recipe did not generalize. See the [detailed report](../reports/multi_maze_vlm_rl_report_2026-07-15.md); the ignored local comparison artifact is `run_artifacts/multi_maze_v1_20260715/comparison.json`.

The alpha-4 safe-progress ablation also completed `24/24` updates. Validation selected epoch 0 from macro rates `12.91% / 10.07% / 10.63%`; selected train/validation/test rates were `5.08% / 12.91% / 0.00%` versus frozen baseline `3.91% / 12.50% / 5.08%`. Reward audit covered `18,032` training steps with zero alpha/formula mismatch. Denser safe progress alone did not improve held-out topology generalization. The ignored local comparison artifact is `run_artifacts/multi_maze_v1_safeprogress_alpha4_20260715/comparison.json`.

```bash
cd /home/xinglu/life/vla-research/areal-pacman
python scripts/synthetic/build_multi_maze_suite.py
pytest -q tests/test_env.py tests/test_dataset_workflow.py

for split in train validation test; do
  python -m areal_pacman.prepare_hf_dataset \
    --mode episode \
    --maze-split "$split" \
    --episodes-per-layout 1 \
    --max-steps 30 \
    --prompt-style ghost_legal_strict \
    --reward-mode sparse \
    --jsonl "run_artifacts/multi_maze_v1/${split}.jsonl" \
    --hf-dir "run_artifacts/multi_maze_v1/${split}_hf_dataset"
done
```

Training uses [multi-maze GRPO config](configs/archive/vision/config_vision_rollout_qwen3p5_9b_imageonly_multimaze_v1_grpo_3epoch_4sample_8gpu.yaml): `64` train mazes, `n_samples=4`, `3` epochs, `batch_size=8`, and `24` optimizer steps. Frozen base/post-RL evaluation uses [multi-maze no-train config](configs/archive/vision/config_vision_rollout_qwen3p5_9b_imageonly_multimaze_v1_notrain_4sample_8gpu.yaml), with `lr=0`, `eps_clip=0`, and no actor update.

The [safe-progress config](configs/archive/vision/config_vision_rollout_qwen3p5_9b_imageonly_multimaze_v1_safeprogress_grpo_3epoch_4sample_8gpu.yaml) keeps those settings fixed and adds `reward_mode=safe_progress`, `safe_progress_alpha=4.0`.

On an idle node5, run the complete guarded pipeline:

```bash
cd /home/z00819216/vla-research/areal-pacman
nohup bash scripts/synthetic/remote_supervise_multimaze_v1.sh \
  > run_artifacts/multi_maze_v1_20260715_supervisor.log 2>&1 &
```

The supervisor aborts if any GPU compute process exists. It never stops an existing GPU process. Final split summaries and `comparison.json` are written under `run_artifacts/multi_maze_v1_20260715/`.

If node5 is busy with a known non-Dummy workload, use the detached waiter. It requires two consecutive empty-GPU checks one minute apart, uses a lock to prevent duplicate launches, and never kills a process:

```bash
nohup bash scripts/synthetic/remote_wait_for_idle_multimaze_v1.sh \
  > run_artifacts/multi_maze_v1_20260715_waiter.log 2>&1 &
```

## Node5 Dry Run

```bash
cd /home/z00819216/vla-research/areal-pacman
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --episodes 5
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_pacman.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --mode episode --jsonl run_artifacts/pacman_episode_specs.jsonl --hf-dir run_artifacts/pacman_episode_hf_dataset --episodes 5
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --mode episode --jsonl run_artifacts/pacman_episode_longer_specs.jsonl --hf-dir run_artifacts/pacman_episode_longer_hf_dataset --episodes 8 --max-steps 30
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_longer.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_1p5b.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --mode episode --jsonl run_artifacts/pacman_episode_legal_specs.jsonl --hf-dir run_artifacts/pacman_episode_legal_hf_dataset --episodes 8 --max-steps 30 --prompt-style legal --illegal-action-penalty -4
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_1p5b_legal.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --mode episode --jsonl run_artifacts/pacman_episode_rules_specs.jsonl --hf-dir run_artifacts/pacman_episode_rules_hf_dataset --episodes 8 --max-steps 30 --prompt-style default --illegal-action-penalty 0
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_1p5b_rules.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_1p5b_grpo.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --mode episode --jsonl run_artifacts/pacman_episode_overfit_specs.jsonl --hf-dir run_artifacts/pacman_episode_overfit_hf_dataset --episodes 4 --max-steps 30 --prompt-style ghost_legal --illegal-action-penalty -4
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_1p5b_overfit_grpo.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --mode episode --jsonl run_artifacts/pacman_episode_tiny_overfit_specs.jsonl --hf-dir run_artifacts/pacman_episode_tiny_overfit_hf_dataset --episodes 4 --max-steps 4 --prompt-style ghost_legal --illegal-action-penalty -4 --layout-name tiny_corridor
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_1p5b_tiny_overfit_grpo.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --mode episode --jsonl run_artifacts/pacman_episode_teacher_overfit_specs.jsonl --hf-dir run_artifacts/pacman_episode_teacher_overfit_hf_dataset --episodes 4 --max-steps 30 --prompt-style teacher_hint --illegal-action-penalty -4
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_1p5b_teacher_overfit_grpo.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --mode episode --jsonl run_artifacts/pacman_episode_teacher_pure_overfit_specs.jsonl --hf-dir run_artifacts/pacman_episode_teacher_pure_overfit_hf_dataset --episodes 4 --max-steps 30 --prompt-style teacher_hint_pure --illegal-action-penalty -4
/home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_1p5b_teacher_pure_overfit_grpo.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --mode episode --jsonl run_artifacts/pacman_episode_medium_default_specs.jsonl --hf-dir run_artifacts/pacman_episode_medium_default_hf_dataset --episodes 4 --max-steps 20 --prompt-style ghost_legal_strict --illegal-action-penalty -4 --layout-name medium_default
PYTHONPATH=/home/z00819216/vla-research/areal-pacman:$PYTHONPATH AREAL_FORCE_HOST_IP=10.0.12.183 /home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_7b_medium_default_overfit_grpo_8gpu.yaml --dry-run
/home/z00819216/miniconda/envs/areal-vla/bin/python -m areal_pacman.prepare_hf_dataset --mode episode --jsonl run_artifacts/pacman_episode_medium_default_reason_specs.jsonl --hf-dir run_artifacts/pacman_episode_medium_default_reason_hf_dataset --episodes 4 --max-steps 20 --prompt-style ghost_legal_reason --illegal-action-penalty -4 --layout-name medium_default
PYTHONPATH=/home/z00819216/vla-research/areal-pacman:$PYTHONPATH AREAL_FORCE_HOST_IP=10.0.12.183 /home/z00819216/miniconda/envs/areal-vla/bin/python train_areal.py --config configs/archive/text/config_text_rollout_7b_medium_default_reason_debug_8gpu.yaml --dry-run
```

## AReaL 方向

当前 AReaL agentic RL docs 推荐 proxy-style workflow: 先写普通 agent loop, 再让 AReaL 提供 OpenAI-compatible model endpoint, 并在训练期间收集 token-level traces. 这个 scaffold 明确保留这条边界.

## Prompt 方向

当前 [dataset.py](areal_pacman/synthetic/dataset.py) 的 system prompt 已经写明 text PacMan 规则: `#` wall, `P` PacMan, `G` ghost, `.` pellet, wall move penalty, pellet objective, ghost terminal, win condition, and one-token action output. 这是为了以后迁移到更复杂 true PacMan 时, prompt contract 可以逐步扩展, 而不是只靠 observation 里的 `Legal actions`.

`prompt_style=ghost_legal` 会在 observation 里额外写出 `Unsafe immediate ghost actions`, 用来测试模型能不能同时遵守 legal action 和避开下一步 ghost.

`layout_name=tiny_corridor` 是一个两步小地图 curriculum: PacMan 只需要连续输出 `right` 吃完两个 pellet. 它不包含 teacher hint, 用来证明 pure text sparse-reward path 在最小任务上可以通关.

`layout_name=small_default` 是用户提出的 4-row default-like 小图. 当前 ghost 规则下 BFS 找不到 winning path, 所以它保留为 hard diagnostic, 不作为 pass gate.

`layout_name=medium_default` 使用 default 的通路结构, 但只保留 5 个 pellets. BFS shortest path 是 10 步: `right, right, down, down, right, right, up, up, right, right`. 它是 `tiny_corridor` 和 full default 之间的 passable curriculum gate.

`prompt_style=ghost_legal_strict` 会在 observation 里写 `Allowed output tokens now` 和 `Forbidden output tokens now`. 它不是 teacher hint, 只强调当前 legal token constraint.

`prompt_style=ghost_legal_reason` 是 debug-only 模式, 允许模型先写一两句 reason, 最后用单独一行 `Action: <token>` 给动作. Parser 会优先读 `Action:` 行, 避免把解释里的第一个方向词误当动作. 这个模式适合看模型为什么选择 `down/up`, 暂不作为默认 RL 训练格式.

`prompt_style=ghost_legal_json_first` 是 no-BFS JSON control 模式. 它仍然只给 grid, allowed tokens, forbidden tokens, and unsafe ghost actions, 不给 teacher action, BFS route, or distance feature. 与 `ghost_legal_json` 的区别是硬性要求答案第一字符为 `{`, 并要求不要先写 reasoning text, 用来避免 Qwen3.5-9B 在 `max_new_tokens=256` 下因为 verbose preamble 被截断后 fallback 到 `stay`.

`prompt_style=teacher_hint` 会在 observation 里写出 `Teacher action hint`. 当前 teacher-enforced overfit gate 会记录模型输出, 但执行 `env.teacher_action()` 来验证 env, AReaL rollout, reward, and trajectory pipeline 是否能稳定通关. 这不是 pure model policy pass.

`prompt_style=teacher_hint_pure` 也会在 observation 里写出 `Teacher action hint`, 但 workflow 执行模型 parsed action, 不执行 `env.teacher_action()`. 这个 gate 用来检查模型 raw output 能不能稳定复制 hint 并通关.

## 当前两个 workflow

- [configs/archive/text/config_pacman.yaml](configs/archive/text/config_pacman.yaml): 单步 smoke, 使用 baseline 生成的 `answer` 给 `0/1` reward.
- [configs/archive/text/config_text_rollout.yaml](configs/archive/text/config_text_rollout.yaml): pure text episode rollout, 不使用 ground truth answer, reward 来自 [PacmanEnv.step](areal_pacman/synthetic/env.py).
- [configs/archive/text/config_text_rollout_longer.yaml](configs/archive/text/config_text_rollout_longer.yaml): 比 smoke 稍大的 text rollout, 用于检查多 episode reward 和 trajectory 分布.
- [configs/archive/text/config_text_rollout_1p5b.yaml](configs/archive/text/config_text_rollout_1p5b.yaml): 1.5B model-size ablation, 用同一套 episode 和 analyzer 对比 0.5B.
- [configs/archive/text/config_text_rollout_1p5b_legal.yaml](configs/archive/text/config_text_rollout_1p5b_legal.yaml): 1.5B legal-action ablation, 用 `prompt_style=legal` 和 `illegal_action_penalty=-4`.
- [configs/archive/text/config_text_rollout_1p5b_rules.yaml](configs/archive/text/config_text_rollout_1p5b_rules.yaml): 1.5B rules-prompt PPO baseline, 用 expanded system prompt 但不加 illegal-action penalty.
- [configs/archive/text/config_text_rollout_1p5b_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_grpo.yaml): 1.5B GRPO-style ablation, same rules dataset, `n_samples=4`, group reward normalization.
- [configs/archive/text/config_text_rollout_1p5b_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_overfit_grpo.yaml): 1.5B overfit gate, 4 fixed episodes, `prompt_style=ghost_legal`, `illegal_action_penalty=-4`, `total_train_epochs=8`.
- [configs/archive/text/config_text_rollout_1p5b_tiny_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_tiny_overfit_grpo.yaml): 1.5B tiny sparse overfit gate, `layout_name=tiny_corridor`, `prompt_style=ghost_legal`, no teacher hint.
- [configs/archive/text/config_text_rollout_1p5b_teacher_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_teacher_overfit_grpo.yaml): 1.5B teacher-hint overfit gate, 4 fixed episodes, `prompt_style=teacher_hint`, `illegal_action_penalty=-4`.
- [configs/archive/text/config_text_rollout_1p5b_teacher_pure_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_teacher_pure_overfit_grpo.yaml): 1.5B pure hint-only overfit gate, 4 fixed episodes, `prompt_style=teacher_hint_pure`, `greedy=false`, `temperature=0.1`, no execution-time teacher override.
- [configs/archive/text/config_text_rollout_1p5b_long_debug_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_long_debug_grpo.yaml): larger debug config, only after overfit succeeds.
- [configs/archive/text/config_text_rollout_7b_overfit_grpo_8gpu.yaml](configs/archive/text/config_text_rollout_7b_overfit_grpo_8gpu.yaml): 7B default-map sparse overfit on 8 GPUs, no teacher hint.
- [configs/archive/text/config_text_rollout_7b_medium_default_overfit_grpo_8gpu.yaml](configs/archive/text/config_text_rollout_7b_medium_default_overfit_grpo_8gpu.yaml): 7B medium-map sparse overfit on 8 GPUs, no teacher hint.

## 当前 H100 对比

| Config | Episodes | Wins | Avg reward | Illegal actions | Terminal reasons |
| --- | ---: | ---: | ---: | ---: | --- |
| [configs/archive/text/config_text_rollout_longer.yaml](configs/archive/text/config_text_rollout_longer.yaml) | 20 | 0 | -20.00 | n/a | `max_steps=20` |
| [configs/archive/text/config_text_rollout_1p5b.yaml](configs/archive/text/config_text_rollout_1p5b.yaml) | 20 | 0 | 3.65 | n/a | `caught=3`, `max_steps=17` |
| [configs/archive/text/config_text_rollout_1p5b_legal.yaml](configs/archive/text/config_text_rollout_1p5b_legal.yaml) | 20 | 0 | -10.90 | 1 | `caught=4`, `max_steps=16` |
| [configs/archive/text/config_text_rollout_1p5b_rules.yaml](configs/archive/text/config_text_rollout_1p5b_rules.yaml) | 20 | 0 | 5.20 | 350 | `caught=3`, `max_steps=17` |
| [configs/archive/text/config_text_rollout_1p5b_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_grpo.yaml) | 64 | 0 | 0.30 | 148 | `caught=12`, `max_steps=52` |
| [configs/archive/text/config_text_rollout_1p5b_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_overfit_grpo.yaml) | 264 | 0 | -56.51 | 3879 | `caught=6`, `max_steps=258` |
| [configs/archive/text/config_text_rollout_1p5b_tiny_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_tiny_overfit_grpo.yaml) tiny sparse | 12 | 12 | 118.00 | 0 | `all_pellets=12` |
| [configs/archive/text/config_text_rollout_1p5b_teacher_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_teacher_overfit_grpo.yaml) teacher-enforced | 40 | 40 | 264.00 | 0 | `all_pellets=40` |
| [configs/archive/text/config_text_rollout_1p5b_teacher_pure_overfit_grpo.yaml](configs/archive/text/config_text_rollout_1p5b_teacher_pure_overfit_grpo.yaml) pure hint-only | 12 | 12 | 264.00 | 0 | `all_pellets=12` |
| [configs/archive/text/config_text_rollout_7b_overfit_grpo_8gpu.yaml](configs/archive/text/config_text_rollout_7b_overfit_grpo_8gpu.yaml) default sparse 8GPU | 256 | 0 | -78.93 | 3569 | `max_steps=256` |
| [configs/archive/text/config_text_rollout_7b_medium_default_overfit_grpo_8gpu.yaml](configs/archive/text/config_text_rollout_7b_medium_default_overfit_grpo_8gpu.yaml) medium sparse 8GPU | 207 | 0 | -78.91 | 2541 | `max_steps=207` |

Interpretation: the expanded game-rules prompt helps reward, but still allows many wall moves. The GRPO-style config reduces illegal actions relative to rules PPO, but creates many `stay` actions, so reward drops. The default-map pure sparse-reward overfit gate failed even on 4 fixed episodes. 7B plus 8 GPUs fixes the system-capacity/OOM problem, but not the learning problem. The medium-map run shows a different failure mode: strict legal prompt reduces some wall-repeat behavior but learns a `down/up` loop without eating pellets. The tiny sparse gate passed with `has_teacher_hint=False`, `has_teacher_action=False`, and `all_exact=True`, proving a no-teacher curriculum path works only at the smallest scale. The pure hint-only gate also passed on the default map, but it is still a warm-start/action-prior gate, not yet sparse-reward learning from scratch.
