# AReaL Pacman RL Recipe Design

Status: executable recipe and two-epoch H100 training gate complete; greedy
overfit-improvement gate not met  
Environment provider: MaaPacman `PygamePacmanEnv` API `1.0`  
Environment ID: `pacman-python-level1-pygame-v1`  
Production episode cap: `287` actions  
Remote Linux backend: SDL dummy  
Last verified: `2026-07-23`

Companion design:
[MaaPacman Environment Interface Design](../MaaPacman/PACMAN_ENV_DESIGN.md)

Workspace paths:

```text
C:\Users\x84241863\Desktop\life\MaaPacman
C:\Users\x84241863\Desktop\life\pacman-python
C:\Users\x84241863\Desktop\life\areal-pacman
```

## 1. Ownership

```text
pacman-python
  owns the original game rules, resources and pygame renderer

MaaPacman
  owns the external process wrapper, frame-boundary action protocol,
  RGB Surface extraction and stable environment API

areal-pacman
  owns episode datasets, prompts, model calls, parsing, reward shaping,
  trajectory records, AReaL configuration, checkpoints and evaluation
```

`pacman-python` is a sibling dependency and must remain source-clean. It must
not import MaaPacman or AReaL.

### 1.1 Production environment API

Production level-1 files import only `PygamePacmanEnv`:

- `areal_pacman/workflow.py`
- `areal_pacman/level1_dataset.py`
- `train_areal.py` production dry-run path
- `scripts/write_level1_manifest.py`
- `tests/test_level1_recipe.py`

The accepted environment import is
`from maapacman.env import PygamePacmanEnv, PygamePacmanEnvConfig`. The recipe
does not define or expose an alternative Pacman environment contract.

## 2. Episode architecture

```text
episode row
  -> PacmanImageOnlyWorkflow
  -> PygamePacmanEnv.reset()
  -> original pacman-python process and pygame Surface
  -> RGB observation
  -> AReaL multimodal rollout endpoint
  -> VLM completion
  -> canonical U/D/L/R/S parser
  -> PygamePacmanEnv.step(action)
  -> original score delta and state metrics
  -> recipe reward adapter
  -> completion-token reward
  -> repeat until terminated or truncated
```

The environment is a local Python object. The OpenAI-compatible HTTP API is
only between AReaL and the model server; it is not the game protocol.

After reset and after every completed action transaction, the pygame worker is
paused at the returned `display.flip()` boundary while AReaL waits for the
model response. Pacman, ghosts, timers, and animation do not advance during
model inference. Model latency therefore changes rollout wall time but does not
create hidden game frames.

## 3. Required installation and mirror

The production training node needs a persistent three-repository mirror under
the confirmed owner-specific `z0xxx` root:

```text
/home/ubuntu/z00819216/maapacman-stack/MaaPacman
/home/ubuntu/z00819216/maapacman-stack/pacman-python
/home/ubuntu/z00819216/maapacman-stack/areal-pacman
```

The node5 owner root has been confirmed as `/home/ubuntu/z00819216`, a symlink
to `/mnt/data/z00819216`. Do not reuse this path on another node without a live
ownership and symlink check.

The AReaL framework checkout is intentionally outside the deployable recipe
mirror and is separated from robotics development:

```text
/home/ubuntu/z00819216/xinglu/AReaL
  local branch: areal-main
  upstream: https://github.com/inclusionAI/AReaL.git main

/home/ubuntu/z00819216/xinglu/AReaL-VLA
  local branch: robotics/morgan-vla
  purpose: preserve Morgan/robotics development and its existing dirty state
```

The physical official paths are
`/mnt/data-node1/z00819216/xinglu/AReaL` on node1 and
`/mnt/data/z00819216/xinglu/AReaL` on node5. The `maapacman-rl` editable
binding and production launcher must resolve `areal` from this official
worktree. The launcher prepends `AREAL_ROOT` to `PYTHONPATH` and rejects an
import resolved outside it. `AReaL-VLA` is not a Pacman training dependency.

Use a project-specific Conda prefix rather than system Python or an unrelated
existing environment:

```bash
OWNER_ROOT=/home/ubuntu/z00819216
CODE_ROOT="$OWNER_ROOT/maapacman-stack"
ENV_ROOT="$OWNER_ROOT/miniconda/envs/maapacman-rl"

"$OWNER_ROOT/miniconda/bin/conda" create -y \
  -p "$ENV_ROOT" --clone "$OWNER_ROOT/miniconda/envs/areal-vla"

"$ENV_ROOT/bin/python" -m pip install "pygame==2.6.1"
"$ENV_ROOT/bin/python" -m pip install \
  -e "$CODE_ROOT/MaaPacman[pygame]" \
  -e "$CODE_ROOT/areal-pacman"
```

Pin and record the three Git revisions before training. Treat the
`pacman-python` mirror as read-only during rollout. Per-worker copies,
`agent_state.json`, pygame processes, and IPC remain disposable under `/tmp`.

The H100 validation deliberately used `/tmp + pip --target` so it could be
removed without touching persistent environments. That method proves runtime
compatibility but is not the production installation recipe.

The existing `/home/ubuntu/miniconda3/envs/pacman_gym` environment was audited
but not modified. It currently uses Python `3.11.15` and lacks pygame and
MaaPacman, so it is not the accepted recipe environment.

The launcher sets `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`.
Xvfb is not part of the recipe runtime or deployment dependencies. Both node1
and node5 have passed the original-pygame worker and oracle gates with SDL
dummy, so the recipe has one Linux display contract rather than two branches.

## 4. Environment construction

```python
from maapacman.env import PygamePacmanEnv, PygamePacmanEnvConfig

env = PygamePacmanEnv(
    PygamePacmanEnvConfig(
        pacman_python_root=(
            "/home/ubuntu/z00819216/maapacman-stack/pacman-python"
        ),
        level=1,
        max_steps=287,
        video_driver="dummy",
        audio_driver="dummy",
    )
)
```

`287` is the production recipe contract, not merely a permissive wrapper
default. The workflow must pass it explicitly rather than relying on the
current `PygamePacmanEnvConfig` default.

The workflow validates before rollout:

```python
if env.spec.api_version != "1.0":
    raise RuntimeError("unsupported original-pygame environment API")
if env.spec.env_id != "pacman-python-level1-pygame-v1":
    raise RuntimeError("wrong Pacman environment")
if env.spec.action_tokens != ("U", "D", "L", "R", "S"):
    raise RuntimeError("incompatible action contract")
if env.config.max_steps != 287:
    raise RuntimeError("production level-1 recipe requires max_steps=287")
```

It must import only the public `maapacman.env` API, not the worker module.

## 5. Dataset contract

One row constructs one complete original-game episode:

```json
{
  "id": "level1-seed0-train-0001",
  "split": "train",
  "env": {
    "name": "pacman-python-level1-pygame-v1",
    "api_version": "1.0",
    "backend": "original-pygame",
    "pacman_python_revision": "d258122eecf6e0dc0a04d6fb8ff57a9b43f0c1d8",
    "level_revision": "36116c17c6c0805fdb1a07216357ac64c88d2c3108a0e37dce2a01b4ea2a8b97",
    "level": 1,
    "seed": 0,
    "max_steps": 287,
    "observation_mode": "rgb"
  }
}
```

Do not reuse historical rows labeled `maapacman-level1-v1` or API `1.0`
without checking their `env_id`: those rows describe the older private AReaL
environment, not this original-pygame wrapper.

Production level-1 dataset validation must require `env.max_steps == 287`.
This cap matches the verified nearest-normal-pellet oracle exactly. On action
287 the original game reaches mode `6`; `terminated=True` takes precedence
over the step cap, so that successful final action must not be reported as
`truncated=True`. A row using a shorter or longer cap is a different experiment
and must not be mixed into the production training/evaluation split.

## 6. Observation and action protocol

Each model turn contains:

1. One fixed system prompt.
2. Exactly one PNG encoded from the current `(400,336,3)` RGB Surface.
3. One fixed instruction to return exactly one action.

Allowed response tokens are `U`, `D`, `L`, `R`, and `S`. Parsing failure must
not send arbitrary text to the environment; the recipe applies its documented
fallback and parse penalty.

One environment step is one action transaction ending at an original pygame
frame boundary. Directional input moves one cell in the committed level-1
game; the transaction may run more than one internal frame until that grid
transition completes. Completion is detected from game state and the
corresponding `display.flip()`, never from a fixed sleep. `S` sends no
directional event and advances exactly one original frame.

The worker blocks inside the wrapped `display.flip()` after reset and after a
completed transaction. It resumes only when AReaL sends the next action. This
pause is part of the environment contract: an action selected after a `50 ms`
or a `5 s` model call is applied to the same returned state.

Every rollout worker owns a unique temporary runtime directory, original script
copy, resource link/private copy, pygame process, IPC channel, state file, and
Surface. Workers do not own an X server or `DISPLAY`; pygame renders through
SDL dummy and MaaPacman reads the completed pygame Surface directly.

### Timing and IPC decision

The current implementation uses newline-delimited JSON over subprocess pipes.
Actions and state use small messages. Each RGB frame is copied from the pygame
Surface, compressed with zlib level 1, base64 encoded into JSON, and decoded by
the parent. Cross-process shared memory is not currently used.

The unchanged original loop still executes `clock.tick(60)` after the paused
`flip()` resumes. `tick(60)` measures elapsed wall time since its previous
call, including time spent paused inside `flip()` waiting for the model. It
waits only when that total elapsed time is less than approximately `16.67 ms`;
it does not add another `16.67 ms` after a slower model call.

Local profiling with real `L/R` movement and the committed renderer measured:

```text
1 worker env.step:                p50 16.81 ms, p95 18.58 ms
16 worker env.step:               p50 40.52 ms, p95 61.80 ms
zlib+base64+JSON frame roundtrip: p50  1.38 ms, p95  2.07 ms
shared-memory two-copy estimate:  p50  0.024 ms, p95 0.037 ms
small pipe notification:          p50  0.053 ms, p95 0.091 ms
1 worker, simulated 50 ms model wait:
  action to observation:          p50 11.80 ms, p95 14.98 ms
  complete model+environment turn:p50 62.13 ms, p95 65.18 ms
16 workers, simulated 50 ms model wait:
  action to observation:          p50 14.52 ms, p95 21.09 ms
  complete model+environment turn:p50 64.88 ms, p95 71.49 ms
```

The immediate-action `16.81 ms` result is the rate-limited scripted-agent case;
it must not be added directly to a `50 ms` model call. With the simulated
`50 ms` wait, the measured single-worker p50 is `62.13 ms` per complete turn,
or approximately `17.83 s` for 287 actions. The corresponding 16-worker local
p50 is `64.88 ms`, or approximately `18.62 s` per worker episode. These are
local measurements, not H100 guarantees.

The current RGB codec represents about `0.40 s` of serial CPU work across 287
frames. Replacing RGB pipe payloads with shared memory is estimated to save
about `0.38 s` per worker episode. The `clock.tick(60)` wait has little or no
sleep to remove when model latency already exceeds the `16.67 ms` frame
budget; it is a bottleneck only for faster scripted or low-latency policies.

The recipe keeps both pipe IPC and the original `clock.tick(60)` behavior.
H100 profiling with the real model and selected rollout concurrency decides
only whether the shared-memory IPC optimization is justified. Changing or
bypassing the original pygame clock is outside the recipe design.

A later hybrid protocol may keep pipe messages for actions, state, request IDs,
and frame sequence while placing each worker's RGB buffer in shared memory.
This is required only if H100 profiling at the selected rollout concurrency
shows frame serialization causing CPU saturation; it does not block the first
training gate.

## 7. Reward ownership

MaaPacman returns the original game score delta:

| Original event | Base reward |
|---|---:|
| Empty move, wall or `S` | `0` |
| Normal pellet | `10` |
| Power pellet | `100` |

AReaL may add:

- Progress shaping.
- Wall penalties.
- Parse-failure penalties.
- Episode completion bonuses.
- Other experiment-specific terms.

Every step record stores both `base_reward` and `shaped_reward`.

The committed original game enters mode `6` when normal pellets reach zero;
power pellets are not part of that internal clear counter. For this revision,
the terminal reason is therefore `all_normal_pellets`. The recipe must not
reinterpret it as `all_pellets`.

## 8. Trajectory provenance

Each trajectory records at least:

```text
env_api_version
env_id
backend
pacman_python_revision
level_revision
renderer_revision
seed and max_steps
RGB frame hashes where requested
action and parse status
base_reward and shaped_reward
score and collectible counts
pygame_mode
terminated, truncated and terminal_reason
```

Training outputs from different `pacman_python_revision` values must not be
merged as if they used the same environment.

## 9. Current executable and evaluation contract

The production workflow constructs `PygamePacmanEnv` directly and validates
`pacman-python-level1-pygame-v1`. The current local and node5 suites pass `107`
tests plus `13` subtests, including the 287-step oracle, real request contract,
thinking-disabled decoding, cancellation cleanup, and collision-free
trajectory persistence.

The H100 recipe uses eight GPUs:

```text
actor:   fsdp:d4p1t1  -> four actor workers on GPUs 0-3
rollout: vllm:d4p1t1  -> four rollout workers on GPUs 4-7
```

The launcher explicitly exports the pinned
`MAAPACMAN_PACMAN_PYTHON_ROOT`, places `maapacman-rl/bin` first on `PATH`,
refuses busy GPUs, and never preempts unrelated work.

Every rollout has a random `trajectory_sample_id` and is persisted as:

```text
<dataset-row-id>--sample-<trajectory-sample-id>.json
```

Exclusive file creation prevents repeated GRPO samples of one dataset row from
replacing each other. The earlier group-12 run wrote only the last sample for
each row; its two validation files are not a 24-sample mean.

Training and evaluation decoding are independent:

```text
training: sampled, configured temperature/top_p, group size 12
current per-update validation: sampled12, temperature 0.7, top_p 0.95
post-run validation: separately report greedy1 and sampled12 for every checkpoint
```

`PacmanImageOnlyWorkflow` defaults to `enable_thinking=false`, rejects
`enable_thinking=true`, and explicitly sends
`chat_template_kwargs.enable_thinking=false`. Trajectories record the prompt
style, decoding contract, request body, verbatim response, and reasoning
content. Both sampled and greedy validation require thinking to be disabled and
the report rejects any observed reasoning content.

## 10. Corrected group-12 evidence

Historical run:

```text
/home/ubuntu/z00819216/run_artifacts/maapacman-rl/
  level1-group12-20260723b
```

The original post-training test omitted `enable_thinking=false`; in-training
validation also sampled at temperature `1.0`. The corrected evaluator rebuilt
complete VLM checkpoints for base/epoch0/epoch1/epoch2/epoch3/final, ran one
true greedy episode on each, then ran 24 matched sampled episodes on final.
All used seed `0`, 287 steps, original pygame RGB, and thinking disabled.

Evidence:

```text
.../corrected_eval_20260723/comparison.json
```

| Policy | Decode | Score | Pellet clear | Walls | Actions |
|---|---|---:|---:|---:|---|
| base | greedy | 0 | 0.00% | 287 | `U x287` |
| epoch0 | greedy | 20 | 1.02% | 283 | `L x287` |
| epoch1-3 | greedy | 20 | 1.02% | 283 | `L x287` |
| final | greedy | 20 | 1.02% | 283 | `L x287` |
| final | sampled, 24 episodes | 20 avg | 1.02% avg | 283 avg | `L x6888` |

Every corrected episode has zero reasoning turns. The first update caused
single-action collapse and later updates did not recover.

## 11. Static image-only prompt A/B

The A/B compared:

- `minimal_v1`, the original short image-only prompt;
- `live_static_v2`, which borrows only static visual landmarks, absolute screen
  directions, the nearby-pellet objective, and the blue-wall rule.

`live_static_v2` has no coordinates, pellet counts, legal action sets,
OPEN/BLOCKED directions, route hints, cell history, or live-controller action
veto.

Base and collapsed-final policies each ran one greedy plus 12 sampled episodes
for both prompts. Evidence:

```text
.../prompt_ab_20260723b/comparison.json
```

The live-inspired prompt changed base greedy from score `0` to `20`, but the
training-relevant sampled distribution was worse:

| Base sampled policy | Avg score | Pellet clear | Avg walls |
|---|---:|---:|---:|
| `minimal_v1` | 389.17 | 15.65% | 130.50 |
| `live_static_v2` | 357.50 | 14.80% | 151.75 |

Both prompts left the collapsed final checkpoint at score `20` with `L` on
every turn. The next training gate therefore freezes `minimal_v1`; the
live-inspired prompt remains a rejected ablation.

## 12. Anti-collapse gate and conditional next gates

Formal config:

```text
configs/level1_image_anticollapse_4update_group12_8gpu.yaml
```

It restarts from base Qwen3.5-9B with:

```text
max_steps = 287
group size = 12
rollout + actor workers = 4 + 4
reference = BF16, actor-colocated with native FSDP CPU parameter offload
optimizer updates = exactly 4
rollout temperature = 0.7
learning rate = 1.5e-6
KL coefficient = 0.01
actor storage = FP32 master weights
frozen reference storage = BF16, resident
prompt = minimal_v1
validation = greedy temperature 0 after every update
reward = unchanged score_delta - step_penalty - wall_penalty
```

No progress shaping or Oracle action is active in this gate. Acceptance
requires retaining sampled exploration and improving the best corrected greedy
checkpoint without parse failures or reasoning content.

The first `20260723a` launch completed rollout and PPO compute but OOMed during
the first xccl weight sync: the newly enabled reference engine and actor left
only `1.63 GiB`, while the FSDP full-tensor gather required `3.79 GiB`.
`20260723b` tested reference offload, but AReaL's nested `stdbuf` wrappers made
the first TMS preload string invalid. A recipe-owned `sitecustomize.py` shim
fixed that parsing issue. `20260723c` then completed all `48` first-batch
rollouts and the initial reference offload, but TMS failed to restore the
large colocated FSDP reference with `CUDA error: invalid argument`.

TMS is therefore rejected for this production gate. `20260723d` kept the
colocated reference resident in BF16. It completed all `48` rollouts, ref-logp,
PPO, xccl weight synchronization, and wrote a complete `17.9 GB` checkpoint.
The actor plus reference nevertheless peaked at about `79.75 / 81.56 GiB`;
an asynchronous allocation failure surfaced in the checkpoint
`torch.cuda.synchronize()` before validation.

An isolated `20260723e` attempt used `3 rollout + 4 actor + 1 independent
reference`. It proved the physical separation, but this AReaL controller
produced only three of the four dataset groups and waited indefinitely at
`36/48` trajectories: one rollout worker is required per synchronized consumer
item in this configuration.

An attempted `20260723f` topology kept `4 + 4` and targeted the reference at
rollout, but AReaL initializes reference before rollout and rejected the job
with `WorkerNotFoundError`; changing upstream initialization order is out of
scope.

The accepted eight-GPU topology therefore remains `4 rollout + 4 actor`, with
the BF16 reference actor-colocated but configured with native FSDP2
`fsdp.offload_params: true`. It keeps frozen parameter shards on CPU outside
reference forward passes, then streams them to GPU as FSDP layers execute.
This is distinct from AReaL's TMS engine `offload`: global `enable_offload` and
`ref.offload` remain false. The actor retains `optimizer_dtype: float32`; the
reference uses `optimizer_dtype: bfloat16`. The design preserves the required
four synchronized rollout groups and four-way actor sharding while freeing
reference storage during PPO, xccl, and checkpoint synchronization. The TMS
smoke script and shim remain diagnostic evidence only and are not activated by
the formal config.

The accepted topology completed as run:

```text
/home/ubuntu/z00819216/run_artifacts/maapacman-rl/
  level1-anticollapse-4update-20260723h
```

It finished all four optimizer updates, saved four complete `17.9 GB`
checkpoints, and persisted 200 unique trajectories: 192 sampled training
episodes plus eight historical greedy validation episodes. Native FSDP
reference offload reduced static reference GPU use from about `19.9 GiB` to
`2.3 GiB`; PPO later plateaued around `72.22 / 79.19 GiB` without unbounded
growth or OOM.

The corrected matched evaluation nevertheless failed the quality gate:

| Policy | Decode | Avg score | Pellet clear | Avg walls |
|---|---|---:|---:|---:|
| base | greedy1 | 0 | 0.00% | 287.00 |
| update02 | greedy1 | 20 | 1.02% | 283.00 |
| base | sampled12, `0.7/0.95` | 248.33 | 9.99% | 165.42 |
| update02 | sampled12, `0.7/0.95` | 213.33 | 8.97% | 180.42 |

All corrected episodes had thinking disabled, zero reasoning turns, and zero
parse failures. Greedy selected update02 only because it repeated `R` for all
287 actions; under the training-matched sampled distribution, update02 was
worse than base. Sparse training must therefore not be scaled.

The next isolated change is
`alpha * (nearest_normal_pellet_distance_before -
nearest_normal_pellet_distance_after)` with `alpha=1`, using BFS over the
MaaPacman-owned level representation. Distance and reward terms must be logged
and independently audited; `pacman-python` remains unchanged.

That follow-up is implemented by
`configs/level1_image_progress_4update_group12_8gpu.yaml`. It changes only the
auditable alpha-1 reward term and the validation contract:

```text
in-training sampled validation:
  n_samples = 12
  temperature = 0.7
  top_p = 0.95
  enable_thinking = false

post-run checkpoint report:
  sampled12 = the same 0.7 / 0.95 contract
  greedy1 = temperature 0 / top_p 1
  enable_thinking = false for both
```

Sampled validation is the primary checkpoint-selection distribution because it
matches training. Greedy remains a separate collapse/determinism diagnostic;
the report exposes both `best_sampled_label` and `best_greedy_label` rather
than merging the two results.

If that also fails, use Oracle SFT curricula at horizons `64`, `128`, and full
`287`, followed by RL. Scale to 8-12 updates and deploy into the live demo only
after an overfit gate passes.

## 13. Official-main migration gate

On 2026-07-23 both remote nodes were moved to clean official AReaL worktrees at
commit `4d7ee11479d61ebe6c6f020e2bdcda5d76c6a76b`; the former checkouts and all
Morgan/robotics changes remain untouched on `robotics/morgan-vla`.

The node5 eight-GPU diagnostic used
`configs/level1_official_areal_smoke_3b_4gpu.yaml` with Qwen2.5-VL-3B. It
proved that the official scheduler, four vLLM workers, four actor workers, and
the real SDL-dummy `PygamePacmanEnv` screenshot path initialize and generate
valid game trajectories. It then failed before the first optimizer update:

```text
ref.compute_logp
  -> FSDPEngine._prepare_mb_list
  -> KeyError: 'mm_token_type_ids'
```

This is an interface mismatch, not an OOM and not a Pygame failure. The
official OpenAI proxy caches token IDs, log-probs, versions, masks, and rewards,
but its `InteractionWithTokenLogpReward.to_tensor_dict()` does not preserve
the image tensors or Qwen-VL `mm_token_type_ids`. Official Qwen-VL FSDP
correctly requires both. The vLLM `awex_adapter` warning about a missing
Megatron package is optional-plugin noise; all four inference servers became
ready and served the RGB requests.

Do not patch the official AReaL worktree or silently train these samples as
text-only. The production migration must replace the OpenAI-proxy export with
a recipe-owned native multimodal `RolloutWorkflow` that returns the official
tensor contract:

```text
input_ids
attention_mask
loss_mask
logprobs
versions
rewards
mm_token_type_ids
multi_modal_input[pixel_values, image_grid_thw]
```

The recipe now provides
`areal_pacman.workflow.PacmanNativeVisionWorkflow` and the official 3B smoke
config selects it. The workflow calls `InferenceEngine.agenerate()` directly,
requires exact equality between the processor `input_ids` and the rollout
response `input_tokens`, and returns one complete multimodal training row per
Pacman action. It does not modify the official AReaL worktree.

The isolated node5 verification passed the complete recipe suite
(`138 passed, 13 subtests`) and the real cached
`Qwen2.5-VL-3B-Instruct` processor produced:

```text
input_ids          (1, 410)
mm_token_type_ids  (1, 410)
pixel_values       (1344, 1176)
image_grid_thw     (1, 3) = [[1, 32, 42]]
```

Until the revised smoke passes a real reference-logp, actor update,
checkpoint save/reload, and fixed evaluation, Sections 9-12 remain historical
results from the AReaL-VLA-based stack rather than proof of official-main
training compatibility.

## Appendix A. Superseded 2026-07-22 snapshot

The remainder is retained only as historical evidence. Its test counts,
six-GPU topology, filename-overwrite behavior, and early-run conclusions are
superseded by Sections 9-12 above.

The executable level-1 recipe constructs `PygamePacmanEnv` directly, validates
`pacman-python-level1-pygame-v1`, records full environment provenance and
collectible state, and uses the original game's `all_normal_pellets` terminal
reason. Local, node1, and node5 recipe suites pass `99` tests plus `13`
subtests. This includes the 287-step oracle and cancellation while awaiting a
model response.

The production launcher uses a six-GPU `d3` actor plus `d3` rollout topology.
It places the selected Conda environment first on `PATH`; this is required
because AReaL launches nested workers with `python3`. It refuses busy selected
GPUs and never preempts unrelated processes.

Trajectory JSON filenames use the dataset episode ID. Repeated samples of the
same ID replace the earlier file. Therefore the directory is an auditable
final sample set, but its file count is not the total rollout count. Epoch and
global-step counts must be read from AReaL metrics and checkpoint names.

### Historical acceptance gates

### Local gate — completed

- `pacman-python` source checkout is clean at commit `d258122e...`.
- MaaPacman wrapper uses the original pygame Surface.
- Windows native and SDL dummy produce identical reset and `L,L,L,S` hashes.
- `PygamePacmanEnv` tests pass `6/6`, including four concurrent workers.
- Complete MaaPacman unittest suite passes `23/23`.
- Complete areal-pacman pytest suite passes `98` tests and `11` subtests.

### Linux display gate — completed on `h100-node1`

- Actual pygame driver: SDL dummy, pygame `2.6.1`, SDL `2.28.4`.
- Three repeated `L,L,L,S` runs matched Windows raw RGB hashes exactly.
- Four- and sixteen-worker concurrent tests passed with unique runtime IDs.
- Read-only source-resource concurrency passed.
- The 287-step original-game oracle matched Windows state and RGB hash.
- No worker process or temporary worker directory remained.

Xvfb was not installed or used and is not part of the production recipe.

### Recipe CPU gate — completed locally and on both remote nodes

- The node5 mirror and dedicated Conda prefix are now present.
- Direct PygamePacmanEnv tests, RGB parity, 4/16-worker isolation, and the
  complete oracle have passed in that persistent environment.
- Node1 now has the same three-editable-package layout. Its direct pygame gates
  pass, and the deployed recipe suite passes `98` tests and `11` subtests.
- Node1 uses `/mnt/data-node1/z00819216/xinglu/AReaL-VLA` at base
  `da645a37...` with 26 dirty entries; node5 uses its own physical checkout.
  The paths are intentionally node-local rather than shared.
- Both nodes regenerate identical 8-row train and 2-row validation datasets
  with `max_steps=287` and canonical level revision `36116c17...`.
- Both nodes pass AReaL config loading for six GPUs, `vllm:d3p1t1`, and
  `fsdp:d3p1t1`.
- The real model request, canonical parser, reward mapping, trajectory audit,
  and process cleanup have been exercised on node5.

### GPU gate — training passed; greedy improvement failed

Node5 run `impl-20260722b` used GPUs 1–6 and completed in 856.94 seconds.
GPU0 and GPU7 workloads were left untouched. Evidence:

- three vLLM rollout servers loaded Qwen3.5-9B and served real RGB requests;
- each saved trajectory contains 287 original-pygame environment steps;
- three PPO optimizer updates completed, ending at `v_theta=3`;
- checkpoints exist for epoch 0/global step 1 and epoch 1/global step 3;
- all nine final trajectory files pass independent reward recomputation;
- the deployable final checkpoint contains 427 trained keys and 333 restored
  frozen visual keys.

The frozen greedy seed-0 evaluation did **not** improve after this short run:

```text
                 baseline   after 2 epochs
base reward          0.0          0.0
pellet clear rate    0.0          0.0
shaped reward     -287.0       -287.0
parse failures       0            0
```

Thus the recipe is accepted as an executable RL training path, but the
two-epoch overfit-quality criterion remains failed. The next experiment should
first reduce the 287-turn credit horizon or use demonstration-biased initial
actions, then compare several fixed seeds and sampled as well as greedy policy
metrics. Do not claim policy improvement from the current run.
