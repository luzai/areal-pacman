# AReaL patches

## Required for the two-stage recipe

The pinned AReaL revision includes lossless JSON RPC serialization of non-finite
float bounds, required to preserve C1 `reward_clip: .inf` instead of converting it
to `null`. This does not permit non-finite task rewards. The release remains
source/recipe-only; this runtime fix does not establish a passing GPU smoke test.

It also counts the actual sequence rows inside grouped PPO batches when
computing the synchronized padding target. C1 keeps multi-row prompt groups;
their container count is not the worker batch size. C2 singleton row dispatch
retains its existing target. CPU regressions cover both representations and
metadata-only RTensors; distributed training still requires the GPU gates.

## Required for dynamic open-action-mask training

- `areal_pacman_action_logprobs.patch`

This patch adds `vLLMConfig.logprobs_mode` and applies the same Pacman action
mask and sampling temperature when actor/reference log-probabilities are
recomputed. Configurations with `open_action_mask: true` require it. The
Level-1 launcher checks for its marker before training starts.

## Conditional

- `areal_fsdp_cpu_offload_empty_cache.patch`

Required only when the selected configuration enables parameter CPU offload.
The launcher checks for it only in that mode.

Always run `git apply --check <patch>` before applying a patch. Do not apply the
same patch twice.
