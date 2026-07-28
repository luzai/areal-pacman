# AReaL patches

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

## Historical or diagnostic

- `areal_fsdp_multimodal_device.patch`
- `areal_qwen3p5_weight_sync_name_mapping.patch`
- `areal_vllm_weight_update_name_diagnostic.patch`

Do not apply these by default. They are retained to reproduce earlier
compatibility investigations and should be reviewed against the exact AReaL
revision first.

Always run `git apply --check <patch>` before applying a patch. Do not apply the
same patch twice.
