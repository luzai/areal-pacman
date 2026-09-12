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

## Required for constrained (option-code / action-token) sampling

- `vllm_allowed_token_ids_mask_swap.patch` (applies to vLLM 0.22.1, not AReaL)

`InputBatch.swap_states()` swapped the two `allowed_token_ids_mask_cpu_tensor`
rows with the Python tuple-swap idiom. That idiom is correct for the 1-D numpy
arrays around it, where `arr[i]` yields a scalar copy, but `t[i]` on a 2-D torch
tensor is a **view**: the first assignment overwrites row `i1`, and the second
then reads that already-overwritten row back. Both rows end up holding row
`i2`'s mask and row `i1`'s mask is lost.

The attention backends call `swap_states()` from
`reorder_batch_to_split_decodes_and_prefills()`
(`vllm/v1/attention/backends/utils.py`) to group a batch into
decode / short_extend / long_extend / prefill regions. Any constrained request
that gets reordered therefore samples with another request's constraint, or with
no constraint at all when it was swapped against an unconstrained row. It then
emits a token outside its own `allowed_token_ids`, which this recipe surfaces as
`token_constraints.ObjectiveParseError: output tokens (N,) are not one canonical
objective` and discards as a contract violation (~0.25-2.8% of episodes on an
8-GPU, 12-sample, image-prompt run).

Because reordering needs a batch with mixed request categories, the defect only
shows up under concurrency with long (image) prefills — serial or text-only
traffic does not reproduce it.

The patch also clears the vacated mask row in `condense()`. That row is left
holding an active request's mask today, and `add_request()` only writes the mask
for requests that set `allowed_token_ids`, so a later unconstrained request
landing on that index would inherit a stale restriction.

Reproductions (CPU only, no GPU required):

- `vllm_mask_swap_repro_minimal.py` — the tensor-level defect in isolation.
- `vllm_mask_swap_repro_insitu.py` — drives the real `InputBatch.swap_states()`.

`areal_pacman/level1/workflow.py` additionally retries a decision whose sampled
token falls outside the advertised support (`_MASK_LEAK_RETRY_ATTEMPTS`), which
recovers these episodes even on an unpatched vLLM.

## Conditional

- `areal_fsdp_cpu_offload_empty_cache.patch`

Required only when the selected configuration enables parameter CPU offload.
The launcher checks for it only in that mode.

Always run `git apply --check <patch>` before applying a patch. Do not apply the
same patch twice.
