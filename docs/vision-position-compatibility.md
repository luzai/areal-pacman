# Experimental Qwen3.5 BF16 vision backbone compatibility

Disabled by default in AReaL; the current development C1/C2 YAMLs explicitly opt in.
This is not a final training release: the complete rollout audit and performance
comparison remain required. No checkpoint weights are rewritten.

The companion FSDP change in `areal/engine/fsdp_utils/__init__.py` preserves FP32
sin/cos at Qwen3.5 vision-block boundaries (`cast_forward_inputs=False` for those blocks
only). Parameter and gradient-reduction dtypes remain configured as before; the root and
other model blocks retain their existing policies. Deploy both sides together: the
rollout adapter alone does not match the old actor's rounded angles.

The opt-in AReaL worker extension selects a dense Qwen3.5 model adapter before model
loading/profile. Model weights and vision outputs stay BF16. It aligns:

- Patch embedding: Conv3D instead of vLLM's GEMM shortcut.
- Position interpolation: the installed Transformers reference and operation order.
- Vision RoPE: independent HF-compatible FP32 frequencies and sin/cos; rotate Q/K in
  FP32, then return BF16. No injected HF activations are required.
- Encoder attention: TORCH_SDPA, selected by the opt-in model adapter when unset; a
  conflicting explicit backend is rejected. This adapter does not patch decoder kernels.

The integrated candidate additionally makes the FSDP worker's backend policy explicit:
before loading a dense Qwen3.5 VLM on CUDA, it disables cuDNN SDPA in that engine
process, matching vLLM's CUDA-platform policy. This is process-wide and can affect
decoder SDPA too; it does not change BF16, other SDPA enable flags, or GDN kernels.
Other model families, non-vision engines and non-CUDA device types are not changed. The
initialization log records the policy. This avoids depending on incidental vLLM imports
and addresses the observed cuDNN execution-plan failure in real-sized vision.

Exact whole-model parity is not promised. This is not the earlier fix to 3D position
IDs. Use matching Transformers versions and HF SDPA on the actor side. The frequency
table follows HF's normal GPU initialization. Keep actor
`fsdp.memory_efficient_load: false` for this validated path (CPU parameter offload is a
separate setting and is supported). CPU-initialized model loading is not yet covered:
CPU/GPU FP32 frequency construction is not necessarily bitwise identical.

The development C1/C2 YAMLs set this existing field (use the same setting in a separate
diagnostic configuration):

```yaml
vllm:
  worker_extension_cls: areal.engine.vllm_ext.qwen35_torch_vision_worker.Qwen35TorchVisionWorkerExtension
```

For rollback use the default
`areal.engine.vllm_ext.vllm_worker_extension.VLLMWorkerExtension` and a fresh server
process. For a full rollback also restore the previous FSDP source revision. Do not
switch implementations inside a running training service.

Keep matching Transformers versions on actor and rollout hosts. Startup logs record the
reference implementation hash and Transformers version; record these alongside the
source revision. No interpolation output is cached, so updated weights remain live. Do
not enable solely because a handful of cases pass the 7% audit threshold.

Required validation: exact operator parity across image grids, live weight-update
behavior, real worker startup, representative/full-batch action-logprob audit,
single/mixed/chunked requests, and warmed rollout latency/throughput. Compare the
default implementation and adapter with identical settings and inputs. A local FP32
whole-model conversion is not included in this switch. Keep the regular BF16 training
configuration. This adapter is initially validated with eager TP1; other parallelism and
encoder CUDA-graph modes need explicit acceptance tests.

## Historical pre-push verification (20260909 artifacts)

The submitted implementation at AReaL `79698eecf91bf50ede480b94380d962f835bc6d8` was
checked on an isolated H800 GPU, with PyTorch 2.11.0+cu130, Transformers 5.7.0 and vLLM
0.22.1:

- 16 vision tests passed, including real BF16 FSDP forward/backward with gradient
  checkpointing on/off and CPU parameter offload on/off.
- Five real game images and batches of 5/12/48 images matched the GPU-initialized HF
  vision reference exactly (maximum absolute embedding difference zero).
- A fresh multiprocess worker reported `Qwen35TorchVisionModel`, FP32 rotary arithmetic,
  Conv3D patch embedding and TORCH_SDPA. Single requests and 5/12/48-request batches
  completed with image caches cleared between calls.
- The recipe regression suite passed 562 tests, with one Windows-only unsupported
  AF_UNIX diagnostic skipped on Linux. This used the pinned compatible game
  `cbb97115e407abc86a44adc82a1b8f360b3e8da0`, not the legacy source branch.
- The legacy game source branch separately passed its 19 rule/event tests; it still
  lacks the recipe's explicit ghost-mode interface.

These historical tests verify the vision path, not the complete training-fix
integration. The development commit above lacks the `98028b2b` row-isolation and
3D-position-ID fixes. The combined release candidate is
`9f93d1deb59c1c99cd7019afbd0e83bb40b62fc0`; see the
[candidate acceptance record](release-candidate-20260909.md). Neither the historical
vision tests nor their 562 recipe regressions prove a complete C1/C2 training rerun or
whole-batch language log-prob parity. Language-side mismatch remains under
investigation. The scoped AReaL commit hooks passed; the separate repository-wide
pre-commit attempt was not fully green on Windows because of unrelated formatting and
platform/tooling failures.
