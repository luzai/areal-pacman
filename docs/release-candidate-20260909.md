# Pacman release candidate: integration and acceptance

Status: **local candidate, not a published or training-accepted release**.

## Fixed source set

| Component       | Revision                                   | Role                                                               |
| --------------- | ------------------------------------------ | ------------------------------------------------------------------ |
| AReaL candidate | `9f93d1deb59c1c99cd7019afbd0e83bb40b62fc0` | Combined training/vision fixes and explicit actor CUDA SDPA policy |
| Recipe runtime  | `a54a9d9e7fd1507f56dc8ba2c11c62a5c2d50b06` | Committed C1/C2 recipe; candidate changes are documentation only   |
| pacman-python   | `cbb97115e407abc86a44adc82a1b8f360b3e8da0` | Explicit ghost-mode release interface                              |

The initial candidate `7e19f4ad` tree is `d48de6729fa69bee5b68003dd2c8f3f65942877d`. The
GPU test archive was generated from this exact Git tree, excluding local untracked files
and formatting-only worktree changes. The test-only follow-up candidate tree is
`3924965543da33632c373ae8489c1927b68805ab`; its archive SHA256 is
`96b6005f48f7290a4df94f05dcd80deb37a9e8cb22b0d1556b1817a27cc9df3a`. The follow-up
archive hash also matched locally and remotely. Its `areal/` runtime subtree is
unchanged; only the small GPU test's backend context differs.

Integration keeps the curated release ancestry:

```text
ee872bae  previous release/pacman-v0.1.0
  1b31d73d  Qwen3.5 row isolation
  23659fd8  native action audit and safe offloaded teardown
  98028b2b  multimodal 3D position IDs
  7e19f4ad  BF16 vision adapter and FSDP FP32-angle preservation
  2c2a5caf  test-only: explicit non-cuDNN backend for the tiny vision fixture
  9f93d1de  explicit non-cuDNN SDPA policy for CUDA Qwen3.5 VLM engine processes
```

The vision change applies the eight-file patch from `79698eec` without merging that
branch or its blog history. Crucially, `79698eec` alone is not a replacement for the
training-fix lineage. The game legacy-source branch is also not a compatible replacement
for the pinned game release.

## Configuration boundary

Keep the checked-in C1/C2 opt-in worker extension and explicit actor/reference
`fsdp.memory_efficient_load=false`. C1 gradient checkpointing is false, C2 is true;
phase offload stays enabled and parameter CPU offload stays disabled. Historical C1
GC-OFF results are not acceptance of this newly integrated tree.

This candidate deliberately excludes uncommitted concurrent work: changes to the
recipe's evaluation schedule and the separate RAM-result-lifetime fixes. Do not copy
those working trees wholesale into a release or a running job. Review and test them
separately if they become necessary for the full smoke.

## Acceptance gates

| Gate                                                 | Status                                                                                 |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Curated release ancestry and combined fix coverage   | Verified locally                                                                       |
| Scoped lint, license checks and commit hooks         | Passed                                                                                 |
| Repository-wide pre-commit                           | Not fully green: Windows GBK output, CRLF shell script, missing mdformat dependency    |
| Core CPU regressions                                 | 136 passed, including real two-rank Gloo row guards                                    |
| Combined regressions at 2c2a5caf                     | 152 passed, 9 warnings                                                                 |
| Combined regressions at 9f93d1de                     | 159 passed, 9 warnings in 109.26 seconds                                               |
| Whole vision embedding comparison                    | Passed: BF16 five real frames plus 5/12/48-image batches, max absolute difference zero |
| Fresh adapter worker                                 | Passed; correct adapter status, single/5/12/48 requests, exit 0 and orderly shutdown   |
| Entire real rollout batch actor/vLLM log-prob audit  | Pending                                                                                |
| Multi-GPU C1 updates and checkpoint reload/rollout   | Blocked by GPU availability                                                            |
| C1 checkpoint to C2 multi-GPU smoke                  | Pending after C1                                                                       |
| Paired source publication and immutable version pins | Held until required gates pass                                                         |

The new test run is under
`H100_2_2:/home/h100-repro/codex-tests/pacman-release-integration-20260909`. It uses the
existing pinned environment, not a reinstall into an active job. Tests check the actual
imported AReaL path and require the Qwen3.5 training helpers to exist. Each phase
produces its own log and exit marker.

Archive SHA256: `4faebba067e9c371b8742532b227c3ddfe7f6658a3b809fcde5d51ce7a15e28d`.
Local and remote archive hashes matched.

The original combined `unit.log`/`unit.exit` are retained (exit 139). A fresh process
stopped at the first failing case, without a segmentation fault:
`test_real_vision_fsdp_bf16_forward_backward[False-False]` failed with
`cuDNN Frontend error: No valid execution plans built` (2 passed, 1 failed).
Fresh-process controls compare the vision suite without a full engine import and with
engine import plus cuDNN SDPA disabled. These are diagnostic controls, not changes to
the production config or a waiver of the failed default path.

Both with and without engine import, the small unmodified vision fixture fails cuDNN
plan construction. Engine import did not change the four SDPA backend enable flags.
Disabling only cuDNN SDPA passed all 16 vision tests. Installed vLLM also explicitly
disables cuDNN SDPA in its CUDA platform module. Commit `2c2a5caf` therefore bounds
backend selection to the tiny fixture's forward/backward context and restores flags
afterward. No production source, dtype, error tolerance or training config changed from
`7e19f4ad`. The default production model path still needs the full distributed gates.

A real-sized HF vision control with cuDNN SDPA enabled also failed plan construction
(`actor-default-sdpa.log`, exit 1). Therefore the test-only fix was not sufficient to
close the runtime risk. Candidate `9f93d1de` explicitly disables cuDNN SDPA in dense
Qwen3.5 VLM CUDA engine processes before model loading, matching vLLM's existing
platform policy. It logs the choice and does not change other SDPA enable flags or BF16
dtypes. This process-wide setting can also affect decoder SDPA; it is not a GDN or
complete language-parity fix. Seven tests cover applicability, idempotence and
preservation of other flags. A full combined regression and real vision comparison
invoking this exact helper both completed successfully on GPU 7. The combined suite
passed 159 tests. The helper-based real vision comparison passed all five frames and
5/12/48-image batches with zero maximum absolute embedding difference. This is a
standalone vision check, not the still-required full FSDP training initialization,
optimizer update, weight synchronization or whole-language audit.

All owned diagnostic runners have exited. GPU 7 was verified at 4 MiB, 0% utilization,
with no compute clients. Original failed-control logs remain available; failures were
not erased or reported as passing. No existing training or retrieval job was killed.

Latest candidate tree: `6b3ce1b3d2e4fac1a148350ceb1796f6ca9d84b5`. Its uploaded archive
SHA256 matched locally/remotely:
`79ded1f37223b797aa714f3171d07ac8a3eab3b1eb7453b955863f4173043387`.

At launch, H100_2_1 had no free GPU. On H100_2_2, GPUs 0-6 belonged to existing rollout
workers; only GPU 7 was free. No existing job was killed or switched. Single-device
vision tests cannot substitute for multi-GPU optimizer updates, weight synchronization,
or checkpoint reload tests.

All eight configured H100 aliases were checked again after the user requested other
servers. H100_2_3 has six free devices (2-7); devices 0/1 belong to existing retrieval
services and were not touched. H100_2_2 has only device 7 free. H100_1_1, H100_1_2,
H100_2_1, H100_2_4, H100_2_5 and H100_2_6 have no free devices. Low utilization with
resident jobs is not availability. The user explicitly said not to kill existing jobs.
No six-device smoke was started: it would not replace the original eight-device
acceptance test (actor/reference on four devices, rollout on four). No monitoring
automation was created in this side conversation.

## Finish publication

1. Recheck source/remote heads, active jobs and free GPUs. Do not stop existing jobs
   without explicit authorization for the exact processes.
1. Run the full-batch audit on the combined candidate and report the denominator,
   out-of-bound fraction, probability-ratio distribution and input/weight identity. Do
   not relax the threshold to obtain a pass or call remaining differences zero.
1. Run the distributed smoke with the actual intended parallelism and configs. Verify
   update markers, losses, memory peaks, orderly exit and GPU cleanup.
1. Export a complete smoke checkpoint and actually load it for image rollout; then
   validate C2 initialization from that real C1 checkpoint.
1. Freeze the tested recipe/runtime manifest, correct the candidate warnings in README,
   and push AReaL to `release/pacman-v0.1.0` by normal fast-forward only. Publish
   matching recipe pins and verify all remote SHAs. Do not force-update existing tags or
   publish a final-agent claim without its separate evidence.

The full 100-update curricula, final checkpoint distribution and download-based agent
acceptance remain separate from this source integration task. No new long-running
training budget or weight-hosting destination is assumed here.
