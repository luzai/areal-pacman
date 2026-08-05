# Run artifacts

Generated datasets, trajectories, checkpoints, logs, rendered slides, videos,
and presentations are deliberately excluded from Git. They can be large,
machine-specific, or contain thousands of generated files.

## Latest completed production run

Run ID:

```text
openmask16-step32-r12-20260728T160000Z
```

Training contract:

- Qwen3.5-9B
- 8 GPUs
- 8 epochs / 16 optimizer updates
- group size 12
- maximum 32 environment steps per episode
- thinking disabled
- one canonical action token per model turn
- dynamic mask restricted to the environment-reported open actions

The run completed all 16 updates. The final trajectory audit covered 768
training episodes and 24,576 actions, with zero parse failures, zero
open-action-mask violations, and zero wall collisions. These are on-policy
training rollout results, not an independent held-out evaluation of the final
checkpoint.

The originating machine stored the complete run beneath:

```text
/home/ubuntu/z00819216/run_artifacts/maapacman-rl/openmask16-step32-r12-20260728T160000Z
```

That location is operational history, not a portable default. New runs should
set `ARTIFACT_ROOT` to storage appropriate for the target machine.

## What belongs in Git

- Source under `areal_pacman/`
- Tests
- Training and evaluation scripts
- Reproducible YAML configurations
- Small patch files required by documented configurations
- Design and usage documentation

## What stays outside Git

- `artifacts/runs/`
- `artifacts/datasets/`
- `artifacts/reports/`
- `artifacts/workbench/`
- Model checkpoints and `*.safetensors`
- Generated datasets and trajectories
- Logs
- MP4/PPTX deliverables and rendered slide caches
- Python, pytest, packaging, and local virtual-environment caches

For long-term retention, publish large artifacts to an artifact store and
record the immutable run ID, configuration hash, dataset manifest, checkpoint
hashes, and evaluation summary in the release or experiment record.
