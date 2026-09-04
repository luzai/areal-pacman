# Configurations

## Canonical Level-1 training

The public training recipe has exactly two canonical full-run configurations:

- `level1/train/level1_curriculum1_step256_100update_group12_8gpu.yaml`
  starts from `Qwen/Qwen3.5-9B` and trains the safe pellet-collection mode.
- `level1/train/level1_curriculum2_step256_100update_group12_8gpu.yaml`
  starts from `${CURRICULUM1_CHECKPOINT}` and trains the full ghostdoor-v3 mode.
  This variable must point to a complete Transformers/vLLM-loadable checkpoint;
  use `scripts/level1/report/build_complete_vlm_checkpoint.py` to combine the
  selected Curriculum 1 AReaL saver checkpoint with frozen base-model weights.

Both use 8 training rows, batch size 4, 50 epochs, 256 environment steps per
episode, and 12 complete-game samples per group: 100 optimizer updates in total.
Each RL action advances up to 16 game-logic frames, stopping early on terminal.

The two-update smoke test is a launcher mode, not a third configuration:

```bash
SMOKE_TEST=1 bash scripts/level1/train/run_level1_training.sh
```

`level1/train/` contains only these two canonical configurations.
`level1/eval/` contains evaluation-only configurations. Earlier runs and probes
live under `level1/archive/`, `archive/text/`, and `archive/vision/` and are not
the production default.

Production Level-1 uses the bundled `maapacman.PygamePacmanEnv` plus the paired
`pacman-python` checkout; no separate MaaPacman repository is required.
