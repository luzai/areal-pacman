# Configurations

## Canonical Level-1 training

The public training recipe has exactly two full-run configurations:

- `level1/train/level1_curriculum1_step256_100update_group12_8gpu.yaml`
  starts from `Qwen/Qwen3.5-9B` and trains safe pellet collection.
- `level1/train/level1_curriculum2_step256_100update_group12_8gpu.yaml`
  starts from `${CURRICULUM1_CHECKPOINT}` and trains full ghostdoor-v3.

`CURRICULUM1_CHECKPOINT` must be a complete Transformers/vLLM-loadable
checkpoint. Use `scripts/level1/report/build_complete_vlm_checkpoint.py` to
combine the selected Curriculum 1 AReaL saver checkpoint with frozen base-model
weights.

Both configurations use 8 training rows, batch size 4, 50 epochs, 256
environment steps per episode, and 12 complete-game samples per group: 100
optimizer updates total. Each RL action advances up to 16 game-logic frames and
stops early on terminal.

The two-update Curriculum 1 smoke test is a launcher mode, not a third config:

```bash
SMOKE_TEST=1 bash scripts/level1/train/run_level1_training.sh
```

`level1/train/` contains only these two canonical files. `level1/eval/` contains
evaluation-only configurations. Earlier Level-1 gates remain in
`level1/archive/`; machine-specific synthetic/text and early vision configs are
not shipped on this release branch.

Production Level-1 uses the bundled `maapacman.PygamePacmanEnv` plus the paired
`pacman-python` checkout; no separate MaaPacman repository is required.
