# Scripts

## Production Level-1

- `level1/train/`: validated training launchers and the earlier overfit
  pipeline.
- `level1/evaluate/`: checkpoint evaluation, prompt A/B, and comparison tools.
- `level1/dataset/`: deterministic dataset and manifest generation.
- `level1/report/`: trajectory auditing, checkpoint assembly, demo export, and
  report helpers.

Primary training entry point:

```bash
bash scripts/level1/train/run_level1_training.sh
```

## Historical synthetic experiments

- `synthetic/`: multi-maze generation, rollout summaries, reports, and legacy
  remote supervisors.

The synthetic scripts are retained for reproducibility and are not the
production MaaPacman Level-1 launcher.
