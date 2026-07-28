# Configurations

## Production Level-1

- `level1/train/`: active training configurations.
  - `level1_live_state_step32_16update_group12_8gpu.yaml`: completed short
    training run.
  - `level1_live_state_step32_200update_group12_8gpu.yaml`: next long-run
    configuration.
- `level1/eval/`: evaluation-only configurations.
- `level1/archive/`: earlier Level-1 gates and smoke configurations retained
  for reproducibility.

## Historical experiments

- `archive/text/`: synthetic/text Pacman experiments.
- `archive/vision/`: earlier synthetic multi-maze and vision experiments.

Production Level-1 uses MaaPacman's `PygamePacmanEnv`. Configurations under
`archive/` are not the default production path.
