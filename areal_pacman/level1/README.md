# Production Level-1 package

This package owns the MaaPacman Level-1 reinforcement-learning recipe:

- `workflow.py`: AReaL rollout workflow, no-thinking decoding, and dynamic
  open-action masking.
- `level1_dataset.py`: deterministic rows backed by
  `maapacman.env.PygamePacmanEnv`.
- `prompts.py`: production image/live-state prompt contracts.
- `rewards.py`: Level-1 reward composition and audit.
- `trajectories.py`: Level-1 trajectory validation and summaries.

New code should import from `areal_pacman.level1`. Root modules with the same
names are compatibility shims for existing Python callers and YAML workflow
paths.
