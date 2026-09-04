# Configurations

## Production Level-1

- `level1/train/`: active training configurations.
  - `curriculum1.yaml`: complete 256-step Curriculum 1 initialized from
    Qwen3.5-9B. It is a 50-epoch configuration (100 optimizer updates) with
    the production Level-1 environment, reward, and whole-episode GRPO
    contract.
  - `curriculum2.yaml`: complete Curriculum 2 initialized from the model path
    in `CURRICULUM1_CHECKPOINT`. It intentionally starts a new optimizer and
    scheduler lineage while keeping the same formal training contract.

    Both recipes are 256-step,
    50-epoch configurations (100 optimizer updates) with explicit
    pellet/completion rewards, late-game BFS nearest-pellet guidance, and
    whole-episode GRPO over 12 complete games from the same initial state. Each
    episode first averages its valid objective-token loss, then the 12 episode
    losses are averaged with equal weight. The contract requires
    `actor.ppo_n_minibatches=1` so one optimizer update sees the complete set.
    Fruit events remain auditable, but direct fruit training reward is disabled
    with `fruit_reward: 0.0` and `use_base_reward: false`. A post-option Edward
    safety refusal is recorded explicitly and subtracts
    `safety_refusal_penalty: 25.0` from that completed option and the episode
    return. The executed-step cost is fixed at `0.05`; it no longer increases
    with pellet-clear progress. A parse/canonical contract violation fails closed with
    `contract_violation_return: -1.0`; the terminal audit record stores the
    exact adjustment needed to make the whole-episode return equal `-1.0`.
  - Smoke testing reuses `curriculum1.yaml` with `--smoke-updates 2`; it is not
    a third formal configuration.
- `level1/eval/`: evaluation-only configurations.
- `level1/archive/`: earlier Level-1 gates, the former standalone smoke gate,
  and the log-prob probe retained for reproducibility.

## Historical experiments

Machine-specific synthetic/text and early vision experiment configurations are
not shipped on this release branch. They are not part of the Level-1
reproduction path.

Production Level-1 uses the bundled `maapacman.PygamePacmanEnv`; it does not
require a separate MaaPacman checkout. Configurations under `level1/archive/`
are not the default production path.
