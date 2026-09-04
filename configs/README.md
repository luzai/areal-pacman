# Configurations

## Production Level-1

- `level1/train/`: active training configurations.
  - `level1_live_state_step32_16update_group12_8gpu.yaml`: completed short
    training run.
  - `level1_live_state_step32_200update_group12_8gpu.yaml`: next long-run
    configuration.
  - `level1_live_state_step256_100update_group12_8gpu.yaml`: 256-step,
    50-epoch configuration (100 optimizer updates) with explicit
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
  - `level1_edward_step512_2update_group12_8gpu.yaml`: formal Edward-v3
    Step512 gate with 2 optimizer updates, 48 rollout samples per update, and
    one-step recovery checkpoints. It uses unnormalized per-option returns, but
    weights each option token by `1 / episode_option_count`: loss is averaged
    within each complete game first, then across all 48 games. Reward and
    advantage group normalization remain disabled, and
    `actor.ppo_n_minibatches=1` keeps the 48-game average in one optimizer
    update. Fruit events remain auditable, but direct fruit training reward is
    disabled with `fruit_reward: 0.0` and `use_base_reward: false`. The v3
    recipe also applies `safety_refusal_penalty: 25.0` when Edward cannot
    advertise a provably safe next option after a completed option. Its
    executed-step cost is a fixed `0.05`. A
    parse/canonical contract violation fails closed at whole-episode return
    `-1.0`, independently of the reward accumulated before the violation.
- `level1/eval/`: evaluation-only configurations.
- `level1/archive/`: earlier Level-1 gates and smoke configurations retained
  for reproducibility.

## Historical experiments

- `archive/text/`: synthetic/text Pacman experiments.
- `archive/vision/`: earlier synthetic multi-maze and vision experiments.

Production Level-1 uses the bundled `maapacman.PygamePacmanEnv`; it does not
require a separate MaaPacman checkout. Configurations under `archive/` are not
the default production path.
