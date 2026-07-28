# Qwen3.5-9B Action-Only 8192 Parser-Fix Failed Example

Date: 2026-07-09

Setting: [config_text_rollout_qwen3p5_9b_actiononly_medium_overfit_grpo_6epoch_8192_parserfix_8gpu.yaml](../../configs/archive/text/config_text_rollout_qwen3p5_9b_actiononly_medium_overfit_grpo_6epoch_8192_parserfix_8gpu.yaml)

Summary from first partial batch:

- `episodes=14`
- `wins=0`
- `avg_reward=-50.0`
- `parse_failures=14`
- `exact_actions=0`
- terminal reasons: `parse_failed=14`

## Step 0 Observation

```text
Step 0/20
Score: 0
Pellets left: 5
Legal actions: down, right, stay
Grid:
#########
#P..#...#
# # # #G#
# #   # #
#   #   #
#########
Allowed output tokens now: down, right, stay.
Forbidden output tokens now: up, left.
Unsafe immediate ghost actions: none.
If a token is forbidden now, it is a wall move and loses reward.
Output exactly one allowed token from the Allowed output tokens list. No explanation.
```

## Model Output Pattern

The model did eventually choose the right move, but only after a long reasoning preamble:

```text
The user wants me to play a PacMan game.
I need to select one token from the provided `Allowed output tokens` list.
...
The instructions say: "Answer with exactly one token from Allowed output tokens. Do not explain your reasoning."

I will output `right`.
</think>

right
```

## Parser Result

```text
parsed_action: __parse_failed__
parse_failed: True
exact_action: False
legal_action: False
reward: -50
done: True
reason: parse_failed
```

Interpretation: `max_new_tokens=8192` lets the model finish its thought, but the output is not action-only. For strict action-only RL, this is correctly penalized. The next useful fix is output control, not more tokens.
