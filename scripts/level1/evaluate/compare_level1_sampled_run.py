from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare base and level-1 checkpoints under one uniform sampled "
            "train/validation/test contract."
        )
    )
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    return parser.parse_args()


def _read(path: Path, episodes: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decoding = payload.get("decoding", {})
    expected = {
        "temperature": 0.7,
        "top_p": 0.95,
        "enable_thinking": False,
    }
    for key, value in expected.items():
        if decoding.get(key) != value:
            raise ValueError(
                f"{path} has {key}={decoding.get(key)!r}; expected {value!r}"
            )
    if int(payload.get("episodes", -1)) != episodes:
        raise ValueError(f"{path} does not contain {episodes} episodes")
    if payload.get("image_prompt_style") != "live_state_v3":
        raise ValueError(f"{path} does not use live_state_v3")
    if payload.get("observation_contract") != (
        "screenshot_plus_live_state_and_navigation_history"
    ):
        raise ValueError(f"{path} has the wrong observation contract")
    reward = payload.get("reward_contract", {})
    if float(reward.get("nearest_pellet_alpha", -1.0)) != 0.0:
        raise ValueError(f"{path} uses nearest-pellet distance shaping")
    if int(payload.get("reasoning_turns", -1)) != 0:
        raise ValueError(f"{path} contains reasoning content")
    if int(payload.get("parse_failures", -1)) != 0:
        raise ValueError(f"{path} contains parse failures")
    return payload


def _episode_values(
    payload: dict[str, Any], field: str
) -> list[float]:
    return [
        float(episode[field]) for episode in payload["episode_results"]
    ]


def _bootstrap_delta_ci(
    before: list[float],
    after: list[float],
    *,
    iterations: int = 10_000,
) -> list[float]:
    generator = random.Random(1)
    deltas = []
    for _ in range(iterations):
        before_mean = sum(generator.choice(before) for _ in before) / len(before)
        after_mean = sum(generator.choice(after) for _ in after) / len(after)
        deltas.append(after_mean - before_mean)
    deltas.sort()
    return [
        deltas[int(0.025 * iterations)],
        deltas[int(0.975 * iterations) - 1],
    ]


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "episodes",
        "average_base_reward",
        "average_shaped_reward",
        "average_normal_pellets_eaten",
        "average_normal_pellet_clear_rate",
        "full_completions",
        "average_final_score",
        "average_wall_collisions",
        "average_wall_hit_rate",
        "average_oscillation_returns",
        "average_oscillation_rate",
        "average_episode_length",
        "parse_failures",
        "reasoning_turns",
        "action_counts",
    )
    return {field: payload[field] for field in fields}


def _selection_key(payload: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(payload["full_completions"]),
        float(payload["average_normal_pellet_clear_rate"]),
        float(payload["average_shaped_reward"]),
        -float(payload["average_wall_hit_rate"]),
        -float(payload["average_oscillation_rate"]),
    )


def _delta(
    base: dict[str, Any],
    trained: dict[str, Any],
    field: str,
) -> float:
    return float(trained[field]) - float(base[field])


def compare(eval_root: Path, episodes: int) -> dict[str, Any]:
    paths = sorted(eval_root.glob("*/sampled12.json"))
    results = {path.parent.name: _read(path, episodes) for path in paths}
    if "base" not in results:
        raise ValueError("base/sampled12.json is required")
    trained_labels = sorted(
        (label for label in results if label.startswith("update")),
        key=lambda value: int(value.removeprefix("update")),
    )
    if len(trained_labels) != 4:
        raise ValueError(
            f"expected four update checkpoints, found {trained_labels}"
        )
    base = results["base"]
    best_label = max(
        trained_labels,
        key=lambda label: _selection_key(results[label]),
    )
    comparisons: dict[str, Any] = {}
    base_clear = _episode_values(base, "normal_pellet_clear_rate")
    for label in trained_labels:
        trained = results[label]
        clear_delta = _delta(
            base, trained, "average_normal_pellet_clear_rate"
        )
        pellet_delta = _delta(
            base, trained, "average_normal_pellets_eaten"
        )
        reward_delta = _delta(base, trained, "average_shaped_reward")
        completion_delta = _delta(base, trained, "full_completions")
        wall_rate_delta = _delta(base, trained, "average_wall_hit_rate")
        comparisons[label] = {
            "delta_average_normal_pellet_clear_rate": clear_delta,
            "delta_average_normal_pellets_eaten": pellet_delta,
            "delta_average_shaped_reward": reward_delta,
            "delta_full_completions": completion_delta,
            "delta_average_wall_hit_rate": wall_rate_delta,
            "delta_average_oscillation_rate": _delta(
                base, trained, "average_oscillation_rate"
            ),
            "normal_pellet_clear_rate_delta_bootstrap_95_ci": (
                _bootstrap_delta_ci(
                    base_clear,
                    _episode_values(
                        trained, "normal_pellet_clear_rate"
                    ),
                )
            ),
            "overfit_pass": (
                clear_delta >= 0.10
                and pellet_delta > 0.0
                and (reward_delta > 0.0 or completion_delta > 0.0)
                and wall_rate_delta <= 0.05
            ),
        }
    return {
        "contract": {
            "environment": "pacman-python-level1-pygame-v1",
            "max_steps": 287,
            "observation": (
                "screenshot + Pacman position + legal/open actions + blocked "
                "actions + per-cell exit history + anti-oscillation history"
            ),
            "prompt_style": "live_state_v3",
            "temperature": 0.7,
            "top_p": 0.95,
            "enable_thinking": False,
            "episodes_per_model": episodes,
            "nearest_pellet_alpha": 0.0,
        },
        "acceptance": {
            "normal_pellet_clear_rate_improvement": ">= 0.10",
            "normal_pellets_eaten_improvement": "> 0",
            "reward_or_completion_improvement": "> 0",
            "wall_hit_rate_regression": "<= 0.05",
            "parse_failures": 0,
            "reasoning_turns": 0,
        },
        "base": _compact(base),
        "checkpoints": {
            label: _compact(results[label]) for label in trained_labels
        },
        "comparisons_to_base": comparisons,
        "best_checkpoint": best_label,
        "overfit_pass": comparisons[best_label]["overfit_pass"],
    }


def main() -> None:
    args = parse_args()
    result = compare(args.eval_root, args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
