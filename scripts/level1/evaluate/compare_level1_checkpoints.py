from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABELS = ("base", "epoch0", "epoch1", "epoch2", "epoch3", "final")
METRICS = (
    "average_final_score",
    "average_pellet_clear_rate",
    "average_wall_collisions",
    "average_episode_length",
    "max_no_progress_streak",
    "average_base_reward",
    "average_shaped_reward",
    "full_completions",
    "reasoning_turns",
    "action_counts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine corrected level-1 checkpoint evaluation summaries."
    )
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    decoding = payload.get("decoding", {})
    if decoding.get("enable_thinking") is not False:
        raise ValueError(f"thinking was not disabled in {path}")
    if int(payload.get("reasoning_turns", -1)) != 0:
        raise ValueError(f"reasoning content was observed in {path}")
    return payload


def compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "episodes": payload["episodes"],
        "decoding": payload["decoding"],
        **{metric: payload[metric] for metric in METRICS},
    }


def main() -> None:
    args = parse_args()
    greedy = {
        label: compact(read_summary(args.eval_root / label / "greedy.json"))
        for label in LABELS
    }
    sampled = compact(
        read_summary(args.eval_root / "final" / "sampled24.json")
    )
    payload = {
        "contract": {
            "environment": "pacman-python-level1-pygame-v1",
            "max_steps": 287,
            "thinking": False,
            "greedy": {
                "episodes_per_checkpoint": 1,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "matched_sampled_final": {
                "episodes": 24,
                "temperature": 1.0,
                "top_p": 0.95,
            },
        },
        "greedy": greedy,
        "final_sampled24": sampled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
