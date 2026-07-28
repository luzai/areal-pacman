from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = ("base", "final")
PROMPTS = ("minimal_v1", "live_static_v2")
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
        description="Combine level-1 static image-only prompt A/B results."
    )
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_summary(path: Path, prompt_style: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("image_prompt_style") != prompt_style:
        raise ValueError(f"prompt-style mismatch in {path}")
    if payload.get("decoding", {}).get("enable_thinking") is not False:
        raise ValueError(f"thinking was not disabled in {path}")
    if int(payload.get("reasoning_turns", -1)) != 0:
        raise ValueError(f"reasoning content was observed in {path}")
    return payload


def compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "episodes": payload["episodes"],
        "image_prompt_style": payload["image_prompt_style"],
        "decoding": payload["decoding"],
        **{metric: payload[metric] for metric in METRICS},
    }


def main() -> None:
    args = parse_args()
    results: dict[str, Any] = {}
    for model in MODELS:
        results[model] = {}
        for prompt in PROMPTS:
            directory = args.eval_root / model / prompt
            results[model][prompt] = {
                "greedy": compact(
                    read_summary(directory / "greedy.json", prompt)
                ),
                "sampled12": compact(
                    read_summary(directory / "sampled12.json", prompt)
                ),
            }
    payload = {
        "contract": {
            "environment": "pacman-python-level1-pygame-v1",
            "max_steps": 287,
            "thinking": False,
            "prompts": list(PROMPTS),
            "models": list(MODELS),
            "greedy": {"episodes": 1, "temperature": 0.0, "top_p": 1.0},
            "sampled": {"episodes": 12, "temperature": 1.0, "top_p": 0.95},
        },
        "results": results,
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
