from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
        description=(
            "Build separate matched-sampled and greedy level-1 validation "
            "reports for one run."
        )
    )
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampled-filename", default="sampled12.json")
    parser.add_argument("--require-complete-dual", action="store_true")
    return parser.parse_args()


def read_summary(
    path: Path,
    *,
    expected_episodes: int,
    expected_temperature: float,
    expected_top_p: float,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decoding = payload.get("decoding", {})
    if decoding.get("enable_thinking") is not False:
        raise ValueError(f"thinking was not disabled in {path}")
    if int(payload.get("reasoning_turns", -1)) != 0:
        raise ValueError(f"reasoning content was observed in {path}")
    if int(payload.get("episodes", -1)) != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episodes in {path}, got "
            f"{payload.get('episodes')}"
        )
    if float(decoding.get("temperature", -1.0)) != expected_temperature:
        raise ValueError(
            f"unexpected validation temperature in {path}: "
            f"{decoding.get('temperature')}"
        )
    if float(decoding.get("top_p", -1.0)) != expected_top_p:
        raise ValueError(
            f"unexpected validation top_p in {path}: "
            f"{decoding.get('top_p')}"
        )
    return payload


def compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "episodes": payload["episodes"],
        "decoding": payload["decoding"],
        **{metric: payload[metric] for metric in METRICS},
    }


def selection_key(payload: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(payload["full_completions"]),
        float(payload["average_pellet_clear_rate"]),
        float(payload["average_final_score"]),
        -float(payload["average_wall_collisions"]),
        -float(payload["max_no_progress_streak"]),
    )


def compare(
    eval_root: Path,
    sampled_filename: str,
    *,
    require_complete_dual: bool = False,
) -> dict[str, Any]:
    greedy_paths = sorted(eval_root.glob("*/greedy.json"))
    if not greedy_paths:
        raise ValueError(f"no greedy summaries found under {eval_root}")
    greedy = {
        path.parent.name: compact(
            read_summary(
                path,
                expected_episodes=1,
                expected_temperature=0.0,
                expected_top_p=1.0,
            )
        )
        for path in greedy_paths
    }
    best_greedy_label = max(
        greedy, key=lambda label: selection_key(greedy[label])
    )
    sampled12_by_label = {}
    for label in greedy:
        sampled_path = eval_root / label / sampled_filename
        if sampled_path.is_file():
            sampled12_by_label[label] = compact(
                read_summary(
                    sampled_path,
                    expected_episodes=12,
                    expected_temperature=0.7,
                    expected_top_p=0.95,
                )
            )
    missing_sampled_labels = [
        label for label in greedy if label not in sampled12_by_label
    ]
    if require_complete_dual and missing_sampled_labels:
        raise ValueError(
            "missing sampled validation for labels: "
            + ", ".join(missing_sampled_labels)
        )
    best_sampled_label = (
        max(
            sampled12_by_label,
            key=lambda label: selection_key(sampled12_by_label[label]),
        )
        if sampled12_by_label
        else None
    )
    best_label = best_sampled_label or best_greedy_label
    best_sampled = (
        sampled12_by_label[best_sampled_label]
        if best_sampled_label is not None
        else None
    )
    checkpoint_sources: dict[str, Any] = {"base": None}
    for label in greedy:
        manifest = (
            eval_root
            / "complete_checkpoints"
            / label
            / "merge_manifest.json"
        )
        if manifest.is_file():
            checkpoint_sources[label] = json.loads(
                manifest.read_text(encoding="utf-8")
            ).get("trained_checkpoint")
    return {
        "contract": {
            "environment": "pacman-python-level1-pygame-v1",
            "max_steps": 287,
            "prompt": "minimal_v1",
            "thinking": False,
            "greedy": {
                "episodes_per_checkpoint": 1,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "matched_sampled": {
                "episodes": 12,
                "temperature": 0.7,
                "top_p": 0.95,
                "labels": list(greedy),
            },
            "primary_checkpoint_selection": "matched_sampled",
            "greedy_is_separately_reported": True,
            "selection_order": [
                "full_completions",
                "average_pellet_clear_rate",
                "average_final_score",
                "fewer_average_wall_collisions",
                "shorter_max_no_progress_streak",
            ],
        },
        "checkpoint_sources": checkpoint_sources,
        "greedy": greedy,
        "best_label": best_label,
        "best_sampled_label": best_sampled_label,
        "best_greedy_label": best_greedy_label,
        "sampled12_by_label": sampled12_by_label,
        "missing_sampled_labels": missing_sampled_labels,
        "base_sampled12": sampled12_by_label.get("base"),
        "best_sampled12": best_sampled,
    }


def main() -> None:
    args = parse_args()
    payload = compare(
        args.eval_root,
        args.sampled_filename,
        require_complete_dual=args.require_complete_dual,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
