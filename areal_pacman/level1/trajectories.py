"""Auditable trajectory validation, persistence, and summary metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .level1_dataset import SUPPORTED_MAX_STEPS
from .rewards import audit_reward


REQUIRED_ENV_FIELDS = {
    "env_api_version",
    "env_id",
    "backend",
    "pacman_python_revision",
    "level_revision",
    "renderer_revision",
    "seed",
    "max_steps",
    "normal_pellets_initial",
    "normal_pellets_eaten",
    "normal_pellet_clear_rate",
    "final_score",
    "pygame_mode",
    "terminated",
    "truncated",
    "terminal_reason",
}
REQUIRED_STEP_FIELDS = {
    "step",
    "completion",
    "action",
    "parse_failed",
    "base_reward",
    "step_penalty",
    "wall_penalty",
    "shaped_reward",
    "pellet_clear_rate",
    "pellets_remaining",
    "normal_pellets_remaining",
    "normal_pellets_eaten",
    "normal_pellet_clear_rate",
    "power_pellets_remaining",
    "score",
    "pygame_mode",
    "oscillation_return",
    "terminated",
    "truncated",
    "terminal_reason",
    "observation_png_sha256",
}


def audit_trajectory(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_ENV_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"trajectory missing environment fields: {sorted(missing)}")
    if payload["env_api_version"] != "1.0":
        raise ValueError("trajectory env_api_version must be '1.0'")
    if payload["env_id"] != "pacman-python-level1-pygame-v1":
        raise ValueError("trajectory has the wrong environment ID")
    if payload["backend"] != "original-pygame":
        raise ValueError("trajectory backend must be original-pygame")
    if payload["max_steps"] not in SUPPORTED_MAX_STEPS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_MAX_STEPS))
        raise ValueError(f"trajectory max_steps must be one of: {supported}")
    steps = payload.get("trajectory")
    if not isinstance(steps, list) or not steps:
        raise ValueError("trajectory must contain at least one step")
    shaped_total = 0.0
    base_total = 0.0
    for index, step in enumerate(steps, 1):
        if not isinstance(step, Mapping):
            raise ValueError(f"trajectory step {index} must be an object")
        step_missing = REQUIRED_STEP_FIELDS - step.keys()
        if step_missing:
            raise ValueError(
                f"trajectory step {index} missing fields: {sorted(step_missing)}"
            )
        parse_failed = bool(step.get("parse_failed"))
        if parse_failed:
            if step["action"] is not None or not step["terminated"]:
                raise ValueError(
                    "parse-failure step must have action=None and terminated=True"
                )
        else:
            if step["action"] not in {"U", "D", "L", "R", "S"}:
                raise ValueError(f"trajectory step {index} has invalid action")
            audit_reward(step)
        digest = step["observation_png_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"trajectory step {index} has invalid PNG hash")
        if int(step["step"]) != index:
            raise ValueError("trajectory steps must be contiguous and one-based")
        shaped_total += float(step["shaped_reward"])
        base_total += float(step["base_reward"])
    if abs(shaped_total - float(payload["total_shaped_reward"])) > 1e-9:
        raise ValueError("total_shaped_reward does not match trajectory steps")
    if abs(base_total - float(payload["total_base_reward"])) > 1e-9:
        raise ValueError("total_base_reward does not match trajectory steps")
    final = steps[-1]
    for field in ("terminated", "truncated", "terminal_reason", "pygame_mode"):
        if payload[field] != final[field]:
            raise ValueError(f"trajectory payload {field} does not match final step")
    if int(payload["final_score"]) != int(final["score"]):
        raise ValueError("trajectory final_score does not match final step")


def write_trajectory(payload: Mapping[str, Any], directory: Path) -> Path:
    audit_trajectory(payload)
    directory.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(payload["id"])
    )
    sample_id = payload.get("trajectory_sample_id")
    if not sample_id:
        raise ValueError("trajectory_sample_id is required for collision-free persistence")
    safe_sample_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(sample_id)
    )
    output = directory / f"{safe_id}--sample-{safe_sample_id}.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def _max_no_progress_streak(episode: Mapping[str, Any]) -> int:
    longest = 0
    current = 0
    for step in episode.get("trajectory", []):
        if float(step.get("base_reward", 0.0)) > 0.0:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def summarize_episodes(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("at least one episode is required")
    parse_failures = sum(int(item.get("parse_failures", 0)) for item in episodes)
    canonical_violations = sum(
        int(item.get("canonical_action_violations", 0)) for item in episodes
    )
    clear_rates = [float(item["pellet_clear_rate"]) for item in episodes]
    normal_clear_rates = [
        float(item["normal_pellet_clear_rate"]) for item in episodes
    ]
    action_counts = {action: 0 for action in ("U", "D", "L", "R", "S")}
    reasoning_turns = 0
    no_progress_streaks = []
    for item in episodes:
        no_progress_streaks.append(_max_no_progress_streak(item))
        for step in item.get("trajectory", []):
            action = step.get("action")
            if action in action_counts:
                action_counts[str(action)] += 1
            if str(step.get("reasoning_content") or "").strip():
                reasoning_turns += 1
    return {
        "episodes": len(episodes),
        "average_pellet_clear_rate": sum(clear_rates) / len(clear_rates),
        "average_normal_pellet_clear_rate": (
            sum(normal_clear_rates) / len(normal_clear_rates)
        ),
        "average_normal_pellets_eaten": sum(
            int(item["normal_pellets_eaten"]) for item in episodes
        )
        / len(episodes),
        "full_completions": sum(bool(item.get("won")) for item in episodes),
        "parse_failures": parse_failures,
        "canonical_action_violations": canonical_violations,
        "average_episode_length": sum(int(item["steps"]) for item in episodes)
        / len(episodes),
        "average_base_reward": sum(
            float(item["total_base_reward"]) for item in episodes
        )
        / len(episodes),
        "average_shaped_reward": sum(
            float(item["total_shaped_reward"]) for item in episodes
        )
        / len(episodes),
        "average_final_score": sum(int(item["final_score"]) for item in episodes)
        / len(episodes),
        "average_wall_collisions": sum(
            int(item.get("wall_collisions", 0)) for item in episodes
        )
        / len(episodes),
        "average_wall_hit_rate": sum(
            int(item.get("wall_collisions", 0)) / max(int(item["steps"]), 1)
            for item in episodes
        )
        / len(episodes),
        "average_oscillation_returns": sum(
            int(item.get("oscillation_returns", 0)) for item in episodes
        )
        / len(episodes),
        "average_oscillation_rate": sum(
            int(item.get("oscillation_returns", 0)) / max(int(item["steps"]), 1)
            for item in episodes
        )
        / len(episodes),
        "average_normal_pellets_remaining": sum(
            int(item["normal_pellets_remaining"]) for item in episodes
        )
        / len(episodes),
        "average_power_pellets_remaining": sum(
            int(item["power_pellets_remaining"]) for item in episodes
        )
        / len(episodes),
        "action_counts": action_counts,
        "reasoning_turns": reasoning_turns,
        "max_no_progress_streak": max(no_progress_streaks),
        "average_max_no_progress_streak": sum(no_progress_streaks)
        / len(no_progress_streaks),
    }
