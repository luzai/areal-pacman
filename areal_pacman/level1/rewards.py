"""Minimal training reward over MaaPacman's original score delta."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardConfig:
    step_penalty: float = 1.0
    wall_penalty: float = 1.0
    nearest_pellet_alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.step_penalty < 0:
            raise ValueError("step_penalty must be non-negative")
        if self.wall_penalty < 0:
            raise ValueError("wall_penalty must be non-negative")
        if self.nearest_pellet_alpha < 0:
            raise ValueError("nearest_pellet_alpha must be non-negative")


@dataclass(frozen=True)
class RewardBreakdown:
    base_reward: float
    step_penalty: float
    wall_penalty: float
    nearest_pellet_distance_before: int | None
    nearest_pellet_distance_after: int | None
    nearest_pellet_progress_reward: float
    shaped_reward: float

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def shape_reward(
    base_reward: float,
    previous_info: Mapping[str, Any],
    info: Mapping[str, Any],
    config: RewardConfig,
    *,
    nearest_pellet_distance_before: int | None = None,
    nearest_pellet_distance_after: int | None = None,
) -> RewardBreakdown:
    del previous_info
    step_penalty = config.step_penalty
    wall_penalty = config.wall_penalty if bool(info["wall_collision"]) else 0.0
    if config.nearest_pellet_alpha > 0 and (
        nearest_pellet_distance_before is None
        or nearest_pellet_distance_after is None
    ):
        raise ValueError(
            "nearest-pellet shaping requires both before/after distances"
        )
    progress_reward = 0.0
    if (
        nearest_pellet_distance_before is not None
        and nearest_pellet_distance_after is not None
    ):
        progress_reward = config.nearest_pellet_alpha * (
            nearest_pellet_distance_before - nearest_pellet_distance_after
        )
    shaped = (
        float(base_reward)
        - step_penalty
        - wall_penalty
        + progress_reward
    )
    return RewardBreakdown(
        base_reward=float(base_reward),
        step_penalty=step_penalty,
        wall_penalty=wall_penalty,
        nearest_pellet_distance_before=nearest_pellet_distance_before,
        nearest_pellet_distance_after=nearest_pellet_distance_after,
        nearest_pellet_progress_reward=progress_reward,
        shaped_reward=shaped,
    )


def audit_reward(record: Mapping[str, Any], *, tolerance: float = 1e-9) -> None:
    expected = (
        float(record["base_reward"])
        - float(record["step_penalty"])
        - float(record["wall_penalty"])
        + float(record.get("nearest_pellet_progress_reward", 0.0))
    )
    if abs(expected - float(record["shaped_reward"])) > tolerance:
        raise ValueError(
            f"reward audit failed: expected {expected}, got {record['shaped_reward']}"
        )
