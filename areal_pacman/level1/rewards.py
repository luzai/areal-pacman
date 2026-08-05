"""Auditable Level-1 reward shaping over explicit task events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardConfig:
    step_penalty: float = 1.0
    wall_penalty: float = 1.0
    use_base_reward: bool = True
    normal_pellet_reward: float = 0.0
    power_pellet_reward: float = 0.0
    completion_reward: float = 0.0
    nearest_pellet_alpha: float = 0.0
    nearest_pellet_remaining_ratio_threshold: float = 1.0
    nearest_pellet_scale_by_cleared_ratio: bool = False
    nearest_pellet_skip_on_eat: bool = False

    def __post_init__(self) -> None:
        if self.step_penalty < 0:
            raise ValueError("step_penalty must be non-negative")
        if self.wall_penalty < 0:
            raise ValueError("wall_penalty must be non-negative")
        if self.normal_pellet_reward < 0:
            raise ValueError("normal_pellet_reward must be non-negative")
        if self.power_pellet_reward < 0:
            raise ValueError("power_pellet_reward must be non-negative")
        if self.completion_reward < 0:
            raise ValueError("completion_reward must be non-negative")
        if self.nearest_pellet_alpha < 0:
            raise ValueError("nearest_pellet_alpha must be non-negative")
        if not 0 <= self.nearest_pellet_remaining_ratio_threshold <= 1:
            raise ValueError(
                "nearest_pellet_remaining_ratio_threshold must be in [0, 1]"
            )


@dataclass(frozen=True)
class RewardBreakdown:
    base_reward: float
    base_reward_contribution: float
    normal_pellet_eaten: bool
    normal_pellet_reward: float
    power_pellet_eaten: bool
    power_pellet_reward: float
    level_completed: bool
    completion_reward: float
    step_penalty: float
    wall_penalty: float
    normal_pellet_remaining_ratio: float
    nearest_pellet_shaping_active: bool
    nearest_pellet_distance_before: int | None
    nearest_pellet_distance_after: int | None
    nearest_pellet_progress_weight: float
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
    normal_pellet_remaining_ratio: float = 1.0,
    nearest_pellet_distance_before: int | None = None,
    nearest_pellet_distance_after: int | None = None,
) -> RewardBreakdown:
    del previous_info
    base_reward_contribution = (
        float(base_reward) if config.use_base_reward else 0.0
    )
    normal_pellet_eaten = bool(info.get("pellet_eaten", False))
    normal_pellet_reward = (
        config.normal_pellet_reward if normal_pellet_eaten else 0.0
    )
    power_pellet_eaten = bool(info.get("power_pellet_eaten", False))
    power_pellet_reward = (
        config.power_pellet_reward if power_pellet_eaten else 0.0
    )
    level_completed = (
        str(info.get("terminal_reason")) == "all_normal_pellets"
        and int(info.get("normal_pellets_remaining", -1)) == 0
    )
    completion_reward = (
        config.completion_reward if level_completed else 0.0
    )
    step_penalty = config.step_penalty
    wall_penalty = config.wall_penalty if bool(info["wall_collision"]) else 0.0
    if not 0 <= normal_pellet_remaining_ratio <= 1:
        raise ValueError("normal_pellet_remaining_ratio must be in [0, 1]")
    distances_available = (
        nearest_pellet_distance_before is not None
        and nearest_pellet_distance_after is not None
    )
    shaping_active = (
        config.nearest_pellet_alpha > 0
        and normal_pellet_remaining_ratio
        <= config.nearest_pellet_remaining_ratio_threshold
        and distances_available
        and not (
            config.nearest_pellet_skip_on_eat
            and normal_pellet_eaten
        )
    )
    progress_weight = 0.0
    progress_reward = 0.0
    if shaping_active and (
        nearest_pellet_distance_before is not None
        and nearest_pellet_distance_after is not None
    ):
        progress_weight = config.nearest_pellet_alpha
        if config.nearest_pellet_scale_by_cleared_ratio:
            progress_weight *= 1.0 - normal_pellet_remaining_ratio
        progress_reward = progress_weight * (
            nearest_pellet_distance_before - nearest_pellet_distance_after
        )
    shaped = (
        base_reward_contribution
        + normal_pellet_reward
        + power_pellet_reward
        + completion_reward
        - step_penalty
        - wall_penalty
        + progress_reward
    )
    return RewardBreakdown(
        base_reward=float(base_reward),
        base_reward_contribution=base_reward_contribution,
        normal_pellet_eaten=normal_pellet_eaten,
        normal_pellet_reward=normal_pellet_reward,
        power_pellet_eaten=power_pellet_eaten,
        power_pellet_reward=power_pellet_reward,
        level_completed=level_completed,
        completion_reward=completion_reward,
        step_penalty=step_penalty,
        wall_penalty=wall_penalty,
        normal_pellet_remaining_ratio=float(
            normal_pellet_remaining_ratio
        ),
        nearest_pellet_shaping_active=shaping_active,
        nearest_pellet_distance_before=nearest_pellet_distance_before,
        nearest_pellet_distance_after=nearest_pellet_distance_after,
        nearest_pellet_progress_weight=progress_weight,
        nearest_pellet_progress_reward=progress_reward,
        shaped_reward=shaped,
    )


def audit_reward(record: Mapping[str, Any], *, tolerance: float = 1e-9) -> None:
    progress_weight = float(record.get("nearest_pellet_progress_weight", 0.0))
    progress_reward = float(record.get("nearest_pellet_progress_reward", 0.0))
    distance_before = record.get("nearest_pellet_distance_before")
    distance_after = record.get("nearest_pellet_distance_after")
    if bool(record.get("nearest_pellet_shaping_active", False)):
        if distance_before is None or distance_after is None:
            raise ValueError("active nearest-pellet shaping requires both distances")
        expected_progress = progress_weight * (
            int(distance_before) - int(distance_after)
        )
        if abs(expected_progress - progress_reward) > tolerance:
            raise ValueError(
                "nearest-pellet reward audit failed: "
                f"expected {expected_progress}, got {progress_reward}"
            )
    elif abs(progress_reward) > tolerance or abs(progress_weight) > tolerance:
        raise ValueError("inactive nearest-pellet shaping must have zero reward and weight")
    expected = (
        float(record.get("base_reward_contribution", record["base_reward"]))
        + float(record.get("normal_pellet_reward", 0.0))
        + float(record.get("power_pellet_reward", 0.0))
        + float(record.get("completion_reward", 0.0))
        - float(record["step_penalty"])
        - float(record["wall_penalty"])
        + progress_reward
    )
    if abs(expected - float(record["shaped_reward"])) > tolerance:
        raise ValueError(
            f"reward audit failed: expected {expected}, got {record['shaped_reward']}"
        )
