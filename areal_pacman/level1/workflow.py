"""Production image-only AReaL workflow backed by MaaPacman's public API."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from copy import deepcopy
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from areal.api import RolloutWorkflow
from maapacman.env import (
    Action,
    PacmanEnvSpec,
    Position,
    PygamePacmanEnv,
    PygamePacmanEnvConfig,
    load_bundled_level,
    nearest_reachable_distance,
)

from ..actions import ActionParseError, parse_action
from .level1_dataset import SUPPORTED_MAX_STEPS, validate_episode_row
from .prompts import (
    build_image_messages,
    crop_pacman_local_view,
    encode_png,
    image_count,
    png_sha256,
    prompt_text,
)
from .rewards import RewardBreakdown, RewardConfig, shape_reward
from .trajectories import audit_trajectory, write_trajectory


EXPECTED_ACTIONS = ("U", "D", "L", "R", "S")
MOVEMENT_ACTIONS = ("U", "D", "L", "R")
ACTION_MASK_BIT = {
    action: 1 << index for index, action in enumerate(MOVEMENT_ACTIONS)
}
OPPOSITE_ACTION = {"U": "D", "D": "U", "L": "R", "R": "L"}
LOGGER = logging.getLogger(__name__)


def _nearest_reachable_distance_with_diagnostics(
    level: Any,
    start: Position,
    targets: set[Position],
    *,
    phase: str,
    previous_position: tuple[int, int] | None = None,
    action: str | None = None,
    live_legal_actions: list[str] | None = None,
) -> int:
    if not targets:
        return 0
    try:
        return nearest_reachable_distance(level, start, targets)
    except ValueError as exc:
        if str(exc) != "no target is reachable from the requested position":
            raise
        diagnostic = {
            "phase": phase,
            "start_position": [start.row, start.col],
            "start_in_bounds": bool(level.in_bounds(start)),
            "start_tile": (
                int(level.tile_at(start)) if level.in_bounds(start) else None
            ),
            "previous_position": (
                list(previous_position)
                if previous_position is not None
                else None
            ),
            "action": action,
            "live_legal_actions": list(live_legal_actions or []),
            "remaining_normal_pellet_count": len(targets),
            "remaining_normal_pellet_positions": [
                [position.row, position.col]
                for position in sorted(
                    targets,
                    key=lambda position: (position.row, position.col),
                )
            ],
            "level_height": int(level.height),
            "level_width": int(level.width),
            "level_revision": str(level.revision),
        }
        encoded = json.dumps(diagnostic, sort_keys=True)
        LOGGER.error("nearest-pellet BFS unreachable: %s", encoded)
        raise RuntimeError(
            f"nearest-pellet BFS unreachable: {encoded}"
        ) from exc


def install_vllm_allowed_token_ids_adapter() -> None:
    """Forward recipe-scoped chat and token constraints to vLLM."""
    from areal.engine.vllm_remote import VLLMBackend

    if getattr(VLLMBackend, "_pacman_allowed_token_ids_adapter", False):
        return
    original = VLLMBackend.build_generation_request

    def build_generation_request(
        self: Any,
        req: Any,
        with_lora: bool,
        version: int,
    ) -> Any:
        http_request = original(self, req, with_lora, version)
        metadata = req.metadata or {}
        allowed = metadata.get("allowed_token_ids")
        if allowed:
            http_request.payload["allowed_token_ids"] = [
                int(token_id) for token_id in allowed
            ]
        chat_template_kwargs = metadata.get("chat_template_kwargs")
        if chat_template_kwargs is not None:
            http_request.payload["chat_template_kwargs"] = dict(
                chat_template_kwargs
            )
        return http_request

    VLLMBackend.build_generation_request = build_generation_request
    VLLMBackend._pacman_allowed_token_ids_adapter = True


def preferred_open_actions(
    open_actions: list[str],
    tried_actions: list[str],
    last_action: str | None,
) -> list[str]:
    """Match the live demo's untried/least-used and anti-reverse preference."""
    counts = {action: 0 for action in open_actions}
    for action in tried_actions:
        if action in counts:
            counts[action] += 1
    untried = [action for action in open_actions if counts[action] == 0]
    if untried:
        preferred = untried
    elif open_actions:
        least = min(counts.values())
        preferred = [
            action for action in open_actions if counts[action] == least
        ]
    else:
        preferred = []
    reverse = OPPOSITE_ACTION.get(last_action or "")
    without_reverse = [action for action in preferred if action != reverse]
    return without_reverse or preferred


@dataclass(frozen=True)
class ModelTurn:
    completion: str
    completion_id: str | None
    messages: list[dict[str, Any]]
    reasoning_content: str | None = None
    raw_response: dict[str, Any] | None = None
    request_extra_body: dict[str, Any] | None = None


def validate_env_spec(
    spec: PacmanEnvSpec,
    requested: Mapping[str, Any],
    *,
    pacman_python_revision: str,
) -> None:
    if spec.api_version != "1.0" or requested.get("api_version") != spec.api_version:
        raise RuntimeError("unsupported MaaPacman environment API")
    if spec.env_id != "pacman-python-level1-pygame-v1":
        raise RuntimeError("unsupported MaaPacman environment ID")
    if requested.get("name") != spec.env_id:
        raise RuntimeError("dataset environment ID does not match MaaPacman")
    if requested.get("backend") != "original-pygame":
        raise RuntimeError("production recipe requires original-pygame backend")
    if requested.get("pacman_python_revision") != pacman_python_revision:
        raise RuntimeError("pacman-python revision does not match dataset row")
    if spec.action_tokens != EXPECTED_ACTIONS:
        raise RuntimeError("incompatible MaaPacman action contract")
    if requested.get("level_revision") != spec.level_revision:
        raise RuntimeError("MaaPacman level revision does not match dataset row")
    if spec.observation_dtype != "uint8" or len(spec.observation_shape) != 3:
        raise RuntimeError("incompatible MaaPacman RGB observation contract")


class PacmanImageOnlyWorkflow:
    """One dataset row to one complete level-1 image-only rollout."""

    def __init__(
        self,
        *,
        env_factory: Callable[[PygamePacmanEnvConfig], PygamePacmanEnv] = (
            PygamePacmanEnv
        ),
        **workflow_kwargs: Any,
    ) -> None:
        self.env_factory = env_factory
        self.workflow_kwargs = workflow_kwargs
        self.last_episode: dict[str, Any] | None = None
        self.action_token_id_by_action: dict[str, int] = {}
        if workflow_kwargs.get("open_action_mask"):
            tokenizer_path = workflow_kwargs.get("tokenizer_path")
            if not tokenizer_path:
                raise ValueError(
                    "open action mask requires tokenizer_path"
                )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            for token in MOVEMENT_ACTIONS:
                token_ids = tokenizer.encode(
                    token, add_special_tokens=False
                )
                if len(token_ids) != 1:
                    raise ValueError(
                        f"action {token!r} must map to one tokenizer ID"
                    )
                self.action_token_id_by_action[token] = int(token_ids[0])

    async def run(self, data: Mapping[str, Any], **extra_kwargs: Any) -> Any:
        options = {**self.workflow_kwargs, **extra_kwargs}
        if options.get("enable_thinking") is None:
            options["enable_thinking"] = False
        if options["enable_thinking"] is not False:
            raise ValueError(
                "production image-only Pacman requires enable_thinking=false"
            )
        validate_episode_row(data)
        requested = data["env"]
        seed = int(requested["seed"])
        state_prefix_actions = list(data.get("state_prefix_actions") or [])
        single_step = data.get("decision_steps") == 1
        options["single_step"] = single_step
        trajectory_sample_id = str(
            options.get("trajectory_sample_id") or uuid.uuid4().hex
        )
        config = PygamePacmanEnvConfig(
            pacman_python_root=options.get("pacman_python_root"),
            level=int(requested["level"]),
            max_steps=int(requested["max_steps"]),
            video_driver=options.get("video_driver", "dummy"),
            audio_driver=options.get("audio_driver", "dummy"),
            worker_base_dir=options.get("worker_base_dir"),
        )
        reward_config = RewardConfig(
            step_penalty=float(options.get("step_penalty", 1.0)),
            wall_penalty=float(options.get("wall_penalty", 1.0)),
            use_base_reward=bool(options.get("use_base_reward", True)),
            normal_pellet_reward=float(
                options.get("normal_pellet_reward", 0.0)
            ),
            power_pellet_reward=float(
                options.get("power_pellet_reward", 0.0)
            ),
            completion_reward=float(options.get("completion_reward", 0.0)),
            nearest_pellet_alpha=float(
                options.get("nearest_pellet_alpha", 0.0)
            ),
            nearest_pellet_remaining_ratio_threshold=float(
                options.get(
                    "nearest_pellet_remaining_ratio_threshold",
                    1.0,
                )
            ),
            nearest_pellet_skip_on_eat=bool(
                options.get("nearest_pellet_skip_on_eat", False)
            ),
        )
        parse_failure_penalty = float(options.get("parse_failure_penalty", -50.0))
        image_prompt_style = str(
            options.get("image_prompt_style", "minimal_v1")
        )
        system_prompt, user_instruction = prompt_text(image_prompt_style)
        scripted = list(options.get("scripted_actions") or [])
        scripted_ids = list(options.get("scripted_completion_ids") or [])
        trajectory: list[dict[str, Any]] = []
        rewards_by_completion: dict[str, float] = {}
        parse_failures = 0
        canonical_violations = 0
        cell_exit_history: dict[tuple[int, int], list[str]] = {}
        recent_actions: list[str] = []
        recent_positions: list[tuple[int, int]] = []
        wall_clock_limit_seconds = options.get("wall_clock_limit_seconds")
        if wall_clock_limit_seconds is not None:
            wall_clock_limit_seconds = float(wall_clock_limit_seconds)
            if wall_clock_limit_seconds <= 0:
                raise ValueError("wall_clock_limit_seconds must be positive")
        stuck_no_progress_steps = options.get("stuck_no_progress_steps")
        if stuck_no_progress_steps is not None:
            stuck_no_progress_steps = int(stuck_no_progress_steps)
            if stuck_no_progress_steps <= 0:
                raise ValueError("stuck_no_progress_steps must be positive")
        no_progress_steps = 0
        episode_started_at = time.monotonic()
        level = (
            load_bundled_level(config.level)
            if reward_config.nearest_pellet_alpha > 0
            else None
        )
        remaining_normal_pellets = (
            set(level.pellets) if level is not None else None
        )

        with self.env_factory(config) as env:
            validate_env_spec(
                env.spec,
                requested,
                pacman_python_revision=env.pacman_python_revision,
            )
            if config.max_steps not in SUPPORTED_MAX_STEPS:
                supported = ", ".join(
                    str(value) for value in sorted(SUPPORTED_MAX_STEPS)
                )
                raise RuntimeError(
                    f"level-1 recipe requires max_steps in: {supported}"
                )
            image, previous_info = env.reset(seed=seed)
            for prefix_token in state_prefix_actions:
                image, _, terminated, truncated, previous_info = env.step(
                    Action(prefix_token)
                )
                if previous_info["wall_collision"]:
                    raise RuntimeError(
                        "state_prefix_actions must be collision-free"
                    )
                if terminated or truncated:
                    raise RuntimeError(
                        "state_prefix_actions reached a terminal state"
                    )
            initial_normal_pellets = int(
                previous_info["normal_pellets_remaining"]
            )
            if initial_normal_pellets <= 0:
                raise RuntimeError("level-1 must start with normal pellets")
            recent_positions.append(
                (
                    int(previous_info["pacman_position"][0]),
                    int(previous_info["pacman_position"][1]),
                )
            )
            if level is not None:
                live_state = env.snapshot()
                if (
                    level.width != int(live_state["width"])
                    or level.height != int(live_state["height"])
                ):
                    raise RuntimeError(
                        "nearest-pellet topology dimensions do not match "
                        "the live environment"
                    )
            if (
                remaining_normal_pellets is not None
                and len(remaining_normal_pellets)
                != int(previous_info["normal_pellets_remaining"])
            ):
                raise RuntimeError(
                    "nearest-pellet tracker does not match reset state"
                )
            final_info = previous_info
            while True:
                model_image = (
                    crop_pacman_local_view(image)
                    if image_prompt_style
                    in ("wall_avoidance_local_v2", "wall_avoidance_axis_v3")
                    else image
                )
                png = encode_png(model_image)
                live_snapshot = env.snapshot()
                current_open_actions = [
                    action
                    for action in MOVEMENT_ACTIONS
                    if action in set(live_snapshot.get("open") or [])
                ]
                if not current_open_actions:
                    raise RuntimeError(
                        "live environment reported no open movement actions"
                    )
                state_context = None
                if image_prompt_style == "live_state_v3":
                    position = (
                        int(previous_info["pacman_position"][0]),
                        int(previous_info["pacman_position"][1]),
                    )
                    open_actions = current_open_actions
                    blocked_actions = [
                        action
                        for action in MOVEMENT_ACTIONS
                        if action in set(live_snapshot.get("blocked") or [])
                    ]
                    if set(open_actions) | set(blocked_actions) != set(
                        MOVEMENT_ACTIONS
                    ):
                        blocked_actions = [
                            action
                            for action in MOVEMENT_ACTIONS
                            if action not in set(open_actions)
                        ]
                    current_history_full = list(
                        cell_exit_history.get(position, [])
                    )
                    current_history = list(
                        dict.fromkeys(current_history_full)
                    )
                    current_counts = {
                        action: current_history_full.count(action)
                        for action in MOVEMENT_ACTIONS
                        if action in current_history_full
                    }
                    last_action = (
                        recent_actions[-1] if recent_actions else None
                    )
                    state_context = {
                        "pacman_position": list(position),
                        "facing": str(live_snapshot.get("facing") or "S"),
                        "pellets_remaining": int(
                            previous_info["pellets_remaining"]
                        ),
                        "open_actions": open_actions,
                        "legal_actions": [
                            action.value for action in env.legal_actions()
                        ],
                        "blocked_actions": blocked_actions,
                        "current_cell_exit_history": current_history,
                        "current_cell_exit_counts": current_counts,
                        "last_action": last_action,
                        "immediate_reverse_action": OPPOSITE_ACTION.get(
                            last_action or ""
                        ),
                        "recent_actions": recent_actions[-8:],
                        "recent_positions": [
                            list(item) for item in recent_positions[-8:]
                        ],
                        "preferred_open_actions": preferred_open_actions(
                            open_actions,
                            current_history_full,
                            last_action,
                        ),
                    }
                messages = build_image_messages(
                    png,
                    prompt_style=image_prompt_style,
                    state_context=state_context,
                )
                model_user_instruction = str(
                    messages[1]["content"][0]["text"]
                )
                if image_count(messages) != 1:
                    raise RuntimeError("model request must contain exactly one image")
                if scripted:
                    completion = str(scripted.pop(0))
                    completion_id = (
                        str(scripted_ids.pop(0)) if scripted_ids else None
                    )
                    turn = ModelTurn(completion, completion_id, messages)
                else:
                    turn = await self._call_model(
                        messages,
                        **options,
                        current_open_actions=current_open_actions,
                    )

                try:
                    action = parse_action(turn.completion)
                except ActionParseError:
                    parse_failures += 1
                    canonical_violations += 1
                    record = {
                        "step": len(trajectory) + 1,
                        "env_step": int(previous_info["step"]) + 1,
                        "completion_id": turn.completion_id,
                        "completion": turn.completion,
                        "reasoning_content": turn.reasoning_content,
                        "raw_model_response": turn.raw_response,
                        "request_extra_body": turn.request_extra_body,
                        "model_user_instruction": model_user_instruction,
                        "observation_context": state_context,
                        "open_action_mask": current_open_actions,
                        "action": None,
                        "parse_failed": True,
                        "base_reward": 0.0,
                        "base_reward_contribution": 0.0,
                        "normal_pellet_eaten": False,
                        "normal_pellet_reward": 0.0,
                        "power_pellet_eaten": False,
                        "power_pellet_reward": 0.0,
                        "level_completed": False,
                        "completion_reward": 0.0,
                        "step_penalty": 0.0,
                        "wall_penalty": 0.0,
                        "normal_pellet_remaining_ratio": (
                            int(previous_info["normal_pellets_remaining"])
                            / initial_normal_pellets
                        ),
                        "nearest_pellet_shaping_active": False,
                        "nearest_pellet_distance_before": None,
                        "nearest_pellet_distance_after": None,
                        "nearest_pellet_progress_reward": 0.0,
                        "shaped_reward": parse_failure_penalty,
                        "pellet_clear_rate": float(previous_info["pellet_clear_rate"]),
                        "pellets_remaining": int(previous_info["pellets_remaining"]),
                        "normal_pellets_remaining": int(
                            previous_info["normal_pellets_remaining"]
                        ),
                        "normal_pellets_eaten": (
                            initial_normal_pellets
                            - int(previous_info["normal_pellets_remaining"])
                        ),
                        "normal_pellet_clear_rate": (
                            initial_normal_pellets
                            - int(previous_info["normal_pellets_remaining"])
                        )
                        / initial_normal_pellets,
                        "power_pellets_remaining": int(
                            previous_info["power_pellets_remaining"]
                        ),
                        "score": int(previous_info["score"]),
                        "pygame_mode": int(previous_info["pygame_mode"]),
                        "wall_collision": False,
                        "oscillation_return": False,
                        "terminated": True,
                        "truncated": False,
                        "terminal_reason": "parse_failed",
                        "observation_png_sha256": png_sha256(png),
                    }
                    trajectory.append(record)
                    if turn.completion_id:
                        rewards_by_completion[turn.completion_id] = parse_failure_penalty
                    break

                distance_before = None
                if level is not None and remaining_normal_pellets is not None:
                    # A step can consume at most one normal pellet.  Avoid
                    # consulting the hidden BFS topology while its reward term
                    # cannot activate; in skip-on-eat mode even a threshold-
                    # crossing pellet step does not need distances.
                    earliest_ratio_after_step = (
                        len(remaining_normal_pellets)
                        if reward_config.nearest_pellet_skip_on_eat
                        else max(0, len(remaining_normal_pellets) - 1)
                    ) / initial_normal_pellets
                    if (
                        earliest_ratio_after_step
                        <= reward_config.nearest_pellet_remaining_ratio_threshold
                    ):
                        before_row, before_col = previous_info[
                            "pacman_position"
                        ]
                        distance_before = (
                            _nearest_reachable_distance_with_diagnostics(
                                level,
                                Position(int(before_row), int(before_col)),
                                remaining_normal_pellets,
                                phase="before_step",
                                action=action.value,
                                live_legal_actions=current_open_actions,
                            )
                        )
                source_position = (
                    int(previous_info["pacman_position"][0]),
                    int(previous_info["pacman_position"][1]),
                )
                if (
                    options.get("open_action_mask")
                    and action.value not in current_open_actions
                ):
                    raise RuntimeError(
                        "model backend violated open action mask: "
                        f"{action.value} not in {current_open_actions}"
                    )
                previous_action = (
                    recent_actions[-1] if recent_actions else None
                )
                next_image, base_reward, terminated, truncated, info = env.step(action)
                info = dict(info)
                score_progress = int(info["score"]) != int(previous_info["score"])
                pellet_progress = int(info["normal_pellets_remaining"]) != int(
                    previous_info["normal_pellets_remaining"]
                )
                no_progress_steps = (
                    0 if score_progress or pellet_progress else no_progress_steps + 1
                )
                safety_reason = None
                if not (terminated or truncated):
                    if (
                        wall_clock_limit_seconds is not None
                        and time.monotonic() - episode_started_at
                        >= wall_clock_limit_seconds
                    ):
                        safety_reason = "safety_timeout"
                    elif (
                        stuck_no_progress_steps is not None
                        and no_progress_steps >= stuck_no_progress_steps
                    ):
                        safety_reason = "stuck"
                elif (
                    truncated
                    and int(info["step"]) >= config.max_steps
                    and config.max_steps == 2000
                ):
                    safety_reason = "safety_step_limit"
                if safety_reason is not None:
                    info["terminal_reason"] = safety_reason
                    terminated = False
                    truncated = True
                if single_step and not (terminated or truncated):
                    info["terminal_reason"] = "single_step_complete"
                    terminated = True
                next_position = (
                    int(info["pacman_position"][0]),
                    int(info["pacman_position"][1]),
                )
                oscillation_return = (
                    len(recent_positions) >= 2
                    and next_position == recent_positions[-2]
                    and action.value
                    == OPPOSITE_ACTION.get(previous_action or "")
                )
                normal_pellet_remaining_ratio = (
                    int(info["normal_pellets_remaining"])
                    / initial_normal_pellets
                )
                distance_after = None
                if level is not None and remaining_normal_pellets is not None:
                    after_row, after_col = info["pacman_position"]
                    after_position = Position(int(after_row), int(after_col))
                    if bool(info["pellet_eaten"]):
                        if after_position not in remaining_normal_pellets:
                            raise RuntimeError(
                                "live game ate a normal pellet outside the tracker"
                            )
                        remaining_normal_pellets.remove(after_position)
                    if len(remaining_normal_pellets) != int(
                        info["normal_pellets_remaining"]
                    ):
                        raise RuntimeError(
                            "nearest-pellet tracker diverged from the live game"
                        )
                    if (
                        distance_before is not None
                        and normal_pellet_remaining_ratio
                        <= reward_config.nearest_pellet_remaining_ratio_threshold
                        and not (
                            reward_config.nearest_pellet_skip_on_eat
                            and bool(info["pellet_eaten"])
                        )
                    ):
                        distance_after = (
                            _nearest_reachable_distance_with_diagnostics(
                                level,
                                after_position,
                                remaining_normal_pellets,
                                phase="after_step",
                                previous_position=source_position,
                                action=action.value,
                                live_legal_actions=list(info["legal_actions"]),
                            )
                        )
                reward = shape_reward(
                    base_reward,
                    previous_info,
                    info,
                    reward_config,
                    normal_pellet_remaining_ratio=(
                        normal_pellet_remaining_ratio
                    ),
                    nearest_pellet_distance_before=distance_before,
                    nearest_pellet_distance_after=distance_after,
                )
                if single_step:
                    avoided_wall = (
                        action.value in MOVEMENT_ACTIONS
                        and not bool(info["wall_collision"])
                    )
                    immediate_reward = 1.0 if avoided_wall else -1.0
                    reward = RewardBreakdown(
                        base_reward=immediate_reward,
                        base_reward_contribution=immediate_reward,
                        normal_pellet_eaten=False,
                        normal_pellet_reward=0.0,
                        power_pellet_eaten=False,
                        power_pellet_reward=0.0,
                        level_completed=False,
                        completion_reward=0.0,
                        step_penalty=0.0,
                        wall_penalty=0.0,
                        normal_pellet_remaining_ratio=(
                            normal_pellet_remaining_ratio
                        ),
                        nearest_pellet_shaping_active=False,
                        nearest_pellet_distance_before=None,
                        nearest_pellet_distance_after=None,
                        nearest_pellet_progress_reward=0.0,
                        shaped_reward=immediate_reward,
                    )
                record = {
                    "step": len(trajectory) + 1,
                    "env_step": int(info["step"]),
                    "completion_id": turn.completion_id,
                    "completion": turn.completion,
                    "reasoning_content": turn.reasoning_content,
                    "raw_model_response": turn.raw_response,
                    "request_extra_body": turn.request_extra_body,
                    "model_user_instruction": model_user_instruction,
                    "observation_context": state_context,
                    "open_action_mask": current_open_actions,
                    "action": action.value,
                    "parse_failed": False,
                    **reward.as_dict(),
                    "pellet_clear_rate": float(info["pellet_clear_rate"]),
                    "pellets_remaining": int(info["pellets_remaining"]),
                    "normal_pellets_remaining": int(
                        info["normal_pellets_remaining"]
                    ),
                    "normal_pellets_eaten": (
                        initial_normal_pellets
                        - int(info["normal_pellets_remaining"])
                    ),
                    "normal_pellet_clear_rate": (
                        initial_normal_pellets
                        - int(info["normal_pellets_remaining"])
                    )
                    / initial_normal_pellets,
                    "power_pellets_remaining": int(info["power_pellets_remaining"]),
                    "score": int(info["score"]),
                    "pygame_mode": int(info["pygame_mode"]),
                    "wall_collision": bool(info["wall_collision"]),
                    "oscillation_return": oscillation_return,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "terminal_reason": info["terminal_reason"],
                    "observation_png_sha256": png_sha256(png),
                }
                trajectory.append(record)
                if turn.completion_id:
                    rewards_by_completion[turn.completion_id] = reward.shaped_reward
                if action.value in MOVEMENT_ACTIONS:
                    cell_exit_history.setdefault(source_position, []).append(
                        action.value
                    )
                recent_actions.append(action.value)
                recent_positions.append(next_position)
                image, previous_info, final_info = next_image, info, info
                if terminated or truncated:
                    break
                if not scripted and options.get("scripted_actions") is not None:
                    raise RuntimeError("scripted action sequence ended before the episode")

            total_base = sum(float(step["base_reward"]) for step in trajectory)
            total_shaped = sum(float(step["shaped_reward"]) for step in trajectory)
            terminal_reason = str(trajectory[-1]["terminal_reason"])
            payload = {
                "id": str(data["id"]),
                "trajectory_sample_id": trajectory_sample_id,
                "split": str(data["split"]),
                "env_api_version": env.spec.api_version,
                "env_id": env.spec.env_id,
                "backend": "original-pygame",
                "pacman_python_revision": env.pacman_python_revision,
                "level_revision": env.spec.level_revision,
                "renderer_revision": env.spec.renderer_revision,
                "level": config.level,
                "seed": seed,
                "max_steps": config.max_steps,
                "state_prefix_actions": state_prefix_actions,
                "decision_steps": 1 if single_step else None,
                "image_prompt_style": image_prompt_style,
                "system_prompt": system_prompt,
                "user_instruction": user_instruction,
                "observation_contract": (
                    "screenshot_plus_live_state_and_navigation_history"
                    if image_prompt_style == "live_state_v3"
                    else "screenshot_only"
                ),
                "action_constraint": (
                    "dynamic_open_movement_actions"
                    if options.get("open_action_mask")
                    else list(EXPECTED_ACTIONS)
                ),
                "decoding": {
                    "temperature": float(options.get("temperature", 0.0)),
                    "top_p": float(options.get("top_p", 1.0)),
                    "max_completion_tokens": int(
                        options.get("max_completion_tokens", 3)
                    ),
                    "enable_thinking": False,
                    "open_action_mask": bool(
                        options.get("open_action_mask", False)
                    ),
                },
                "step_penalty_coefficient": reward_config.step_penalty,
                "wall_penalty_coefficient": reward_config.wall_penalty,
                "use_base_reward": reward_config.use_base_reward,
                "normal_pellet_reward_coefficient": (
                    reward_config.normal_pellet_reward
                ),
                "power_pellet_reward_coefficient": (
                    reward_config.power_pellet_reward
                ),
                "completion_reward_coefficient": (
                    reward_config.completion_reward
                ),
                "nearest_pellet_alpha": reward_config.nearest_pellet_alpha,
                "nearest_pellet_remaining_ratio_threshold": (
                    reward_config.nearest_pellet_remaining_ratio_threshold
                ),
                "nearest_pellet_skip_on_eat": (
                    reward_config.nearest_pellet_skip_on_eat
                ),
                "nearest_pellet_topology_revision": (
                    level.revision if level is not None else None
                ),
                "total_base_reward": total_base,
                "total_shaped_reward": total_shaped,
                "steps": len(trajectory),
                "pellet_clear_rate": float(final_info["pellet_clear_rate"]),
                "normal_pellets_initial": initial_normal_pellets,
                "normal_pellets_eaten": (
                    initial_normal_pellets
                    - int(final_info["normal_pellets_remaining"])
                ),
                "normal_pellet_clear_rate": (
                    initial_normal_pellets
                    - int(final_info["normal_pellets_remaining"])
                )
                / initial_normal_pellets,
                "normal_pellets_remaining": int(
                    final_info["normal_pellets_remaining"]
                ),
                "power_pellets_remaining": int(
                    final_info["power_pellets_remaining"]
                ),
                "final_score": int(final_info["score"]),
                "pygame_mode": int(final_info["pygame_mode"]),
                "wall_collisions": sum(
                    bool(step["wall_collision"]) for step in trajectory
                ),
                "oscillation_returns": sum(
                    bool(step["oscillation_return"]) for step in trajectory
                ),
                "parse_failures": parse_failures,
                "canonical_action_violations": canonical_violations,
                "won": (
                    terminal_reason == "all_normal_pellets"
                    and int(final_info["normal_pellets_remaining"]) == 0
                ),
                "terminal_reason": terminal_reason,
                "terminated": bool(trajectory[-1]["terminated"]),
                "truncated": bool(trajectory[-1]["truncated"]),
                "trajectory": trajectory,
            }
            audit_trajectory(payload)
            self.last_episode = payload
            trajectory_dir = options.get("trajectory_dir") or os.getenv(
                "PACMAN_TRAJECTORY_DIR"
            )
            if trajectory_dir:
                write_trajectory(payload, Path(trajectory_dir))

        if rewards_by_completion:
            return rewards_by_completion
        return float(payload["total_shaped_reward"])

    async def _call_model(
        self, messages: list[dict[str, Any]], **options: Any
    ) -> ModelTurn:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "install the openai extra or provide scripted_actions"
            ) from exc
        http_client = options.get("http_client")
        client = AsyncOpenAI(
            base_url=options.get("base_url") or os.getenv("OPENAI_BASE_URL"),
            api_key=options.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY",
            http_client=http_client,
            max_retries=0,
        )
        extra_body: dict[str, Any] = {
            "structured_outputs": {
                "choice": (
                    list(options["current_open_actions"])
                    if options.get("open_action_mask")
                    else list(EXPECTED_ACTIONS)
                )
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if options.get("open_action_mask"):
            current_open_actions = list(options["current_open_actions"])
            extra_body["allowed_token_ids"] = [
                self.action_token_id_by_action[action]
                for action in current_open_actions
            ]
        request = {
            "model": options.get("model", "default"),
            "messages": messages,
            "temperature": float(options.get("temperature", 0.0)),
            "top_p": float(options.get("top_p", 1.0)),
            "max_tokens": (
                1
                if options.get("open_action_mask")
                else int(options.get("max_completion_tokens", 3))
            ),
            "extra_body": extra_body,
        }
        try:
            response = await client.chat.completions.create(**request)
        finally:
            # AsyncOpenAI creates an httpx connection pool when no client is
            # supplied.  A Pacman episode makes hundreds of calls, so leaving
            # each short-lived pool open eventually exhausts local TCP
            # connections.  Externally supplied clients remain caller-owned.
            if http_client is None:
                await client.close()
        message = response.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", None)
        raw_response = (
            response.model_dump(mode="json")
            if hasattr(response, "model_dump")
            else None
        )
        return ModelTurn(
            completion=message.content or "",
            completion_id=getattr(response, "id", None),
            messages=messages,
            reasoning_content=reasoning_content,
            raw_response=raw_response,
            request_extra_body=extra_body,
        )


class PacmanNativeVisionWorkflow(PacmanImageOnlyWorkflow, RolloutWorkflow):
    """Native AReaL VLM workflow that keeps every training-time image tensor.

    The legacy agent workflow calls the OpenAI-compatible proxy and returns
    completion IDs plus rewards.  That proxy representation cannot preserve
    Qwen-VL processor outputs.  This workflow instead calls
    ``InferenceEngine.agenerate`` directly and returns the official tensor
    trajectory contract, including ``mm_token_type_ids`` and
    ``multi_modal_input``.
    """

    def __init__(
        self,
        *,
        gconfig: Any,
        tokenizer: Any,
        processor: Any,
        env_factory: Callable[[PygamePacmanEnvConfig], PygamePacmanEnv] = (
            PygamePacmanEnv
        ),
        **workflow_kwargs: Any,
    ) -> None:
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        if isinstance(processor, str):
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(processor)
        super().__init__(env_factory=env_factory, **workflow_kwargs)
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.processor = processor
        self.constrain_action_tokens = bool(
            workflow_kwargs.get("action_token_choice", False)
        )
        action_token_ids = []
        action_token_id_by_action: dict[str, int] = {}
        for token in MOVEMENT_ACTIONS:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"action {token!r} must map to exactly one tokenizer ID"
                )
            token_id = int(token_ids[0])
            action_token_ids.append(token_id)
            action_token_id_by_action[token] = token_id
        if len(set(action_token_ids)) != len(MOVEMENT_ACTIONS):
            raise ValueError("movement actions must have distinct tokenizer IDs")
        self.action_token_ids = action_token_ids
        self.action_token_id_by_action = action_token_id_by_action
        self.open_action_mask = bool(
            workflow_kwargs.get("open_action_mask", False)
        )
        if self.constrain_action_tokens or self.open_action_mask:
            install_vllm_allowed_token_ids_adapter()
        self._native_engine: ContextVar[Any | None] = ContextVar(
            "pacman_native_engine", default=None
        )
        self._native_turns: ContextVar[
            dict[str, tuple[dict[str, Any], Any, list[str]]] | None
        ] = ContextVar("pacman_native_turns", default=None)

    @staticmethod
    def _pil_and_chat_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        from PIL import Image

        image = None
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                chat_messages.append(
                    {"role": message["role"], "content": content}
                )
                continue
            converted: list[dict[str, Any]] = []
            for item in content:
                if item.get("type") == "text":
                    converted.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    prefix = "data:image/png;base64,"
                    if not url.startswith(prefix):
                        raise ValueError(
                            "native Pacman vision workflow requires an inline PNG"
                        )
                    image = Image.open(
                        BytesIO(base64.b64decode(url[len(prefix) :]))
                    ).convert("RGB")
                    converted.append({"type": "image", "image": image})
                else:
                    raise ValueError(
                        f"unsupported multimodal message item: {item!r}"
                    )
            chat_messages.append(
                {"role": message["role"], "content": converted}
            )
        if image is None:
            raise ValueError("native Pacman vision request has no image")
        return image, chat_messages

    def _process_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[Any, list[dict[str, Any]], dict[str, Any], list[int]]:
        image, chat_messages = self._pil_and_chat_messages(messages)
        text = self.processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        processed = self.processor(
            text=[text],
            images=[image],
            padding=False,
            return_tensors="pt",
        )
        mm_ids = processed.get("mm_token_type_ids")
        if mm_ids is None:
            mm_ids = processed.get("token_type_ids")
        if mm_ids is None:
            raise KeyError(
                "processor did not produce mm_token_type_ids or token_type_ids"
            )
        required = {"input_ids", "pixel_values"}
        missing = required - set(processed)
        if missing:
            raise KeyError(f"processor omitted required VLM fields: {missing}")
        return (
            image,
            chat_messages,
            dict(processed),
            processed["input_ids"].tolist()[0],
        )

    @staticmethod
    def _vllm_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build the JSON-safe vLLM chat form used with separate image_data."""
        converted = deepcopy(messages)
        image_parts = 0
        for message in converted:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "image_url"
                ):
                    item["image_url"] = {"url": "placeholder"}
                    image_parts += 1
        if image_parts != 1:
            raise ValueError(
                "native Pacman vLLM request requires exactly one image"
            )
        return converted

    async def _call_model(
        self, messages: list[dict[str, Any]], **options: Any
    ) -> ModelTurn:
        engine = self._native_engine.get()
        native_turns = self._native_turns.get()
        if engine is None or native_turns is None:
            raise RuntimeError("native inference engine is not attached")
        from areal.api import ModelRequest
        from areal.utils.image import image2base64

        image, chat_messages, processed, input_ids = self._process_messages(
            messages
        )
        request_id = uuid.uuid4().hex
        if self.open_action_mask:
            current_open_actions = list(
                options.get("current_open_actions") or []
            )
            if not current_open_actions:
                raise RuntimeError(
                    "open action mask requires current open actions"
                )
            allowed_token_ids = [
                self.action_token_id_by_action[action]
                for action in current_open_actions
            ]
        elif self.constrain_action_tokens:
            allowed_token_ids = self.action_token_ids
            current_open_actions = list(MOVEMENT_ACTIONS)
        else:
            allowed_token_ids = []
            current_open_actions = []
        request = ModelRequest(
            rid=request_id,
            input_ids=input_ids,
            image_data=image2base64(image),
            vision_msg_vllm=[self._vllm_messages(messages)],
            gconfig=self.gconfig.new(
                n_samples=1,
                min_new_tokens=(
                    1
                    if options.get("single_step") or self.open_action_mask
                    else getattr(self.gconfig, "min_new_tokens", 0)
                ),
                max_new_tokens=(
                    1
                    if options.get("single_step") or self.open_action_mask
                    else getattr(self.gconfig, "max_new_tokens", 3)
                ),
            ),
            tokenizer=self.tokenizer,
            processor=self.processor,
            metadata={
                "chat_template_kwargs": {"enable_thinking": False},
                **(
                    {"allowed_token_ids": allowed_token_ids}
                    if allowed_token_ids
                    else {}
                ),
            },
        )
        response = await engine.agenerate(request)
        if response.input_tokens != input_ids:
            raise RuntimeError(
                "rollout input tokens differ from processor input_ids"
            )
        if (
            len(response.output_tokens)
            != len(response.output_logprobs)
            or len(response.output_tokens) != len(response.output_versions)
        ):
            raise RuntimeError("incomplete native rollout token metadata")
        native_turns[request_id] = (
            processed,
            response,
            current_open_actions,
        )
        return ModelTurn(
            completion=self.tokenizer.decode(
                response.output_tokens,
                skip_special_tokens=True,
            ),
            completion_id=request_id,
            messages=messages,
            raw_response={
                "input_len": response.input_len,
                "output_len": response.output_len,
                "stop_reason": response.stop_reason,
            },
            request_extra_body={
                "native_areal_inference": True,
                "enable_thinking": False,
            },
        )

    @staticmethod
    def _tensor_sample(
        processed: dict[str, Any],
        response: Any,
        reward: float,
        allowed_actions: list[str],
    ) -> dict[str, Any]:
        import torch

        input_ids = list(response.input_tokens)
        output_ids = list(response.output_tokens)
        sequence = input_ids + output_ids
        mm_ids = processed.get("mm_token_type_ids")
        if mm_ids is None:
            mm_ids = processed.get("token_type_ids")
        mm_list = mm_ids.tolist()[0] + [0] * len(output_ids)
        if len(mm_list) != len(sequence):
            raise RuntimeError("multimodal token types do not align with sequence")
        action_mask_bits = 0
        for action in allowed_actions:
            try:
                action_mask_bits |= ACTION_MASK_BIT[action]
            except KeyError as exc:
                raise ValueError(
                    f"unknown masked Pacman action: {action!r}"
                ) from exc
        if allowed_actions and not action_mask_bits:
            raise RuntimeError("allowed Pacman actions produced an empty mask")
        multimodal = [{"pixel_values": processed["pixel_values"]}]
        if "image_grid_thw" in processed:
            multimodal[0]["image_grid_thw"] = processed["image_grid_thw"]
        return {
            "input_ids": torch.tensor(
                sequence, dtype=torch.long
            ).unsqueeze(0),
            "mm_token_type_ids": torch.tensor(
                mm_list, dtype=torch.long
            ).unsqueeze(0),
            "loss_mask": torch.tensor(
                [0] * len(input_ids) + [1] * len(output_ids),
                dtype=torch.int32,
            ).unsqueeze(0),
            # The bit mask is placed on each generated token. The causal actor
            # rolls it left once, just like input_ids, so the preceding logit
            # is normalized over the exact actions allowed during rollout.
            "pacman_action_mask_bits": torch.tensor(
                [0] * len(input_ids)
                + [action_mask_bits] * len(output_ids),
                dtype=torch.uint8,
            ).unsqueeze(0),
            "logprobs": torch.tensor(
                [0.0] * len(input_ids) + list(response.output_logprobs),
                dtype=torch.float32,
            ).unsqueeze(0),
            "versions": torch.tensor(
                [-1] * len(input_ids) + list(response.output_versions),
                dtype=torch.int32,
            ).unsqueeze(0),
            "attention_mask": torch.ones(
                len(sequence), dtype=torch.bool
            ).unsqueeze(0),
            "rewards": torch.tensor(
                [float(reward)], dtype=torch.float32
            ),
            "multi_modal_input": multimodal,
        }

    async def arun_episode(
        self, engine: Any, data: dict[str, Any]
    ) -> dict[str, Any]:
        from areal.utils.data import concat_padded_tensors

        engine_token = self._native_engine.set(engine)
        turns_token = self._native_turns.set({})
        try:
            rewards = await super().run(data)
        finally:
            native_turns = self._native_turns.get()
            self._native_turns.reset(turns_token)
            self._native_engine.reset(engine_token)
        if not isinstance(rewards, dict) or not rewards:
            raise RuntimeError(
                "native Pacman rollout did not return per-completion rewards"
            )
        if native_turns is None:
            raise RuntimeError("native processor state was not initialized")
        missing = set(rewards) - set(native_turns)
        if missing:
            raise RuntimeError(
                f"missing native processor data for completions: {sorted(missing)}"
            )
        samples = [
            self._tensor_sample(
                native_turns[completion_id][0],
                native_turns[completion_id][1],
                reward,
                native_turns[completion_id][2],
            )
            for completion_id, reward in rewards.items()
        ]
        return concat_padded_tensors(samples)
