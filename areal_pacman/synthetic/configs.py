from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig


@dataclass
class PacmanAgentConfig(PPOConfig):
    recipe_version: str = field(
        default="research-scaffold",
        metadata={"help": "Recipe contract identifier stored with run artifacts."},
    )
    artifact_root: str = field(
        default="run_artifacts",
        metadata={"help": "Configurable root for datasets, trajectories, and checkpoints."},
    )
    trajectory_dir: str | None = field(
        default=None,
        metadata={"help": "Write one verbatim model-response trajectory JSON per rollout episode."},
    )
    step_penalty: float = field(
        default=1.0,
        metadata={"help": "Penalty charged for every executed environment step."},
    )
    wall_penalty: float = field(
        default=1.0,
        metadata={"help": "Non-negative shaped penalty for a move into a wall."},
    )
    use_base_reward: bool = field(
        default=True,
        metadata={"help": "Include the original Pacman score delta in reward."},
    )
    normal_pellet_reward: float = field(
        default=0.0,
        metadata={"help": "Explicit reward for eating one normal pellet."},
    )
    power_pellet_reward: float = field(
        default=0.0,
        metadata={"help": "Explicit reward for eating one power pellet."},
    )
    completion_reward: float = field(
        default=0.0,
        metadata={"help": "Terminal reward for clearing all normal pellets."},
    )
    nearest_pellet_alpha: float = field(
        default=0.0,
        metadata={
            "help": (
                "Alpha for level-1 nearest-normal-pellet distance progress; "
                "zero preserves the sparse reward contract."
            )
        },
    )
    nearest_pellet_remaining_ratio_threshold: float = field(
        default=1.0,
        metadata={
            "help": (
                "Enable nearest-pellet shaping at or below this fraction "
                "of remaining normal pellets."
            )
        },
    )
    nearest_pellet_scale_by_cleared_ratio: bool = field(
        default=False,
        metadata={
            "help": (
                "Scale nearest-pellet distance progress by the cleared "
                "normal-pellet ratio (1 - remaining ratio)."
            )
        },
    )
    nearest_pellet_skip_on_eat: bool = field(
        default=False,
        metadata={
            "help": (
                "Disable nearest-pellet distance shaping on steps that eat "
                "a normal pellet."
            )
        },
    )
    allow_unoffloaded_actor_colocated_ref_for_smoke: bool = field(
        default=False,
        metadata={
            "help": (
                "Smoke-only escape hatch for small models that fit an "
                "actor-colocated reference without native FSDP parameter "
                "offload. Production 9B runs must leave this false."
            )
        },
    )
    workflow: str = field(
        default="areal_pacman.areal_workflow.PacmanWorkflow",
        metadata={"help": "Workflow class for PacMan training."},
    )
    eval_workflow: str = field(
        default="areal_pacman.areal_workflow.PacmanWorkflow",
        metadata={"help": "Workflow class for PacMan evaluation."},
    )
    validation_contract: str = field(
        default="greedy1",
        metadata={
            "help": (
                "Level-1 validation/reporting contract. Production follow-up "
                "runs use sampled12_uniform when train, validation, and test "
                "all use the same sampled decoding distribution."
            )
        },
    )
    enable_thinking: bool | None = field(
        default=None,
        metadata={"help": "Optional chat-template thinking toggle for Qwen-style models."},
    )
    image_prompt_style: str = field(
        default="minimal_v1",
        metadata={
            "help": (
                "Level-1 prompt contract: minimal_v1, live_static_v2, or "
                "live_state_v3 (screenshot plus authoritative engine state "
                "and bounded navigation history)."
            )
        },
    )
    legal_action_mask: bool = field(
        default=False,
        metadata={"help": "Post-parse PacMan actions to the current legal action set during rollout."},
    )
    open_action_mask: bool = field(
        default=False,
        metadata={
            "help": (
                "Constrain each level-1 model generation to the movement actions "
                "that are open in the current live environment state."
            )
        },
    )
    guided_action_choice: bool = field(
        default=False,
        metadata={"help": "Use vLLM structured_outputs.choice to constrain rollout completions."},
    )
    legal_action_choice: bool = field(
        default=False,
        metadata={"help": "Use vLLM structured_outputs.choice over the current legal PacMan action tokens."},
    )
    non_stay_legal_action_choice: bool = field(
        default=False,
        metadata={"help": "Use vLLM structured_outputs.choice over current legal non-stay actions when any movement is legal."},
    )
    non_backtracking_legal_action_choice: bool = field(
        default=False,
        metadata={"help": "Use vLLM structured_outputs.choice over legal non-stay actions, excluding immediate reverse when another choice exists."},
    )
    action_token_choice: bool = field(
        default=False,
        metadata={"help": "Use vLLM structured_outputs.choice over all PacMan action tokens only, without legal-action masking."},
    )
    completion_api: str = field(
        default="chat",
        metadata={"help": "OpenAI API style for rollout calls: 'chat', 'chat_user', or 'completion'."},
    )
    parse_failure_penalty: int = field(
        default=-50,
        metadata={"help": "Terminal reward used when a model output cannot be parsed into a PacMan action."},
    )
    reward_mode: str = field(
        default="sparse",
        metadata={"help": "Environment reward mode, including sparse, route-prefix variants, and safe_progress."},
    )
    route_shaping_scale: float = field(
        default=1.0,
        metadata={"help": "Scale hidden route/progress shaping terms without changing sparse PacMan rewards."},
    )
    safe_progress_alpha: float = field(
        default=1.0,
        metadata={"help": "Alpha for safe_progress: original reward + alpha * (safe_distance_before - safe_distance_after)."},
    )
    observation_mode: str = field(
        default="text",
        metadata={"help": "Episode observation mode: text, image, image_text, or image_only."},
    )
    vision_tile_size: int = field(
        default=32,
        metadata={"help": "Tile size in pixels for image/image_text PacMan observations."},
    )
    store_observation_images: bool = field(
        default=False,
        metadata={"help": "Store base64 image data URLs in trajectory JSON for debugging."},
    )
