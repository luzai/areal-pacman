from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from areal_pacman.level1.level1_dataset import make_episode_row
from areal_pacman.level1.prompts import PROMPT_STYLES
from areal_pacman.level1.trajectories import summarize_episodes
from areal_pacman.level1.workflow import PacmanImageOnlyWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thinking-disabled MaaPacman level-1 VLM evaluation."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--curriculum", type=int, choices=(1, 2), default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-completion-tokens", type=int, default=3)
    parser.add_argument("--wall-clock-limit-seconds", type=float)
    parser.add_argument("--stuck-no-progress-steps", type=int)
    parser.add_argument("--open-action-mask", action="store_true")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument(
        "--prompt-style",
        choices=PROMPT_STYLES,
        default="live_state_v3",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--pacman-python-root",
        type=Path,
        default=os.getenv("MAAPACMAN_PACMAN_PYTHON_ROOT"),
    )
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.pacman_python_root is not None:
        os.environ["MAAPACMAN_PACMAN_PYTHON_ROOT"] = str(
            args.pacman_python_root
        )
    trajectory_dir = args.trajectory_dir or (
        args.output.parent / f"{args.output.stem}_trajectories"
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_episode(index: int) -> dict[str, object]:
        async with semaphore:
            row = make_episode_row(
                index,
                split="test",
                seed=args.seed,
                max_steps=args.max_steps,
                curriculum=args.curriculum,
            )
            workflow = PacmanImageOnlyWorkflow(
                curriculum=args.curriculum,
                open_action_mask=args.open_action_mask,
                tokenizer_path=args.tokenizer_path,
            )
            await workflow.run(
                row,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                top_p=args.top_p,
                max_completion_tokens=args.max_completion_tokens,
                enable_thinking=False,
                image_prompt_style=args.prompt_style,
                pacman_python_root=args.pacman_python_root,
                trajectory_dir=trajectory_dir,
                wall_clock_limit_seconds=args.wall_clock_limit_seconds,
                stuck_no_progress_steps=args.stuck_no_progress_steps,
            )
            assert workflow.last_episode is not None
            return workflow.last_episode

    episodes = await asyncio.gather(
        *(run_episode(index) for index in range(1, args.episodes + 1))
    )
    return {
        "model": args.model,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "curriculum": args.curriculum,
        "concurrency": args.concurrency,
        "safety_limits": {
            "wall_clock_limit_seconds": args.wall_clock_limit_seconds,
            "stuck_no_progress_steps": args.stuck_no_progress_steps,
        },
        "image_prompt_style": args.prompt_style,
        "pacman_python_root": (
            str(args.pacman_python_root)
            if args.pacman_python_root is not None
            else None
        ),
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_completion_tokens": args.max_completion_tokens,
            "enable_thinking": False,
            "mode": "greedy" if args.temperature == 0.0 else "sampled",
        },
        "reward_contract": {
            "base_reward": "original pygame score delta",
            "step_penalty": 1.0,
            "wall_penalty": 1.0,
            "nearest_pellet_alpha": 0.0,
        },
        "observation_contract": (
            "screenshot_plus_live_state_and_navigation_history"
            if args.prompt_style == "live_state_v3"
            else "screenshot_only"
        ),
        "trajectory_dir": str(trajectory_dir),
        "episode_results": [
            {
                "id": episode["id"],
                "trajectory_sample_id": episode["trajectory_sample_id"],
                "final_score": episode["final_score"],
                "pellet_clear_rate": episode["pellet_clear_rate"],
                "normal_pellet_clear_rate": episode[
                    "normal_pellet_clear_rate"
                ],
                "normal_pellets_initial": episode["normal_pellets_initial"],
                "normal_pellets_eaten": episode["normal_pellets_eaten"],
                "normal_pellets_remaining": episode[
                    "normal_pellets_remaining"
                ],
                "power_pellets_remaining": episode[
                    "power_pellets_remaining"
                ],
                "wall_collisions": episode["wall_collisions"],
                "oscillation_returns": episode["oscillation_returns"],
                "steps": episode["steps"],
                "total_base_reward": episode["total_base_reward"],
                "total_shaped_reward": episode["total_shaped_reward"],
                "won": episode["won"],
                "terminal_reason": episode["terminal_reason"],
            }
            for episode in episodes
        ],
        **summarize_episodes(episodes),
    }


def main() -> None:
    args = parse_args()
    summary = asyncio.run(evaluate(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
