#!/usr/bin/env python3
"""Backfill completed AReaL train-step metrics into a SwanLab experiment.

The source log remains authoritative; this script only creates a dashboard
representation for steps that have already completed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mK]")
REWARD = re.compile(r"ppo_actor/task_reward/avg\s*│\s*([-+0-9.eE]+)")
TRAIN_STEP = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+Step\s+(\d+)/(\d+)\s+Train step\s+(\d+)/(\d+)\s+done"
)
TRAJECTORY_SAMPLE_ID = re.compile(rb'"trajectory_sample_id"\s*:\s*"([^"]+)"')
WON = re.compile(rb'"won"\s*:\s*(true|false)')
SHAPED_REWARD = re.compile(rb'"total_shaped_reward"\s*:\s*([-+0-9.eE]+)')


def parse_completed_steps(log_path: Path) -> list[dict[str, float | int]]:
    """Return reward values only when their corresponding train step finished."""
    rows: list[dict[str, float | int]] = []
    pending_step: dict[str, int] | None = None
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        step_match = TRAIN_STEP.search(line)
        if step_match:
            epoch, total_epochs, _, _, update, total_updates = map(int, step_match.groups())
            pending_step = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "update": update,
                "total_updates": total_updates,
            }
            continue

        reward_match = REWARD.search(line)
        if reward_match and pending_step is not None:
            rows.append(
                {**pending_step, "train/task_reward_avg": float(reward_match.group(1))}
            )
            pending_step = None
    return rows


def add_train_trajectory_metrics(
    rows: list[dict[str, float | int]], trajectory_dir: Path, rollouts_per_update: int
) -> None:
    """Attach game outcome metrics, grouped by chronological training rollout batch.

    Only batches with a matching completed optimizer update are included.  This
    deliberately excludes partial trajectories from an update still in flight.
    """
    paths = sorted(
        trajectory_dir.glob("*-train-*.json"), key=lambda path: path.stat().st_mtime_ns
    )
    required = len(rows) * rollouts_per_update
    if len(paths) < required:
        raise ValueError(
            f"Need {required} train trajectories for {len(rows)} completed updates, "
            f"but found only {len(paths)}."
        )

    for row_index, row in enumerate(rows):
        batch_paths = paths[
            row_index * rollouts_per_update : (row_index + 1) * rollouts_per_update
        ]
        values = []
        for path in batch_paths:
            contents = path.read_bytes()
            sample_id = TRAJECTORY_SAMPLE_ID.search(contents)
            won = WON.search(contents)
            shaped_reward = SHAPED_REWARD.search(contents)
            if not (sample_id and won and shaped_reward):
                raise ValueError(f"Missing scalar outcome fields in {path}.")
            values.append(
                (
                    sample_id.group(1).decode("utf-8"),
                    won.group(1) == b"true",
                    float(shaped_reward.group(1)),
                )
            )
        ids = [sample_id for sample_id, _, _ in values]
        if len(set(ids)) != rollouts_per_update:
            raise ValueError(f"Duplicate trajectory id in update {row['update']}.")
        wins = sum(won for _, won, _ in values)
        rewards = [shaped_reward for _, _, shaped_reward in values]
        row["training/win_rate"] = wins / rollouts_per_update
        row["training/shaped_reward_avg"] = sum(rewards) / rollouts_per_update
        row["training/trajectory_count"] = rollouts_per_update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Previously verified metric rows produced with --print-only.",
    )
    parser.add_argument("--project", default="maapacman-level1")
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=("cloud", "offline"), default="cloud")
    parser.add_argument("--api-key-env", default="SWANLAB_API_KEY")
    parser.add_argument("--trajectory-dir", type=Path)
    parser.add_argument("--rollouts-per-update", type=int, default=48)
    parser.add_argument("--run-id", help="Existing SwanLab run id to resume.")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    if not args.log and not args.metrics_json:
        raise SystemExit("Provide --log or --metrics-json.")
    if args.log and args.metrics_json:
        raise SystemExit("Use either --log or --metrics-json, not both.")
    rows = (
        json.loads(args.metrics_json.read_text(encoding="utf-8"))
        if args.metrics_json
        else parse_completed_steps(args.log)
    )
    if not rows:
        raise SystemExit("No completed AReaL train steps with task_reward/avg were found.")
    if args.trajectory_dir:
        add_train_trajectory_metrics(rows, args.trajectory_dir, args.rollouts_per_update)
    if args.print_only:
        print(json.dumps(rows, indent=2))
        return

    import swanlab

    if args.mode == "cloud":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"Set {args.api_key_env} before cloud upload.")
        swanlab.login(api_key=api_key)

    run = swanlab.init(
        project=args.project,
        experiment_name=args.name,
        mode=args.mode,
        config={
            "source": "AReaL completed-step log backfill",
            "source_log": str(args.log or args.metrics_json),
            "metric_semantics": "rollout task reward, not validation reward",
        },
        id=args.run_id,
        resume="must" if args.run_id else None,
    )
    for row in rows:
        update = int(row["update"])
        swanlab.log(row, step=update)
    run.finish()
    print(f"Backfilled {len(rows)} completed updates.")


if __name__ == "__main__":
    main()
