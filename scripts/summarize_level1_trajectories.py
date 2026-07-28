from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit collision-free level-1 trajectories and summarize the "
            "chronological train/validation batches."
        )
    )
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-batch-size", type=int, default=48)
    parser.add_argument("--validation-batch-size", type=int, default=2)
    return parser.parse_args()


def _read_trajectory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample_id = payload.get("trajectory_sample_id")
    row_id = payload.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"missing trajectory_sample_id: {path}")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError(f"missing dataset row id: {path}")
    expected_name = f"{row_id}--sample-{sample_id}.json"
    if path.name != expected_name:
        raise ValueError(
            f"trajectory filename does not match its IDs: {path.name} != "
            f"{expected_name}"
        )
    decoding = payload.get("decoding")
    if not isinstance(decoding, dict):
        raise ValueError(f"missing decoding contract: {path}")
    if decoding.get("enable_thinking") is not False:
        raise ValueError(f"thinking was not disabled: {path}")
    return payload


def _action_counts(payloads: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for payload in payloads:
        for step in payload.get("trajectory", []):
            action = step.get("action", step.get("executed_action"))
            if action is not None:
                counts[str(action)] += 1
    return dict(sorted(counts.items()))


def _reasoning_turns(payloads: Iterable[dict[str, Any]]) -> int:
    turns = 0
    for payload in payloads:
        turns += sum(
            bool(step.get("reasoning_content"))
            for step in payload.get("trajectory", [])
        )
    return turns


def summarize_batch(
    *,
    index: int,
    split: str,
    files: list[Path],
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    scores = [float(payload["final_score"]) for payload in payloads]
    clear_rates = [float(payload["pellet_clear_rate"]) for payload in payloads]
    normal_clear_rates = [
        float(payload["normal_pellet_clear_rate"]) for payload in payloads
    ]
    normal_pellets_eaten = [
        int(payload["normal_pellets_eaten"]) for payload in payloads
    ]
    walls = [int(payload["wall_collisions"]) for payload in payloads]
    oscillations = [
        int(payload.get("oscillation_returns", 0)) for payload in payloads
    ]
    steps = [int(payload["steps"]) for payload in payloads]
    return {
        "index": index,
        "split": split,
        "episodes": len(payloads),
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "dataset_rows": dict(
            sorted(Counter(str(payload["id"]) for payload in payloads).items())
        ),
        "decoding_contracts": [
            json.loads(value)
            for value in sorted(
                {
                    json.dumps(
                        payload["decoding"],
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for payload in payloads
                }
            )
        ],
        "prompt_styles": sorted(
            {str(payload.get("image_prompt_style")) for payload in payloads}
        ),
        "average_final_score": mean(scores),
        "minimum_final_score": min(scores),
        "maximum_final_score": max(scores),
        "average_pellet_clear_rate": mean(clear_rates),
        "average_normal_pellet_clear_rate": mean(normal_clear_rates),
        "average_normal_pellets_eaten": mean(normal_pellets_eaten),
        "average_wall_collisions": mean(walls),
        "average_wall_hit_rate": mean(
            wall / max(step_count, 1)
            for wall, step_count in zip(walls, steps, strict=True)
        ),
        "average_oscillation_returns": mean(oscillations),
        "average_oscillation_rate": mean(
            count / max(step_count, 1)
            for count, step_count in zip(oscillations, steps, strict=True)
        ),
        "full_completions": sum(bool(payload.get("won")) for payload in payloads),
        "parse_failures": sum(
            int(payload.get("parse_failures", 0)) for payload in payloads
        ),
        "reasoning_turns": _reasoning_turns(payloads),
        "action_counts": _action_counts(payloads),
    }


def audit(
    trajectory_dir: Path,
    *,
    train_batch_size: int,
    validation_batch_size: int,
) -> dict[str, Any]:
    if train_batch_size < 1 or validation_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    files = sorted(
        trajectory_dir.glob("*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not files:
        raise ValueError(f"no trajectory JSON files found: {trajectory_dir}")

    payloads = [_read_trajectory(path) for path in files]
    sample_ids = [str(payload["trajectory_sample_id"]) for payload in payloads]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate trajectory_sample_id values detected")

    expected_sizes = {
        "train": train_batch_size,
        "validation": validation_batch_size,
    }
    batches: list[dict[str, Any]] = []
    start = 0
    while start < len(files):
        split = str(payloads[start].get("split"))
        if split not in expected_sizes:
            raise ValueError(f"unexpected split {split!r}: {files[start]}")
        size = expected_sizes[split]
        stop = min(start + size, len(files))
        batch_files = files[start:stop]
        batch_payloads = payloads[start:stop]
        if any(str(payload.get("split")) != split for payload in batch_payloads):
            raise ValueError(
                f"split changed inside expected {split} batch starting at "
                f"{files[start].name}"
            )
        batches.append(
            summarize_batch(
                index=len(batches),
                split=split,
                files=batch_files,
                payloads=batch_payloads,
            )
        )
        start = stop

    return {
        "trajectory_dir": str(trajectory_dir.resolve()),
        "trajectory_files": len(files),
        "unique_sample_ids": len(set(sample_ids)),
        "complete_batches": sum(
            batch["episodes"] == expected_sizes[batch["split"]]
            for batch in batches
        ),
        "incomplete_batches": sum(
            batch["episodes"] != expected_sizes[batch["split"]]
            for batch in batches
        ),
        "batches": batches,
    }


def main() -> None:
    args = parse_args()
    result = audit(
        args.trajectory_dir,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")


if __name__ == "__main__":
    main()
