from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from areal_pacman.maze_suite import SPLIT_SIZES, maze_records


def load_trajectories(path: Path) -> list[dict[str, object]]:
    trajectories = []
    for file_path in sorted(path.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        payload["_source_file"] = str(file_path)
        trajectories.append(payload)
    if not trajectories:
        raise ValueError(f"no trajectory JSON files found in {path}")
    return trajectories


def summarize(
    trajectories: list[dict[str, object]],
    include_split: str | None = None,
    require_complete: bool = False,
) -> dict[str, object]:
    selected = [
        item
        for item in trajectories
        if include_split is None or item.get("maze_split") == include_split
    ]
    if not selected:
        raise ValueError(f"no trajectories matched split {include_split!r}")

    expected_names = (
        {str(record["name"]) for record in maze_records(include_split)}
        if include_split is not None
        else set()
    )
    observed_names = {str(item.get("layout_name")) for item in selected}
    missing_names = sorted(expected_names - observed_names)
    if require_complete and missing_names:
        raise ValueError(f"missing {len(missing_names)} expected layouts: {missing_names[:5]}")

    per_layout_items: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in selected:
        per_layout_items[str(item.get("layout_name"))].append(item)

    per_layout = {}
    for layout_name, items in sorted(per_layout_items.items()):
        wins = sum(bool(item.get("won")) for item in items)
        per_layout[layout_name] = {
            "samples": len(items),
            "wins": wins,
            "pass_rate": wins / len(items),
            "mean_reward": statistics.fmean(float(item.get("total_reward", 0.0)) for item in items),
            "reasons": dict(Counter(str(item.get("done_reason", "unknown")) for item in items)),
        }

    wins = sum(bool(item.get("won")) for item in selected)
    all_steps = [step for item in selected for step in item.get("trajectory", [])]
    samples_per_layout = Counter(values["samples"] for values in per_layout.values())
    return {
        "maze_suite": "multi_maze_v1",
        "included_split": include_split or "all",
        "trajectory_count": len(selected),
        "wins": wins,
        "pass_rate": wins / len(selected),
        "macro_layout_pass_rate": statistics.fmean(
            float(values["pass_rate"]) for values in per_layout.values()
        ),
        "samples_per_layout_histogram": {
            str(samples): layouts for samples, layouts in sorted(samples_per_layout.items())
        },
        "unique_layouts": len(observed_names),
        "expected_layouts": SPLIT_SIZES.get(include_split) if include_split else None,
        "missing_layouts": missing_names,
        "layouts_with_any_win": sum(values["wins"] > 0 for values in per_layout.values()),
        "reward_min": min(float(item.get("total_reward", 0.0)) for item in selected),
        "reward_max": max(float(item.get("total_reward", 0.0)) for item in selected),
        "reward_mean": statistics.fmean(float(item.get("total_reward", 0.0)) for item in selected),
        "steps_mean": statistics.fmean(int(item.get("steps", 0)) for item in selected),
        "terminal_reasons": dict(Counter(str(item.get("done_reason", "unknown")) for item in selected)),
        "parse_failed_steps": sum(bool(step.get("parse_failed")) for step in all_steps),
        "illegal_action_steps": sum(bool(step.get("illegal_action")) for step in all_steps),
        "per_layout": per_layout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multi-maze AReaL trajectory JSON files.")
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-split", choices=tuple(SPLIT_SIZES))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summary = summarize(
        load_trajectories(args.trajectory_dir),
        include_split=args.include_split,
        require_complete=args.require_complete,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_layout"}, indent=2))


if __name__ == "__main__":
    main()
