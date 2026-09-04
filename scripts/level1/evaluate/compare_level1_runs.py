from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare(baseline: dict[str, Any], trained: dict[str, Any]) -> dict[str, Any]:
    before = float(baseline["average_pellet_clear_rate"])
    after = float(trained["average_pellet_clear_rate"])
    improvement = after - before
    progress_pass = (after >= 0.5 and improvement >= 0.2) or (
        before > 0.5 and after >= 0.9
    )
    zero_action_errors = (
        int(trained["parse_failures"]) == 0
        and int(trained["canonical_action_violations"]) == 0
    )
    return {
        "baseline_pellet_clear_rate": before,
        "post_training_pellet_clear_rate": after,
        "improvement_percentage_points": improvement * 100.0,
        "progress_gate_passed": progress_pass,
        "zero_action_errors_passed": zero_action_errors,
        "accepted": progress_pass and zero_action_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen and trained level-1 evaluations.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--post-training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.post_training.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
