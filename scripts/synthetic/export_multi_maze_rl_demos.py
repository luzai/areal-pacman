from __future__ import annotations

import json
from pathlib import Path

from export_imageonly_notrain_demos import BAD, GOOD, export, load_json


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "run_artifacts" / "multi_maze_v1_20260715"
TRAJECTORY_DIR = RUN_ROOT / "post_rl" / "test" / "trajectories"
OUTPUT_DIR = RUN_ROOT / "report_assets"


def select_examples() -> tuple[Path, Path]:
    trajectories = [
        (path, item)
        for path in sorted(TRAJECTORY_DIR.glob("*.json"))
        if (item := load_json(path)).get("maze_split") == "test"
    ]
    successes = [(path, item) for path, item in trajectories if item.get("won")]
    caught = [
        (path, item)
        for path, item in trajectories
        if not item.get("won") and item.get("done_reason") == "caught"
    ]
    if not successes or not caught:
        raise ValueError("post-RL held-out test needs at least one success and one caught failure")
    success = min(successes, key=lambda pair: (int(pair[1]["steps"]), pair[0].name))
    failure = min(caught, key=lambda pair: (int(pair[1]["steps"]), pair[0].name))
    return success[0], failure[0]


def main() -> None:
    success, failure = select_examples()
    export(
        success,
        "post_rl_test_success",
        "POST-RL HELD-OUT SUCCESS",
        GOOD,
        output_dir=OUTPUT_DIR,
        no_train=False,
    )
    export(
        failure,
        "post_rl_test_failure",
        "POST-RL HELD-OUT FAILURE",
        BAD,
        output_dir=OUTPUT_DIR,
        no_train=False,
    )
    manifest = {
        "maze_suite": "multi_maze_v1",
        "split": "test",
        "phase": "post_rl",
        "success_source": str(success),
        "failure_source": str(failure),
    }
    (OUTPUT_DIR / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
