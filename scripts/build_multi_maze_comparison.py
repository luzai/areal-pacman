from __future__ import annotations

import argparse
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")


def load_summary(root: Path, phase: str, split: str) -> dict[str, object]:
    path = root / phase / f"summary_{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("included_split") != split:
        raise ValueError(f"{path} does not summarize split {split!r}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and post-RL multi-maze split metrics.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    result: dict[str, object] = {
        "maze_suite": "multi_maze_v1",
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "splits": {},
    }
    for split in SPLITS:
        baseline = load_summary(args.artifact_root, "baseline", split)
        post_rl = load_summary(args.artifact_root, "post_rl", split)
        result["splits"][split] = {
            "baseline": baseline,
            "post_rl": post_rl,
            "absolute_pass_rate_gain": float(post_rl["pass_rate"]) - float(baseline["pass_rate"]),
            "absolute_macro_layout_pass_rate_gain": float(post_rl["macro_layout_pass_rate"])
            - float(baseline["macro_layout_pass_rate"]),
            "layouts_with_any_win_gain": int(post_rl["layouts_with_any_win"])
            - int(baseline["layouts_with_any_win"]),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    compact = {
        split: {
            "baseline": values["baseline"]["pass_rate"],
            "post_rl": values["post_rl"]["pass_rate"],
            "gain": values["absolute_pass_rate_gain"],
            "baseline_macro": values["baseline"]["macro_layout_pass_rate"],
            "post_rl_macro": values["post_rl"]["macro_layout_pass_rate"],
            "macro_gain": values["absolute_macro_layout_pass_rate_gain"],
        }
        for split, values in result["splits"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
