from __future__ import annotations

import argparse
import json
from pathlib import Path

from areal_pacman.level1.level1_dataset import (
    generate_episode_rows,
    write_hf_dataset,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic MaaPacman level-1 episode rows.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/datasets/level1_dataset"))
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=287)
    parser.add_argument("--write-hf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest: dict[str, object] = {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "splits": {},
    }
    for split, count in (
        ("train", args.train_episodes),
        ("validation", args.validation_episodes),
    ):
        rows = list(
            generate_episode_rows(
                count,
                split=split,
                seed=args.seed,
                max_steps=args.max_steps,
            )
        )
        jsonl = args.output_root / f"{split}.jsonl"
        digest = write_jsonl(rows, jsonl)
        if args.write_hf:
            write_hf_dataset(rows, args.output_root / f"{split}_hf")
        manifest["splits"][split] = {
            "rows": count,
            "jsonl": str(jsonl),
            "sha256": digest,
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
