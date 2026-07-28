from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from maapacman.env import PygamePacmanEnv


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an auditable level-1 run manifest.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    maa_repo = repo.parent / "MaaPacman"
    env = PygamePacmanEnv()
    try:
        spec = env.spec
    finally:
        env.close()
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    payload = {
        "recipe_version": "maapacman-level1-v1",
        "areal_pacman_revision": git_revision(repo),
        "maapacman_revision": git_revision(maa_repo),
        "env_api_version": spec.api_version,
        "env_id": spec.env_id,
        "level_revision": spec.level_revision,
        "action_tokens": list(spec.action_tokens),
        "model_revision": args.model_revision,
        "dataset": dataset,
        "seed_set": [int(dataset["seed"])],
        "launch_command": shlex.join(sys.argv),
    }
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    (args.artifact_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
