from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import yaml
from maapacman.env import PygamePacmanEnv, PygamePacmanEnvConfig

from areal_pacman.level1.level1_dataset import repository_revisions
from areal_pacman.level1.rewards import REWARD_RECIPE_VERSION


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an auditable level-1 run manifest.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-override", action="append", default=[])
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    curriculum = config.get("curriculum") if isinstance(config, dict) else None
    if (
        not isinstance(curriculum, int)
        or isinstance(curriculum, bool)
        or curriculum not in (1, 2)
    ):
        raise ValueError("run manifest requires curriculum 1 or 2")
    env = PygamePacmanEnv(
        PygamePacmanEnvConfig(curriculum=int(curriculum))
    )
    try:
        spec = env.spec
        environment_provenance = env.provenance
    finally:
        env.close()
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if config.get("recipe_version") != "maapacman-level1-ghostdoor-v3":
        raise ValueError("run manifest requires the ghostdoor-v3 recipe")
    if config.get("reward_recipe_version") != REWARD_RECIPE_VERSION:
        raise ValueError("run manifest requires the API-v3 reward recipe")
    if dataset.get("curriculum") != curriculum:
        raise ValueError("dataset and training config curricula differ")
    split_seeds = {
        int(seed)
        for split in dataset.get("splits", {}).values()
        for seed in split.get("seeds", [])
    }
    if not split_seeds:
        raise ValueError("dataset manifest does not enumerate split seeds")
    source_revisions = repository_revisions()
    recipe_revision = source_revisions["areal-pacman"]["commit"]
    if (
        environment_provenance["maapacman_commit"] != recipe_revision
        or bool(environment_provenance["maapacman_dirty"])
        is not source_revisions["areal-pacman"]["dirty"]
    ):
        raise RuntimeError(
            "maapacman must be bundled in the active areal-pacman checkout"
        )
    payload = {
        "recipe_version": config["recipe_version"],
        "reward_recipe_version": config["reward_recipe_version"],
        "source_revisions": source_revisions,
        "areal_pacman_revision": recipe_revision,
        "maapacman_revision": recipe_revision,
        "pacman_python_revision": source_revisions["pacman-python"]["commit"],
        "areal_revision": source_revisions["AReaL"]["commit"],
        "env_api_version": spec.api_version,
        "env_id": spec.env_id,
        "level_revision": spec.level_revision,
        "renderer_revision": spec.renderer_revision,
        "ruleset_revision": spec.ruleset_revision,
        "curriculum": int(curriculum),
        "environment_provenance": environment_provenance,
        "action_tokens": list(spec.action_tokens),
        "model_revision": args.model_revision,
        "dataset": dataset,
        "seed_set": sorted(split_seeds),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_overrides": list(args.config_override),
        "launch_command": shlex.join(sys.argv),
    }
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    with (args.artifact_root / "manifest.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
