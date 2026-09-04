from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from areal_pacman.level1.level1_dataset import (
    DATASET_CONTRACT_VERSION,
    environment_metadata,
    generate_episode_rows,
    repository_revisions,
    validate_episode_row,
    write_hf_dataset,
    write_jsonl,
)
try:
    from scripts.level1.dataset.prepare_level1_v3_audits import (
        AUDIT_CONTRACT_VERSION,
        REPO_ROOT,
        _canonical_sha256,
        _generator_provenance,
        _reward_config,
        audit_planner_record,
        collect_initial_audit_anchor,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"scripts", "scripts.level1"}:
        raise
    from prepare_level1_v3_audits import (  # type: ignore[no-redef]
        AUDIT_CONTRACT_VERSION,
        REPO_ROOT,
        _canonical_sha256,
        _generator_provenance,
        _reward_config,
        audit_planner_record,
        collect_initial_audit_anchor,
    )


DATASET_PREPARATION_CONTRACT_VERSION = "maapacman-level1-split-bundle-v3"
DATASET_ROLES = ("train", "validation")


def _tree_sha256(root: Path) -> str:
    """Hash a directory without embedding its machine-specific absolute path."""

    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"cannot hash empty artifact directory: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _write_text_exclusive(
    path: Path, text: str, *, encoding: str = "utf-8"
) -> None:
    """Create an immutable manifest/checksum file and reject replacement."""

    with path.open("x", encoding=encoding, newline="\n") as stream:
        stream.write(text)


def _split_generator_provenance() -> dict[str, Any]:
    return _generator_provenance(
        [
            Path(__file__),
            REPO_ROOT
            / "scripts"
            / "level1"
            / "dataset"
            / "prepare_level1_v3_audits.py",
            REPO_ROOT / "areal_pacman" / "level1" / "level1_dataset.py",
            REPO_ROOT / "areal_pacman" / "level1" / "prompts.py",
            REPO_ROOT / "areal_pacman" / "level1" / "rewards.py",
            REPO_ROOT / "areal_pacman" / "level1" / "trajectories.py",
        ]
    )


def audit_episode_spec_row(row: dict[str, Any]) -> None:
    """Validate the real one-step evidence nested beside an episode spec."""

    anchor = row.get("audit_anchor")
    if not isinstance(anchor, dict):
        raise ValueError("episode specification is missing audit_anchor")
    audit_planner_record(anchor)
    if row.get("source_revisions") != anchor.get("source_revisions"):
        raise ValueError("episode source revisions differ from audit_anchor")
    if anchor.get("audit_role") != "episode_spec_audit_anchor":
        raise ValueError("episode audit_anchor has an invalid role")
    if anchor.get("audit_anchor_semantics") != {
        "scope": "initial_state_one_edward_step",
        "model_rollout": False,
        "training_sample": False,
    }:
        raise ValueError("episode audit_anchor semantics are invalid")
    if anchor.get("seed") != row["env"]["seed"] or anchor.get("step") != 1:
        raise ValueError("episode audit_anchor does not match the initial seed/state")
    if anchor.get("terminated") or anchor.get("truncated"):
        raise ValueError("episode audit_anchor initial transition must be nonterminal")
    identity_fields = {
        "name",
        "api_version",
        "backend",
        "pacman_python_revision",
        "pacman_python_source_sha256",
        "pacman_python_dirty",
        "maapacman_revision",
        "maapacman_env_source_sha256",
        "maapacman_dirty",
        "level_revision",
        "renderer_revision",
        "ruleset_revision",
    }
    if any(
        anchor["env"].get(field) != row["env"].get(field)
        for field in identity_fields
    ):
        raise ValueError("episode audit_anchor provenance differs from episode spec")
    validate_episode_row(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare immutable API-v3 MaaPacman level-1 episode rows.")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/datasets/level1_dataset"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--write-hf", action="store_true")
    parser.add_argument("--pacman-python-root")
    return parser.parse_args()


@contextmanager
def _temporary_pacman_python_root(
    pacman_python_root: str | os.PathLike[str] | None,
) -> Iterator[None]:
    """Select one checkout for all provenance calls without leaking state."""

    if pacman_python_root is None:
        yield
        return

    variable = "MAAPACMAN_PACMAN_ROOT"
    previous = os.environ.get(variable)
    os.environ[variable] = str(Path(pacman_python_root).resolve())
    repository_revisions.cache_clear()
    environment_metadata.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
        repository_revisions.cache_clear()
        environment_metadata.cache_clear()


def _prepare_dataset(args: argparse.Namespace) -> None:
    if args.train_episodes < 1 or args.validation_episodes < 1:
        raise ValueError("train and validation datasets must both be non-empty")
    args.output_root.mkdir(parents=True, exist_ok=False)
    metadata = environment_metadata()
    reward_config, config_sha256 = _reward_config(args.config)
    manifest: dict[str, Any] = {
        "preparation_contract_version": DATASET_PREPARATION_CONTRACT_VERSION,
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "dataset_roles": list(DATASET_ROLES),
        "environment": metadata,
        "source_revisions": repository_revisions(),
        "generator_provenance": _split_generator_provenance(),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "split_seed_contract": "disjoint_contiguous_seeds",
        "training_config_sha256": config_sha256,
        "reward_config": asdict(reward_config),
        "audit_anchor_contract": {
            "audit_contract_version": AUDIT_CONTRACT_VERSION,
            "scope": "initial_state_one_edward_step",
            "planner": "EdwardPlanner",
            "model_rollout": False,
            "training_sample": False,
            "per_row_reward_breakdown": "complete_and_audited",
        },
        "splits": {},
    }
    next_seed = args.seed
    for split, count in (
        ("train", args.train_episodes),
        ("validation", args.validation_episodes),
    ):
        rows = [
            next(
                generate_episode_rows(
                    1,
                    split=split,
                    seed=next_seed + index,
                    max_steps=args.max_steps,
                )
            )
            for index in range(count)
        ]
        next_seed += count
        for row in rows:
            row["audit_anchor"] = collect_initial_audit_anchor(
                seed=int(row["env"]["seed"]),
                max_steps=int(row["env"]["max_steps"]),
                reward_config=reward_config,
                pacman_python_root=args.pacman_python_root,
            )
            row["source_revisions"] = row["audit_anchor"]["source_revisions"]
            row["audit_anchor_sha256"] = _canonical_sha256(row["audit_anchor"])
            audit_episode_spec_row(row)
        jsonl = args.output_root / f"{split}.jsonl"
        digest = write_jsonl(rows, jsonl)
        artifact: dict[str, Any] = {
            "artifact_kind": "episode_specification",
            "rows": count,
            "seeds": [row["env"]["seed"] for row in rows],
            "jsonl": jsonl.relative_to(args.output_root).as_posix(),
            "sha256": digest,
            "reward_breakdown": {
                "episode_spec": "not_applicable",
                "audit_anchor": "complete_and_audited",
                "model_rollout": "required_at_training",
            },
        }
        if args.write_hf:
            hf_path = args.output_root / f"{split}_hf"
            write_hf_dataset(rows, hf_path)
            artifact["hf"] = {
                "path": hf_path.relative_to(args.output_root).as_posix(),
                "sha256": _tree_sha256(hf_path),
            }
        manifest["splits"][split] = artifact
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _write_text_exclusive(
        args.output_root / "manifest.json",
        manifest_text,
    )
    manifest_digest = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    _write_text_exclusive(
        args.output_root / "manifest.sha256",
        f"{manifest_digest}  manifest.json\n",
        encoding="ascii",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    with _temporary_pacman_python_root(args.pacman_python_root):
        _prepare_dataset(args)


if __name__ == "__main__":
    main()
