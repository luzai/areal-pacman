"""Deterministic episode specifications for the production level-1 recipe."""

from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache
from typing import Any, Iterable, Iterator, Mapping

from maapacman.env import (
    Position,
    PygamePacmanEnv,
    load_bundled_level,
    route_to_nearest,
)


ENV_NAME = "pacman-python-level1-pygame-v1"
ENV_API_VERSION = "1.0"
ENV_BACKEND = "original-pygame"
PRODUCTION_MAX_STEPS = 287
SHORT_HORIZON_MAX_STEPS = 32
LONG_HORIZON_MAX_STEPS = 256
DEMO_SAFETY_MAX_STEPS = 2000
SUPPORTED_MAX_STEPS = frozenset(
    {
        SHORT_HORIZON_MAX_STEPS,
        LONG_HORIZON_MAX_STEPS,
        PRODUCTION_MAX_STEPS,
        DEMO_SAFETY_MAX_STEPS,
    }
)


@lru_cache(maxsize=1)
def environment_metadata() -> dict[str, str]:
    env = PygamePacmanEnv()
    try:
        return {
            "name": env.spec.env_id,
            "api_version": env.spec.api_version,
            "backend": ENV_BACKEND,
            "pacman_python_revision": env.pacman_python_revision,
            "level_revision": env.spec.level_revision,
            "renderer_revision": env.spec.renderer_revision,
        }
    finally:
        env.close()


def environment_revision() -> str:
    return environment_metadata()["level_revision"]


def validate_episode_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("episode id must be a non-empty string")
    if row.get("split") not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    env = row.get("env")
    if not isinstance(env, Mapping):
        raise ValueError("env must be an object")
    installed = environment_metadata()
    if env.get("name") != ENV_NAME or env.get("name") != installed["name"]:
        raise ValueError(f"env.name must be {ENV_NAME!r}")
    if (
        env.get("api_version") != ENV_API_VERSION
        or env.get("api_version") != installed["api_version"]
    ):
        raise ValueError(f"env.api_version must be {ENV_API_VERSION!r}")
    if env.get("backend") != ENV_BACKEND:
        raise ValueError(f"env.backend must be {ENV_BACKEND!r}")
    if env.get("pacman_python_revision") != installed["pacman_python_revision"]:
        raise ValueError(
            "env.pacman_python_revision does not match installed pacman-python"
        )
    if env.get("level_revision") != installed["level_revision"]:
        raise ValueError("env.level_revision does not match installed MaaPacman")
    if env.get("level") != 1 or isinstance(env.get("level"), bool):
        raise ValueError("env.level must be integer 1")
    seed = env.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("env.seed must be an integer")
    max_steps = env.get("max_steps")
    if max_steps not in SUPPORTED_MAX_STEPS or isinstance(max_steps, bool):
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_MAX_STEPS))
        raise ValueError(f"env.max_steps must be one of: {supported}")
    if env.get("observation_mode") != "rgb":
        raise ValueError("env.observation_mode must be 'rgb'")
    prefix_actions = row.get("state_prefix_actions", [])
    if (
        not isinstance(prefix_actions, list)
        or any(action not in {"U", "D", "L", "R"} for action in prefix_actions)
    ):
        raise ValueError(
            "state_prefix_actions must be a list containing only U, D, L, or R"
        )
    decision_steps = row.get("decision_steps")
    if decision_steps is not None and decision_steps != 1:
        raise ValueError("decision_steps, when present, must be exactly 1")


def make_episode_row(
    index: int,
    *,
    split: str,
    seed: int = 0,
    max_steps: int = PRODUCTION_MAX_STEPS,
) -> dict[str, Any]:
    if not isinstance(index, int) or isinstance(index, bool) or index < 1:
        raise ValueError("index must be a positive integer")
    installed = environment_metadata()
    row = {
        "id": f"level1-seed{seed}-{split}-{index:04d}",
        "split": split,
        "env": {
            "name": ENV_NAME,
            "api_version": ENV_API_VERSION,
            "backend": ENV_BACKEND,
            "pacman_python_revision": installed["pacman_python_revision"],
            "level_revision": installed["level_revision"],
            "level": 1,
            "seed": seed,
            "max_steps": max_steps,
            "observation_mode": "rgb",
        },
    }
    validate_episode_row(row)
    return row


def generate_episode_rows(
    count: int,
    *,
    split: str,
    seed: int = 0,
    max_steps: int = PRODUCTION_MAX_STEPS,
) -> Iterator[dict[str, Any]]:
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    for index in range(1, count + 1):
        yield make_episode_row(index, split=split, seed=seed, max_steps=max_steps)


def oracle_state_prefixes(*, seed: int = 0) -> list[list[str]]:
    """Return deterministic, collision-free prefixes along the level-1 oracle."""
    return [record["prefix"] for record in oracle_state_records(seed=seed)]


def oracle_state_records(*, seed: int = 0) -> list[dict[str, Any]]:
    """Return oracle prefixes plus legal movement masks for state selection."""
    level = load_bundled_level()
    env = PygamePacmanEnv()
    try:
        _, info = env.reset(seed=seed)
        remaining = set(level.pellets)
        actions: list[str] = []
        records: list[dict[str, Any]] = []
        terminated = truncated = False
        while not (terminated or truncated):
            snapshot = env.snapshot()
            records.append(
                {
                    "state_index": len(actions),
                    "prefix": list(actions),
                    "open_actions": [
                        token
                        for token in ("U", "D", "L", "R")
                        if token in set(snapshot.get("open") or [])
                    ],
                }
            )
            row, col = info["pacman_position"]
            position = Position(int(row), int(col))
            remaining.discard(position)
            route = route_to_nearest(level, position, remaining)
            if not route:
                break
            action = route[0]
            _, _, terminated, truncated, info = env.step(action)
            if info["wall_collision"]:
                raise RuntimeError("level-1 oracle prefix collided with a wall")
            actions.append(action.value)
            if info["pellet_eaten"]:
                row, col = info["pacman_position"]
                remaining.discard(Position(int(row), int(col)))
        if not records:
            raise RuntimeError("level-1 oracle produced no decision states")
        return records
    finally:
        env.close()


def generate_single_step_rows(
    count: int,
    *,
    split: str,
    seed: int = 0,
    max_steps: int = PRODUCTION_MAX_STEPS,
    offset: int = 0,
) -> Iterator[dict[str, Any]]:
    """Generate varied screenshot -> one action -> immediate reward samples."""
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    prefixes = oracle_state_prefixes(seed=seed)
    if offset < 0 or offset + count > len(prefixes):
        raise ValueError(
            f"requested prefix range [{offset}, {offset + count}) exceeds "
            f"{len(prefixes)} available oracle states"
        )
    for index, prefix in enumerate(prefixes[offset : offset + count], start=1):
        row = make_episode_row(
            index,
            split=split,
            seed=seed,
            max_steps=max_steps,
        )
        row["id"] = f"level1-wall-{split}-{offset + index:04d}"
        row["state_prefix_actions"] = prefix
        row["decision_steps"] = 1
        validate_episode_row(row)
        yield row


def _evenly_spaced(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count > len(records):
        raise ValueError(f"requested {count} states from only {len(records)}")
    return [
        records[((2 * index + 1) * len(records)) // (2 * count)]
        for index in range(count)
    ]


def generate_balanced_corridor_rows(
    count: int,
    *,
    split: str,
    seed: int = 0,
    max_steps: int = PRODUCTION_MAX_STEPS,
) -> Iterator[dict[str, Any]]:
    """Generate a 50/50 horizontal-vs-vertical held-out corridor task.

    Every direction is open in exactly half of the selected states. This
    prevents a constant action such as always-R from earning positive reward
    without reading the screenshot.
    """
    if split not in {"train", "validation"}:
        raise ValueError("balanced corridor split must be train or validation")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count must be a positive integer")
    if count % 2:
        raise ValueError("balanced corridor row count must be even")

    by_mask: dict[tuple[str, ...], list[dict[str, Any]]] = {
        ("L", "R"): [],
        ("U", "D"): [],
    }
    for record in oracle_state_records(seed=seed):
        mask = tuple(record["open_actions"])
        if mask in by_mask:
            by_mask[mask].append(record)

    needed_per_mask = 24
    selected = {
        mask: _evenly_spaced(records, needed_per_mask)
        for mask, records in by_mask.items()
    }
    records_by_mask: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for mask in (("L", "R"), ("U", "D")):
        records = selected[mask]
        candidates = [
            record
            for index, record in enumerate(records)
            if (index % 3 == 2) == (split == "validation")
        ]
        records_by_mask[mask] = candidates[: count // 2]
    split_records = [
        record
        for pair in zip(
            records_by_mask[("L", "R")],
            records_by_mask[("U", "D")],
            strict=True,
        )
        for record in pair
    ]

    for index, record in enumerate(split_records, start=1):
        row = make_episode_row(
            index,
            split=split,
            seed=seed,
            max_steps=max_steps,
        )
        row["id"] = (
            f"level1-wall-{split}-state{int(record['state_index']):04d}"
        )
        row["state_prefix_actions"] = list(record["prefix"])
        row["decision_steps"] = 1
        row["state_open_actions_for_audit"] = list(record["open_actions"])
        validate_episode_row(row)
        yield row


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> str:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        validate_episode_row(row)
        normalized.append(dict(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in normalized
    )
    path.write_text(content, encoding="utf-8", newline="\n")
    import hashlib

    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_hf_dataset(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("install the 'datasets' package to write HF datasets") from exc
    materialized = [dict(row) for row in rows]
    for row in materialized:
        validate_episode_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(materialized).save_to_disk(str(path))
