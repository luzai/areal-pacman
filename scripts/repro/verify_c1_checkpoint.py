#!/usr/bin/env python3
"""Fail-closed checks for the C1 Iter25 stage handoffs."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty file: {path}")


def verify_hf(path: Path, expected_global_step: int) -> dict[str, Any]:
    if not path.is_dir():
        raise RuntimeError(f"HF checkpoint directory does not exist: {path}")
    if f"globalstep{expected_global_step}" not in path.name:
        raise RuntimeError(
            f"expected globalstep{expected_global_step} HF checkpoint, got {path.name}"
        )
    _require_file(path / "config.json")
    index = path / "model.safetensors.index.json"
    if index.exists():
        _require_file(index)
        payload = json.loads(index.read_text(encoding="utf-8"))
        shards = sorted(set(payload.get("weight_map", {}).values()))
        if not shards:
            raise RuntimeError(f"empty safetensors weight map: {index}")
        for shard in shards:
            _require_file(path / shard)
    else:
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise RuntimeError(f"no safetensors weights found in {path}")
        for shard in shards:
            _require_file(shard)
    return {
        "kind": "hf",
        "path": str(path.resolve()),
        "global_step": expected_global_step,
        "weight_files": len(shards),
    }


def _metadata_keys(metadata: object) -> list[str]:
    mapping = getattr(metadata, "state_dict_metadata", None)
    if mapping is None and isinstance(metadata, dict):
        mapping = metadata.get("state_dict_metadata", metadata)
    if not isinstance(mapping, dict):
        raise TypeError("DCP .metadata has no state_dict_metadata mapping")
    return sorted(str(key) for key in mapping)


def verify_dcp(
    path: Path, recover_info: Path, expected_global_step: int
) -> dict[str, Any]:
    if not path.is_dir():
        raise RuntimeError(f"DCP checkpoint directory does not exist: {path}")
    metadata_path = path / ".metadata"
    _require_file(metadata_path)
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)  # trusted output of this training run
    keys = _metadata_keys(metadata)
    model_keys = [key for key in keys if ".model." in f".{key}."]
    optim_keys = [key for key in keys if ".optim." in f".{key}."]
    if not model_keys:
        raise RuntimeError("DCP metadata contains no model state")
    if not optim_keys:
        raise RuntimeError("DCP metadata contains no optimizer state")

    step_info_path = recover_info / "step_info.json"
    _require_file(step_info_path)
    step_info = json.loads(step_info_path.read_text(encoding="utf-8"))
    actual_step = step_info.get("global_step")
    if actual_step != expected_global_step:
        raise RuntimeError(
            f"recover global_step mismatch: expected {expected_global_step}, got {actual_step}"
        )
    for name in (
        "checkpoint_info.json",
        "dataloader_info.pkl",
        "evaluator_info.json",
        "saver_info.json",
        "stats_logger_info.json",
    ):
        _require_file(recover_info / name)
    return {
        "kind": "dcp",
        "path": str(path.resolve()),
        "recover_info": str(recover_info.resolve()),
        "global_step": expected_global_step,
        "model_metadata_keys": len(model_keys),
        "optimizer_metadata_keys": len(optim_keys),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="kind", required=True)
    hf = subparsers.add_parser("hf")
    hf.add_argument("--path", type=Path, required=True)
    hf.add_argument("--expected-global-step", type=int, required=True)
    dcp = subparsers.add_parser("dcp")
    dcp.add_argument("--path", type=Path, required=True)
    dcp.add_argument("--recover-info", type=Path, required=True)
    dcp.add_argument("--expected-global-step", type=int, required=True)
    args = parser.parse_args()
    if args.kind == "hf":
        result = verify_hf(args.path, args.expected_global_step)
    else:
        result = verify_dcp(args.path, args.recover_info, args.expected_global_step)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
