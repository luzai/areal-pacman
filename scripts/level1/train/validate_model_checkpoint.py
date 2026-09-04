"""Fail-closed validation for a local Hugging Face model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class CheckpointValidationError(ValueError):
    """The selected model directory cannot be loaded as a complete checkpoint."""


WEIGHT_LAYOUTS = (
    ("model.safetensors", None),
    (None, "model.safetensors.index.json"),
    ("pytorch_model.bin", None),
    (None, "pytorch_model.bin.index.json"),
)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointValidationError(f"missing {label}: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"invalid {label}: {path.name}") from exc
    if not isinstance(payload, dict):
        raise CheckpointValidationError(f"{label} must contain a JSON object")
    return payload


def _safetensors_keys(path: Path) -> set[str]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise CheckpointValidationError(
            "safetensors is unavailable in the selected training environment"
        ) from exc
    try:
        with safe_open(path, framework="np") as handle:
            keys = set(handle.keys())
    except Exception as exc:
        raise CheckpointValidationError(
            f"invalid safetensors weight file: {path.name}"
        ) from exc
    if not keys:
        raise CheckpointValidationError(
            f"safetensors weight file has no tensors: {path.name}"
        )
    return keys


def _validate_weight_layout(root: Path) -> tuple[str, tuple[str, ...]]:
    for single_name, index_name in WEIGHT_LAYOUTS:
        if single_name is not None:
            weight_file = root / single_name
            if not weight_file.exists():
                continue
            if not weight_file.is_file() or weight_file.stat().st_size <= 0:
                raise CheckpointValidationError(
                    f"weight file is missing or empty: {single_name}"
                )
            if single_name.endswith(".safetensors"):
                _safetensors_keys(weight_file)
            return single_name, (single_name,)

        index_file = root / str(index_name)
        if not index_file.exists():
            continue
        index = _read_json_object(index_file, "weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointValidationError("weight index has no non-empty weight_map")
        if any(not isinstance(key, str) or not key for key in weight_map):
            raise CheckpointValidationError("weight index contains an invalid tensor key")
        raw_shard_names = list(weight_map.values())
        if any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "\\" in name
            for name in raw_shard_names
        ):
            raise CheckpointValidationError(
                "weight index contains an invalid shard filename"
            )
        expected_suffix = (
            ".safetensors"
            if index_name == "model.safetensors.index.json"
            else ".bin"
        )
        if any(not name.endswith(expected_suffix) for name in raw_shard_names):
            raise CheckpointValidationError(
                f"weight index shard must end with {expected_suffix}"
            )
        shard_names = sorted(set(raw_shard_names))
        for shard_name in shard_names:
            shard = root / shard_name
            if not shard.is_file() or shard.stat().st_size <= 0:
                raise CheckpointValidationError(
                    f"referenced weight shard is missing or empty: {shard_name}"
                )
            if shard_name.endswith(".safetensors"):
                actual_keys = _safetensors_keys(shard)
                expected_keys = {
                    key for key, name in weight_map.items() if name == shard_name
                }
                missing_keys = sorted(expected_keys - actual_keys)
                if missing_keys:
                    raise CheckpointValidationError(
                        "weight index references tensors absent from "
                        f"{shard_name}: {missing_keys[0]}"
                    )
        return str(index_name), tuple(shard_names)

    expected = ", ".join(
        single_name or str(index_name) for single_name, index_name in WEIGHT_LAYOUTS
    )
    raise CheckpointValidationError(f"no supported model weights found ({expected})")


def _validate_transformers_metadata(root: Path) -> None:
    try:
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise CheckpointValidationError(
            "transformers is unavailable in the selected training environment"
        ) from exc

    for label, loader in (
        ("config", AutoConfig),
        ("tokenizer", AutoTokenizer),
        ("processor", AutoProcessor),
    ):
        try:
            loader.from_pretrained(
                str(root),
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise CheckpointValidationError(
                f"{label} metadata is not loadable offline: {exc}"
            ) from exc


def validate_model_checkpoint(
    checkpoint: Path,
    *,
    load_transformers_metadata: bool = True,
) -> dict[str, Any]:
    root = checkpoint.expanduser().resolve()
    if not root.is_dir():
        raise CheckpointValidationError(f"checkpoint is not a directory: {root}")
    _read_json_object(root / "config.json", "model config")
    layout, weight_files = _validate_weight_layout(root)
    if load_transformers_metadata:
        _validate_transformers_metadata(root)
    return {
        "checkpoint": str(root),
        "weight_layout": layout,
        "weight_file_count": len(weight_files),
        "offline_metadata": load_transformers_metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a complete local model checkpoint before training."
    )
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    try:
        result = validate_model_checkpoint(args.checkpoint)
    except CheckpointValidationError as exc:
        parser.exit(2, f"checkpoint_validation=failed: {exc}\n")
    print("checkpoint_validation=ok " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
