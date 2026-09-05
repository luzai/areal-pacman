"""Fail-closed validation for a local Hugging Face model checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
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


def _safetensors_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise CheckpointValidationError(
            "safetensors is unavailable in the selected training environment"
        ) from exc
    try:
        with safe_open(path, framework="np") as handle:
            shapes = {
                key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()
            }
    except Exception as exc:
        raise CheckpointValidationError(
            f"invalid safetensors weight file: {path.name}"
        ) from exc
    if not shapes:
        raise CheckpointValidationError(
            f"safetensors weight file has no tensors: {path.name}"
        )
    return shapes


def _weight_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Inspect headers/meta tensors only, never materialize checkpoint weights."""
    if path.suffix == ".safetensors":
        return _safetensors_shapes(path)
    try:
        import torch

        state = torch.load(path, map_location="meta", weights_only=True, mmap=True)
        if not isinstance(state, dict) or not state:
            raise ValueError("expected a non-empty tensor state dict")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, torch.Tensor)
            or value.device.type != "meta"
            for key, value in state.items()
        ):
            raise ValueError("expected named meta tensors only")
        return {key: tuple(value.shape) for key, value in state.items()}
    except Exception as exc:
        raise CheckpointValidationError(
            f"cannot safely inspect weight file without loading weights: {path.name}: {exc}"
        ) from exc


def _validate_weight_layout(root: Path) -> tuple[str, tuple[str, ...]]:
    layouts = [single or index for single, index in WEIGHT_LAYOUTS]
    present = [name for name in layouts if (root / name).exists()]
    if len(present) > 1:
        raise CheckpointValidationError(
            "ambiguous weight layouts: " + ", ".join(present)
        )
    for single_name, index_name in WEIGHT_LAYOUTS:
        if single_name is not None:
            weight_file = root / single_name
            if not weight_file.exists():
                continue
            if not weight_file.is_file() or weight_file.stat().st_size <= 0:
                raise CheckpointValidationError(
                    f"weight file is missing or empty: {single_name}"
                )
            _weight_shapes(weight_file)
            return single_name, (single_name,)

        index_file = root / str(index_name)
        if not index_file.exists():
            continue
        index = _read_json_object(index_file, "weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointValidationError("weight index has no non-empty weight_map")
        if any(not isinstance(key, str) or not key for key in weight_map):
            raise CheckpointValidationError(
                "weight index contains an invalid tensor key"
            )
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
            ".safetensors" if index_name == "model.safetensors.index.json" else ".bin"
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
            actual_keys = set(_weight_shapes(shard))
            expected_keys = {
                key for key, name in weight_map.items() if name == shard_name
            }
            missing_keys = sorted(expected_keys - actual_keys)
            if missing_keys:
                raise CheckpointValidationError(
                    "weight index references tensors absent from "
                    f"{shard_name}: {missing_keys[0]}"
                )
            extra_keys = sorted(actual_keys - expected_keys)
            if extra_keys:
                raise CheckpointValidationError(
                    f"weight shard has tensors not mapped to it: {shard_name}: {extra_keys[0]}"
                )
        return str(index_name), tuple(shard_names)

    expected = ", ".join(
        single_name or str(index_name) for single_name, index_name in WEIGHT_LAYOUTS
    )
    raise CheckpointValidationError(f"no supported model weights found ({expected})")


def _architecture_tensor_policy_local(
    root: Path,
) -> tuple[dict[str, tuple[int, ...]], list[set[str]], str, tuple[str, ...]]:
    """Build the installed, trusted Transformers architecture on the meta device.

    Only actual parameter identity after the model's own tie_weights() is an
    alias exception. Regex ignore lists or the mere presence of similarly named
    tensors must never excuse missing trainable weights.
    """
    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
        )

        config = AutoConfig.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False
        )
        if getattr(config, "quantization_config", None):
            raise ValueError(
                "quantized checkpoint schema is not supported by this validator"
            )
        auto_class = next(
            (
                auto
                for auto in (AutoModelForImageTextToText, AutoModelForCausalLM)
                if type(config) in auto._model_mapping
            ),
            None,
        )
        if auto_class is None:
            raise ValueError(
                f"unsupported architecture for model_type={config.model_type}"
            )
        with torch.device("meta"):
            model = auto_class.from_config(
                config, trust_remote_code=False, attn_implementation="eager"
            )
            model.tie_weights()
        model_class = type(model).__name__
        declared = getattr(config, "architectures", None)
        if declared and model_class not in declared:
            raise ValueError(f"config declares {declared}, constructed {model_class}")
        if any(
            tensor.device.type != "meta"
            for tensor in (*model.parameters(), *model.buffers())
        ):
            raise ValueError(
                "architecture allocated non-meta tensors; cannot safely validate"
            )
        state = model.state_dict(keep_vars=True)
        if not state or any(
            not isinstance(value, torch.Tensor) for value in state.values()
        ):
            raise ValueError("architecture has an unsupported or empty state dict")
        expected = {key: tuple(value.shape) for key, value in state.items()}
        aliases: dict[int, set[str]] = {}
        for key, value in state.items():
            # Meta tensors all have a zero data_ptr; only object identity is safe.
            aliases.setdefault(id(value), set()).add(key)
        tied_groups = [keys for keys in aliases.values() if len(keys) > 1]
        ignored_unexpected = tuple(
            getattr(model, "_keys_to_ignore_on_load_unexpected", None) or ()
        )
        if any(not isinstance(pattern, str) for pattern in ignored_unexpected):
            raise ValueError("unsupported official unexpected-tensor ignore patterns")
        for pattern in ignored_unexpected:
            re.compile(pattern)
        return expected, tied_groups, model_class, ignored_unexpected
    except Exception as exc:
        raise CheckpointValidationError(
            f"cannot verify checkpoint architecture offline on meta device: {exc}"
        ) from exc


def architecture_tensor_policy(
    root: Path,
) -> tuple[dict[str, tuple[int, ...]], list[set[str]], str, tuple[str, ...]]:
    """Keep GPU-dependent constructors out of the caller's device environment.

    Transformers 5.7 Qwen3.5's fused linear-attention norm explicitly allocates
    CUDA parameters when a GPU is visible, even inside torch.device('meta').
    A fresh CPU-only interpreter avoids that branch without weakening the
    all-meta check or mutating a running trainer's process-global environment.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return _architecture_tensor_policy_local(root)
        child_env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        process = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                str(root.resolve()),
                "--internal-architecture-schema",
            ],
            env=child_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
            check=False,
        )
        if process.returncode:
            raise ValueError(
                f"CPU-only schema subprocess exited {process.returncode}: {process.stderr.strip()}"
            )
        result = json.loads(process.stdout)
        expected = result["expected_shapes"]
        groups = result["tied_groups"]
        model_class = result["model_class"]
        patterns = result["ignored_unexpected_patterns"]
        if (
            not isinstance(expected, dict)
            or not expected
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(shape, list)
                or any(type(size) is not int or size < 0 for size in shape)
                for key, shape in expected.items()
            )
        ):
            raise ValueError("invalid expected tensor schema from CPU-only subprocess")
        if not isinstance(groups, list) or any(
            not isinstance(group, list)
            or len(group) < 2
            or any(not isinstance(key, str) or key not in expected for key in group)
            or len(set(group)) != len(group)
            or any(expected[key] != expected[group[0]] for key in group)
            for group in groups
        ):
            raise ValueError("invalid tied tensor schema from CPU-only subprocess")
        if (
            not isinstance(model_class, str)
            or not model_class
            or not isinstance(patterns, list)
            or any(not isinstance(pattern, str) for pattern in patterns)
        ):
            raise ValueError("invalid model policy from CPU-only subprocess")
        for pattern in patterns:
            re.compile(pattern)
        return (
            {key: tuple(shape) for key, shape in expected.items()},
            [set(group) for group in groups],
            model_class,
            tuple(patterns),
        )
    except CheckpointValidationError:
        raise
    except Exception as exc:
        raise CheckpointValidationError(
            f"cannot verify checkpoint architecture in CPU-only schema process: {exc}"
        ) from exc


def _expected_architecture(
    root: Path,
) -> tuple[dict[str, tuple[int, ...]], list[set[str]], str]:
    """Compatibility helper for callers that only need required architecture."""
    return architecture_tensor_policy(root)[:3]


def ignored_auxiliary_tensor_names(
    names, expected: dict[str, tuple[int, ...]], ignored_unexpected: tuple[str, ...]
) -> set[str]:
    """Only selected official model-class extra-tensor rules are exceptions.

    This helper must never be used to excuse an absent expected tensor. In
    particular the VLM policy must not inherit CausalLM-only visual exclusions.
    """
    return {
        name
        for name in set(names) - set(expected)
        if any(re.search(pattern, name) for pattern in ignored_unexpected)
    }


def _validate_architecture(root: Path, weight_files: tuple[str, ...]) -> dict[str, Any]:
    expected, tied_groups, model_class, ignored_patterns = architecture_tensor_policy(
        root
    )
    actual: dict[str, tuple[int, ...]] = {}
    for name in weight_files:
        shard = _weight_shapes(root / name)
        duplicated = set(actual) & set(shard)
        if duplicated:
            raise CheckpointValidationError(
                f"tensor duplicated across shards: {sorted(duplicated)[0]}"
            )
        actual.update(shard)
    ignored_auxiliary = ignored_auxiliary_tensor_names(
        actual, expected, ignored_patterns
    )
    extra = set(actual) - set(expected) - ignored_auxiliary
    if extra:
        raise CheckpointValidationError(
            f"unexpected tensor for architecture: {sorted(extra)[0]}"
        )
    for key, shape in actual.items():
        if key in ignored_auxiliary:
            continue
        if shape != expected[key]:
            raise CheckpointValidationError(
                f"tensor shape mismatch: {key}: expected {expected[key]}, got {shape}"
            )
    missing = set(expected) - set(actual)
    tied_omissions: set[str] = set()
    for group in tied_groups:
        if group & set(actual):
            tied_omissions.update(group & missing)
    missing -= tied_omissions
    if missing:
        raise CheckpointValidationError(
            f"missing required architecture tensor: {sorted(missing)[0]}"
        )
    return {
        "architecture_verified": True,
        "architecture_validation": "offline-meta-keys-shapes-v1",
        "model_class": model_class,
        "expected_tensor_count": len(expected),
        "stored_tensor_count": len(actual),
        "tied_tensor_omissions": sorted(tied_omissions),
        "ignored_auxiliary_tensors": sorted(ignored_auxiliary),
        "official_unexpected_tensor_patterns": list(ignored_patterns),
    }


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


def validate_export_hashes(root: Path, weight_files: tuple[str, ...]) -> str | None:
    """Check exported artifact bytes, without confusing this with model loading."""
    manifest_path = root / "merge_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _read_json_object(manifest_path, "export manifest")
    if manifest.get("export_contract") != "complete-vlm-checkpoint-v2":
        raise CheckpointValidationError(
            "unversioned export manifest; rebuild the complete checkpoint"
        )
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    checksum = root / "merge_manifest.sha256"
    if (
        not checksum.is_file()
        or checksum.read_text(encoding="ascii").strip()
        != f"{manifest_hash}  merge_manifest.json"
    ):
        raise CheckpointValidationError("export manifest checksum mismatch")
    hashes = manifest.get("files_sha256")
    required = set(weight_files) | {"config.json"}
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (root / name).is_file():
            required.add(name)
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise CheckpointValidationError(
            "export manifest does not cover model weights/config/index"
        )
    for name, expected in hashes.items():
        if not isinstance(name, str) or "\\" in name:
            raise CheckpointValidationError("unsafe export manifest path")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise CheckpointValidationError("unsafe export manifest path")
        path = root.joinpath(*relative.parts)
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
        ):
            raise CheckpointValidationError(
                f"export file missing or externally linked: {name}"
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise CheckpointValidationError(f"export file checksum mismatch: {name}")
    return manifest_hash


def validate_model_checkpoint(
    checkpoint: Path,
    *,
    load_transformers_metadata: bool = True,
    validate_architecture: bool = True,
) -> dict[str, Any]:
    root = checkpoint.expanduser().resolve()
    if not root.is_dir():
        raise CheckpointValidationError(f"checkpoint is not a directory: {root}")
    _read_json_object(root / "config.json", "model config")
    layout, weight_files = _validate_weight_layout(root)
    export_hash = validate_export_hashes(root, weight_files)
    if load_transformers_metadata:
        _validate_transformers_metadata(root)
    architecture = (
        _validate_architecture(root, weight_files)
        if validate_architecture
        else {"architecture_verified": False}
    )
    return {
        "checkpoint": str(root),
        "weight_layout": layout,
        "weight_file_count": len(weight_files),
        "offline_metadata": load_transformers_metadata,
        "export_manifest_sha256": export_hash,
        "actual_model_load_verified": False,
        **architecture,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a complete local model checkpoint before training."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--internal-architecture-schema", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    try:
        if args.internal_architecture_schema:
            import torch

            if torch.cuda.is_available():
                raise CheckpointValidationError(
                    "schema subprocess must have no visible CUDA/HIP devices"
                )
            # Keep third-party constructor diagnostics off the JSON channel.
            with contextlib.redirect_stdout(sys.stderr):
                expected, groups, model_class, patterns = (
                    _architecture_tensor_policy_local(args.checkpoint)
                )
            print(
                json.dumps(
                    {
                        "expected_shapes": expected,
                        "tied_groups": [sorted(group) for group in groups],
                        "model_class": model_class,
                        "ignored_unexpected_patterns": patterns,
                    },
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            return 0
        result = validate_model_checkpoint(args.checkpoint)
    except CheckpointValidationError as exc:
        parser.exit(2, f"checkpoint_validation=failed: {exc}\n")
    print("checkpoint_validation=ok " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
