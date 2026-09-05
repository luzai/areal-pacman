from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open

from scripts.level1.train.validate_model_checkpoint import validate_model_checkpoint


MODEL_METADATA = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "chat_template.json",
    "video_preprocessor_config.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package a complete AReaL VLM checkpoint without replacing trained tensors."
    )
    parser.add_argument("--trained-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-prefix", default="model.visual.")
    parser.add_argument(
        "--restore-all-missing",
        action="store_true",
        help="Deprecated compatibility flag: ALL missing tensors are still rejected.",
    )
    parser.add_argument("--base-revision")
    parser.add_argument("--expected-base-revision")
    parser.add_argument(
        "--expected-saved-dtype", choices=("bfloat16",),
        help=("Explicitly require BF16 saved floating tensors and permit base "
              "FP32 -> saved BF16 storage differences. Copies bytes unchanged; "
              "does not prove training updates or model loading."),
    )
    parser.add_argument(
        "--config-comparison", choices=("strict", "qwen3_5"), default="strict",
        help=("qwen3_5 permits only audited serialization defaults/vision alias "
              "and requires equal configs after local Transformers normalization."),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_data_bytes(path: Path) -> int:
    """Safetensors bytes excluding the header, for HF's index total_size."""
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), byteorder="little")
    return path.stat().st_size - 8 - header_size


def weight_layout(root: Path) -> tuple[dict[str, str], dict[str, tuple]]:
    """Inspect actual single/sharded tensors, rejecting unsafe/inexact indices."""
    single = root / "model.safetensors"
    index = root / "model.safetensors.index.json"
    if single.is_file() and index.is_file():
        raise ValueError("ambiguous single and sharded checkpoint layout")
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
    elif single.is_file():
        with safe_open(single, framework="pt", device="cpu") as handle:
            mapping = {key: single.name for key in handle.keys()}
    else:
        raise ValueError(f"no supported safetensors layout: {root}")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("empty or invalid weight_map")
    if any(not isinstance(key, str) or not key for key in mapping):
        raise ValueError("invalid tensor key")
    if any(not isinstance(name, str) or Path(name).name != name or "\\" in name
           or not name.endswith(".safetensors") for name in mapping.values()):
        raise ValueError("unsafe shard filename")
    shapes = {}
    for name in sorted(set(mapping.values())):
        if not (root / name).is_file():
            raise ValueError(f"missing weight shard: {name}")
        with safe_open(root / name, framework="pt", device="cpu") as handle:
            expected = {key for key, shard in mapping.items() if shard == name}
            if set(handle.keys()) != expected:
                raise ValueError(f"weight index tensor keys mismatch: {name}")
            for key in expected:
                tensor = handle.get_slice(key)
                shapes[key] = (tuple(tensor.get_shape()), tensor.get_dtype())
    return mapping, shapes


def validate_storage_dtypes(trained_shapes: dict, base_shapes: dict,
                            expected_saved_dtype: str | None) -> list[dict]:
    """Validate an explicit save policy without casting or reading tensor data."""
    if expected_saved_dtype not in (None, "bfloat16"):
        raise ValueError("unsupported expected saved dtype")
    transitions = []
    for key in sorted(trained_shapes):
        shape, dtype = trained_shapes[key]
        base_shape, base_dtype = base_shapes[key]
        if shape != base_shape:
            raise ValueError(f"trained/base tensor shape or dtype mismatch: {key}")
        floating = dtype.startswith("F") or dtype == "BF16"
        base_floating = base_dtype.startswith("F") or base_dtype == "BF16"
        if expected_saved_dtype == "bfloat16" and (floating or base_floating):
            if dtype != "BF16" or base_dtype not in ("F32", "BF16"):
                raise ValueError(f"BF16 save policy dtype mismatch: {key}: {base_dtype} -> {dtype}")
        elif dtype != base_dtype:
            raise ValueError(f"trained/base tensor shape or dtype mismatch: {key}")
        if dtype != base_dtype:
            transitions.append({"key": key, "shape": list(shape),
                                "base_dtype": base_dtype, "saved_dtype": dtype})
    return transitions


_MISSING = object()


def _config_differences(base, trained, path=()):
    if isinstance(base, dict) and isinstance(trained, dict):
        for key in sorted(set(base) | set(trained)):
            yield from _config_differences(
                base.get(key, _MISSING), trained.get(key, _MISSING), (*path, key),
            )
    elif base != trained:
        yield path, base, trained


def validate_config_comparison(base_root: Path, trained_root: Path,
                               base_config: dict, trained_config: dict | None,
                               policy: str) -> dict:
    """Fail closed on raw changes before comparing normalized Qwen configs."""
    if policy == "strict":
        if trained_config is not None:
            for field in ("model_type", "text_config", "vision_config", "hidden_size", "vocab_size", "tie_word_embeddings"):
                if field in trained_config and field in base_config and trained_config[field] != base_config[field]:
                    raise ValueError(f"trained/base config mismatch: {field}")
        return {"policy": "strict"}
    if policy != "qwen3_5":
        raise ValueError("unsupported config comparison policy")
    if trained_config is None:
        raise ValueError("Qwen config comparison requires the trained config")
    if any(config.get("model_type") != "qwen3_5" for config in (base_config, trained_config)):
        raise ValueError("Qwen config comparison requires model_type=qwen3_5")
    import transformers
    from transformers import AutoConfig

    # A different library may interpret or drop different fields. Use the same
    # version that serialized the trained config, not an unverified migration.
    if trained_config.get("transformers_version") != transformers.__version__:
        raise ValueError("trained config Transformers version must match the installed version")
    defaults = {
        ("text_config", "bos_token_id"): None,
        ("text_config", "pad_token_id"): None,
        ("text_config", "partial_rotary_factor"): 0.25,
        ("text_config", "tie_word_embeddings"): False,
    }
    differences = []
    for path, before, after in _config_differences(base_config, trained_config):
        permitted = (
            (path in defaults and before is _MISSING and after == defaults[path])
            or (path == ("vision_config", "model_type")
                and before == "qwen3_5" and after == "qwen3_5_vision")
            or (path == ("transformers_version",)
                and isinstance(before, str) and after == transformers.__version__)
        )
        if not permitted:
            raise ValueError(f"unapproved raw config difference: {'.'.join(path)}")
        differences.append({"field": ".".join(path), "base_present": before is not _MISSING,
                            "base_value": None if before is _MISSING else before,
                            "trained_value": after})
    normalized = []
    for root in (base_root, trained_root):
        config = AutoConfig.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False,
        ).to_dict()
        config.pop("_name_or_path", None)  # local source location, not model semantics
        normalized.append(config)
    if normalized[0] != normalized[1]:
        raise ValueError("trained/base normalized Qwen config mismatch")
    fingerprint = hashlib.sha256(json.dumps(
        normalized[0], sort_keys=True, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    return {"policy": policy, "transformers_version": transformers.__version__,
            "approved_raw_differences": differences,
            "normalized_ignored_fields": ["_name_or_path"],
            "normalized_config_sha256": fingerprint,
            "actual_model_load_verified": False}


def build_checkpoint(args: argparse.Namespace) -> dict:
    args.trained_dir = args.trained_dir.resolve()
    args.base_dir = args.base_dir.resolve()
    destination = args.output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    if destination.is_relative_to(args.trained_dir) or destination.is_relative_to(args.base_dir):
        raise ValueError("output must be outside both input checkpoints")
    if args.visual_prefix != "model.visual.":
        raise ValueError("only the standard model.visual. prefix is supported")
    base_config_file = args.base_dir / "config.json"
    trained_config_file = args.trained_dir / "config.json"
    base_config_bytes = base_config_file.read_bytes()
    trained_config_bytes = trained_config_file.read_bytes() if trained_config_file.is_file() else None

    def require_unchanged_configs():
        for path, captured in ((base_config_file, base_config_bytes),
                               (trained_config_file, trained_config_bytes)):
            current = path.read_bytes() if path.is_file() else None
            if current != captured:
                raise ValueError(f"source config changed during export: {path}")

    trained_map, trained_shapes = weight_layout(args.trained_dir)
    base_weight_map, base_shapes = weight_layout(args.base_dir)
    base_validation = validate_model_checkpoint(args.base_dir, load_transformers_metadata=False)
    ignored_auxiliary = set(base_validation.get("ignored_auxiliary_tensors", []))
    trained_keys = set(trained_map)
    unexpected = trained_keys - set(base_weight_map)
    if unexpected:
        raise ValueError(f"unexpected trained tensor: {sorted(unexpected)[0]}")
    expected_saved_dtype = getattr(args, "expected_saved_dtype", None)
    dtype_transitions = validate_storage_dtypes(
        trained_shapes, base_shapes, expected_saved_dtype,
    )

    base_config = json.loads(base_config_bytes)
    trained_config = json.loads(trained_config_bytes) if trained_config_bytes is not None else None
    require_unchanged_configs()
    config_comparison = validate_config_comparison(
        args.base_dir, args.trained_dir, base_config, trained_config,
        getattr(args, "config_comparison", "strict"),
    )
    require_unchanged_configs()
    recorded_revision = base_config.get("_commit_hash")
    if args.base_revision and recorded_revision and args.base_revision != recorded_revision:
        raise ValueError("base-model revision mismatch: CLI label conflicts with config")
    # A user-provided revision label alone is not independent identity evidence.
    base_revision = recorded_revision
    if args.expected_base_revision and base_revision != args.expected_base_revision:
        raise ValueError(
            "base-model revision mismatch: "
            f"expected {args.expected_base_revision!r}, got {base_revision!r}"
        )

    # Qwen's official conditional-generation class can ignore auxiliary MTP
    # weights. This is an unused-key exception, never a missing-parameter repair.
    missing_keys = sorted(set(base_weight_map) - trained_keys - ignored_auxiliary)
    nonvisual = [key for key in missing_keys if not key.startswith(args.visual_prefix)]
    if nonvisual:
        raise ValueError(f"missing non-visual trained tensor (no base fallback): {nonvisual[0]}")
    if missing_keys:
        raise ValueError(
            "missing visual trained tensor: frozen-weight provenance is not established; "
            f"refusing base fallback: {missing_keys[0]}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed export retains only a unique partial directory, never a final bundle.
    args.output_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))

    copied_shards = {}
    source_hashes = {}
    source_names = sorted(set(trained_map.values()))
    for index, name in enumerate(source_names, 1):
        target = "trained-model.safetensors" if len(source_names) == 1 else f"trained-{index:05d}.safetensors"
        source_hashes[name] = sha256_file(args.trained_dir / name)
        shutil.copy2(args.trained_dir / name, args.output_dir / target)
        if sha256_file(args.output_dir / target) != source_hashes[name]:
            raise ValueError(f"trained shard changed during copy: {name}")
        copied_shards[name] = target
    weight_map = {key: copied_shards[trained_map[key]] for key in sorted(trained_keys)}
    index = {
        "metadata": {
            "total_size": sum(tensor_data_bytes(args.output_dir / name) for name in set(weight_map.values())),
            "trained_key_count": len(trained_keys),
            "restored_base_key_count": len(missing_keys),
            "restored_visual_key_count": sum(
                key.startswith(args.visual_prefix) for key in missing_keys
            ),
        },
        "weight_map": weight_map,
    }
    (args.output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for name in MODEL_METADATA:
        trained_source = args.trained_dir / name
        base_source = args.base_dir / name
        source = trained_source if trained_source.is_file() else base_source
        if source.is_file():
            metadata_target = args.output_dir / name
            shutil.copy2(source, metadata_target)

    for source_root in (args.base_dir, args.trained_dir):
        templates = source_root / "chat_templates"
        if templates.is_dir():
            for template in templates.glob("*.jinja"):
                (args.output_dir / "chat_templates").mkdir(exist_ok=True)
                shutil.copy2(template, args.output_dir / "chat_templates" / template.name)
    output_map, output_shapes = weight_layout(args.output_dir)
    if set(output_map) != trained_keys or output_shapes != trained_shapes:
        raise ValueError("trained tensor layout changed during export")
    require_unchanged_configs()
    expected_config_bytes = trained_config_bytes if trained_config_bytes is not None else base_config_bytes
    if (args.output_dir / "config.json").read_bytes() != expected_config_bytes:
        raise ValueError("copied config differs from the validated config bytes")
    file_hashes = {path.relative_to(args.output_dir).as_posix(): sha256_file(path)
                  for path in sorted(args.output_dir.rglob("*")) if path.is_file()}

    manifest = {
        "export_contract": "complete-vlm-checkpoint-v2",
        "trained_checkpoint": str(args.trained_dir.resolve()),
        "base_checkpoint": str(args.base_dir.resolve()),
        "base_model_revision": base_revision,
        "base_revision_label": args.base_revision,
        "base_revision_identity_verified": False,
        "missing_tensor_policy": "reject_all_no_frozen_visual_evidence",
        "base_architecture_validation": base_validation,
        "omitted_base_auxiliary_tensors": sorted(ignored_auxiliary - trained_keys),
        "trained_key_count": len(trained_keys),
        "expected_saved_dtype": expected_saved_dtype,
        "trained_storage_dtype_policy": (
            "explicit_bfloat16_save" if expected_saved_dtype else "strict_base_dtype"
        ),
        "trained_dtype_transitions": dtype_transitions,
        "trained_weight_bytes_modified": False,
        "config_comparison": config_comparison,
        "base_config_sha256": hashlib.sha256(base_config_bytes).hexdigest(),
        "trained_config_sha256": hashlib.sha256(trained_config_bytes).hexdigest() if trained_config_bytes is not None else None,
        "restore_all_missing": args.restore_all_missing,
        "restored_base_key_count": len(missing_keys),
        "restored_visual_key_count": sum(
            key.startswith(args.visual_prefix) for key in missing_keys
        ),
        "restored_base_shard_bytes": 0,
        "trained_source_sha256": source_hashes,
        "restored_base_source_sha256": {},
        "restored_tensor_keys": missing_keys,
        "files_sha256": file_hashes,
        "validation": "structural_export_only_not_model_load_or_gameplay",
        "output_index_sha256": sha256_file(args.output_dir / "model.safetensors.index.json"),
    }
    if len(source_names) == 1:
        manifest["trained_model_sha256"] = source_hashes[source_names[0]]
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    checksum = sha256_file(args.output_dir / "merge_manifest.json")
    (args.output_dir / "merge_manifest.sha256").write_text(checksum + "  merge_manifest.json\n", encoding="ascii")
    args.output_dir.rename(destination)
    return manifest


def main() -> None:
    print(json.dumps(build_checkpoint(parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
