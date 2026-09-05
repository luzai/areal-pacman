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
    trained_map, trained_shapes = weight_layout(args.trained_dir)
    base_weight_map, base_shapes = weight_layout(args.base_dir)
    base_validation = validate_model_checkpoint(args.base_dir, load_transformers_metadata=False)
    ignored_auxiliary = set(base_validation.get("ignored_auxiliary_tensors", []))
    trained_keys = set(trained_map)
    unexpected = trained_keys - set(base_weight_map)
    if unexpected:
        raise ValueError(f"unexpected trained tensor: {sorted(unexpected)[0]}")
    for key in trained_keys:
        if trained_shapes[key] != base_shapes[key]:
            raise ValueError(f"trained/base tensor shape or dtype mismatch: {key}")

    base_config_file = args.base_dir / "config.json"
    base_config = json.loads(base_config_file.read_text(encoding="utf-8"))
    if (args.trained_dir / "config.json").is_file():
        trained_config = json.loads((args.trained_dir / "config.json").read_text(encoding="utf-8"))
        for field in ("model_type", "text_config", "vision_config", "hidden_size", "vocab_size", "tie_word_embeddings"):
            if field in trained_config and field in base_config and trained_config[field] != base_config[field]:
                raise ValueError(f"trained/base config mismatch: {field}")
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
    weight_layout(args.output_dir)
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
