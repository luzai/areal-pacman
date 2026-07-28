from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete an AReaL VLM checkpoint with frozen visual weights."
    )
    parser.add_argument("--trained-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-prefix", default="model.visual.")
    parser.add_argument("--base-revision")
    parser.add_argument("--expected-base-revision")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    trained_file = args.trained_dir / "model.safetensors"
    base_index_file = args.base_dir / "model.safetensors.index.json"
    if not trained_file.is_file():
        raise FileNotFoundError(trained_file)
    if not base_index_file.is_file():
        raise FileNotFoundError(base_index_file)

    base_config_file = args.base_dir / "config.json"
    base_config = (
        json.loads(base_config_file.read_text(encoding="utf-8"))
        if base_config_file.is_file()
        else {}
    )
    base_revision = args.base_revision or base_config.get("_commit_hash")
    if args.expected_base_revision and base_revision != args.expected_base_revision:
        raise ValueError(
            "base-model revision mismatch: "
            f"expected {args.expected_base_revision!r}, got {base_revision!r}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with safe_open(trained_file, framework="pt", device="cpu") as handle:
        trained_keys = set(handle.keys())

    base_index = json.loads(base_index_file.read_text(encoding="utf-8"))
    base_weight_map: dict[str, str] = base_index["weight_map"]
    visual_keys = sorted(
        key
        for key in base_weight_map
        if key.startswith(args.visual_prefix) and key not in trained_keys
    )
    if not visual_keys:
        raise ValueError("No missing visual weights found in the base checkpoint")

    visual_tensors = {}
    keys_by_shard: dict[str, list[str]] = {}
    for key in visual_keys:
        keys_by_shard.setdefault(base_weight_map[key], []).append(key)
    for shard_name, keys in sorted(keys_by_shard.items()):
        with safe_open(args.base_dir / shard_name, framework="pt", device="cpu") as handle:
            for key in keys:
                visual_tensors[key] = handle.get_tensor(key)

    visual_file = args.output_dir / "visual-model.safetensors"
    save_file(visual_tensors, visual_file, metadata={"format": "pt"})
    del visual_tensors

    trained_copy = args.output_dir / "trained-model.safetensors"
    shutil.copy2(trained_file, trained_copy)
    restored_keys = set(visual_keys)
    overlap = trained_keys & restored_keys
    if overlap:
        raise ValueError(f"trained and restored keys overlap: {sorted(overlap)[:5]}")
    weight_map = {key: trained_copy.name for key in sorted(trained_keys)}
    weight_map.update({key: visual_file.name for key in visual_keys})
    index = {
        "metadata": {
            "total_size": trained_copy.stat().st_size + visual_file.stat().st_size,
            "trained_key_count": len(trained_keys),
            "restored_visual_key_count": len(visual_keys),
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
            destination = args.output_dir / name
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            shutil.copy2(source, destination)

    manifest = {
        "trained_checkpoint": str(args.trained_dir.resolve()),
        "base_checkpoint": str(args.base_dir.resolve()),
        "base_model_revision": base_revision,
        "trained_key_count": len(trained_keys),
        "restored_visual_key_count": len(visual_keys),
        "visual_shard_bytes": visual_file.stat().st_size,
        "trained_model_sha256": sha256_file(trained_copy),
        "visual_model_sha256": sha256_file(visual_file),
        "output_index_sha256": sha256_file(args.output_dir / "model.safetensors.index.json"),
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
