import argparse
import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.level1.report.build_complete_vlm_checkpoint import build_checkpoint, weight_layout, sha256_file
from scripts.level1.train.validate_model_checkpoint import validate_model_checkpoint, CheckpointValidationError


@pytest.fixture(autouse=True)
def explicit_synthetic_architecture(monkeypatch):
    # These tiny named tensors exercise packaging, not a real HF architecture.
    monkeypatch.setattr(
        "scripts.level1.report.build_complete_vlm_checkpoint.validate_model_checkpoint",
        lambda *args, **kwargs: {"architecture_verified": False, "fixture": True},
    )


def checkpoint(path, tensors, *, sharded=False):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "test", "_commit_hash": "base123"}))
    (path / "tokenizer_config.json").write_text("{}")
    if sharded:
        index = {}
        for i, (name, value) in enumerate(tensors.items()):
            filename = f"part-{i}.safetensors"
            save_file({name: value}, path / filename)
            index[name] = filename
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    else:
        save_file(tensors, path / "model.safetensors")
    return path


def export(base, trained, output, **overrides):
    options = dict(trained_dir=trained, base_dir=base, output_dir=output,
                   visual_prefix="model.visual.", restore_all_missing=False,
                   base_revision=None, expected_base_revision="base123")
    options.update(overrides)
    return build_checkpoint(argparse.Namespace(**options))


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_single_sharded_and_already_complete_export(tmp_path, sharded, complete):
    tensors = {"model.language.weight": torch.ones(2), "model.visual.weight": torch.zeros(3)}
    base = checkpoint(tmp_path / "base", tensors, sharded=True)
    trained_tensors = {"model.language.weight": torch.full((2,), 2.0)}
    if complete:
        trained_tensors["model.visual.weight"] = tensors["model.visual.weight"]
    trained = checkpoint(tmp_path / "trained", trained_tensors, sharded=sharded)
    output = tmp_path / "output"
    if not complete:
        with pytest.raises(ValueError, match="frozen-weight provenance"):
            export(base, trained, output)
        assert not output.exists()
        return
    manifest = export(base, trained, output)
    assert set(weight_layout(output)[0]) == set(tensors)
    assert manifest["restored_visual_key_count"] == (0 if complete else 1)
    assert not any(path.is_symlink() for path in output.rglob("*"))
    for name, checksum in manifest["files_sha256"].items():
        assert sha256_file(output / name) == checksum
    assert manifest["validation"] == "structural_export_only_not_model_load_or_gameplay"
    result = validate_model_checkpoint(output, load_transformers_metadata=False, validate_architecture=False)
    assert result["export_manifest_sha256"] == sha256_file(output / "merge_manifest.json")
    assert result["actual_model_load_verified"] is False
    with pytest.raises(FileExistsError):
        export(base, trained, output)


@pytest.mark.parametrize("restore_all", [False, True])
def test_missing_trainable_weights_never_restored(tmp_path, restore_all):
    base = checkpoint(tmp_path / "base", {"model.language.weight": torch.ones(2), "model.visual.weight": torch.zeros(3)})
    trained = checkpoint(tmp_path / "trained", {"model.visual.weight": torch.zeros(3)})
    with pytest.raises(ValueError, match="non-visual"):
        export(base, trained, tmp_path / "out", restore_all_missing=restore_all)
    assert not (tmp_path / "out").exists()


def test_shape_mismatch_rejected(tmp_path):
    base = checkpoint(tmp_path / "base", {"model.language.weight": torch.ones(2)})
    trained = checkpoint(tmp_path / "trained", {"model.language.weight": torch.ones(3)})
    with pytest.raises(ValueError, match="shape or dtype mismatch"):
        export(base, trained, tmp_path / "out")


def test_revision_mismatch_rejected(tmp_path):
    base = checkpoint(tmp_path / "base", {"model.language.weight": torch.ones(2)})
    trained = checkpoint(tmp_path / "trained", {"model.language.weight": torch.ones(2)})
    with pytest.raises(ValueError, match="revision mismatch"):
        export(base, trained, tmp_path / "out", expected_base_revision="different")
    with pytest.raises(ValueError, match="CLI label conflicts"):
        export(base, trained, tmp_path / "out", base_revision="different", expected_base_revision="different")


def test_unsafe_index_path_rejected(tmp_path):
    root = tmp_path / "bad"
    root.mkdir()
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="unsafe"):
        weight_layout(root)


def test_only_declared_unused_base_auxiliary_tensors_can_be_omitted(tmp_path, monkeypatch):
    base = checkpoint(tmp_path / "base", {
        "model.language.weight": torch.ones(2), "mtp.fc.weight": torch.zeros(3),
    })
    trained = checkpoint(tmp_path / "trained", {"model.language.weight": torch.ones(2)})
    monkeypatch.setattr(
        "scripts.level1.report.build_complete_vlm_checkpoint.validate_model_checkpoint",
        lambda *args, **kwargs: {
            "architecture_verified": False, "fixture": True,
            "ignored_auxiliary_tensors": ["mtp.fc.weight"],
        },
    )
    manifest = export(base, trained, tmp_path / "out")
    assert manifest["omitted_base_auxiliary_tensors"] == ["mtp.fc.weight"]
    assert manifest["restored_base_key_count"] == 0


def test_export_validator_rejects_changed_tensor_bytes(tmp_path):
    tensors = {"model.language.weight": torch.ones(2)}
    base = checkpoint(tmp_path / "base", tensors)
    trained = checkpoint(tmp_path / "trained", tensors)
    output = tmp_path / "out"
    export(base, trained, output)
    save_file({"model.language.weight": torch.zeros(2)}, output / "trained-model.safetensors")
    with pytest.raises(CheckpointValidationError, match="checksum mismatch"):
        validate_model_checkpoint(output, load_transformers_metadata=False, validate_architecture=False)


def test_export_manifest_cannot_omit_config_even_if_rehashed(tmp_path):
    tensors = {"model.language.weight": torch.ones(2)}
    base = checkpoint(tmp_path / "base", tensors)
    trained = checkpoint(tmp_path / "trained", tensors)
    output = tmp_path / "out"
    manifest = export(base, trained, output)
    del manifest["files_sha256"]["config.json"]
    (output / "merge_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    checksum = sha256_file(output / "merge_manifest.json")
    (output / "merge_manifest.sha256").write_text(checksum + "  merge_manifest.json\n", encoding="ascii")
    with pytest.raises(CheckpointValidationError, match="does not cover"):
        validate_model_checkpoint(output, load_transformers_metadata=False, validate_architecture=False)
