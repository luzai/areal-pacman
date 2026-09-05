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


@pytest.mark.parametrize("sharded", [False, True])
def test_explicit_bfloat16_policy_copies_saved_bytes_and_records_dtype_changes(tmp_path, sharded):
    """Precision declaration permits packaging, not a claim of learned changes."""
    tensors = {"model.language.weight": torch.tensor([1.001, 2.003]),
               "model.visual.weight": torch.ones(3, dtype=torch.bfloat16),
               "model.position": torch.arange(3)}
    base = checkpoint(tmp_path / "base", tensors)
    saved = {key: value.bfloat16() if value.is_floating_point() else value
             for key, value in tensors.items()}
    trained = checkpoint(tmp_path / "trained", saved, sharded=sharded)
    output = tmp_path / "out"
    manifest = export(base, trained, output, expected_saved_dtype="bfloat16")
    assert manifest["trained_storage_dtype_policy"] == "explicit_bfloat16_save"
    assert manifest["trained_dtype_transitions"] == [{
        "key": "model.language.weight", "shape": [2],
        "base_dtype": "F32", "saved_dtype": "BF16",
    }]
    assert manifest["trained_weight_bytes_modified"] is False
    assert manifest["restored_base_key_count"] == 0
    mapping, _ = weight_layout(output)
    trained_mapping, _ = weight_layout(trained)
    for key, output_shard in mapping.items():
        assert sha256_file(output / output_shard) == sha256_file(trained / trained_mapping[key])
    assert manifest["validation"] == "structural_export_only_not_model_load_or_gameplay"


def test_storage_dtype_change_without_explicit_policy_is_rejected(tmp_path):
    """Default behavior must not silently accept a changed storage precision."""
    base = checkpoint(tmp_path / "base", {"weight": torch.ones(2)})
    trained = checkpoint(tmp_path / "trained", {"weight": torch.ones(2, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="shape or dtype mismatch"):
        export(base, trained, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("base_dtype,saved_dtype", [
    (torch.float16, torch.bfloat16), (torch.float64, torch.bfloat16),
    (torch.float32, torch.float32), (torch.float32, torch.float16),
    (torch.int32, torch.bfloat16), (torch.float32, torch.int32),
    (torch.int32, torch.int64),
])
def test_bfloat16_policy_rejects_unexpected_precisions(tmp_path, base_dtype, saved_dtype):
    """The explicit policy is not a general dtype-mismatch bypass."""
    base = checkpoint(tmp_path / "base", {"weight": torch.ones(2, dtype=base_dtype)})
    trained = checkpoint(tmp_path / "trained", {"weight": torch.ones(2, dtype=saved_dtype)})
    with pytest.raises(ValueError, match="dtype mismatch"):
        export(base, trained, tmp_path / "out", expected_saved_dtype="bfloat16")
    assert not (tmp_path / "out").exists()


def test_bfloat16_policy_never_permits_shape_change(tmp_path):
    base = checkpoint(tmp_path / "base", {"weight": torch.ones(2)})
    trained = checkpoint(tmp_path / "trained", {"weight": torch.ones(3, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match="shape or dtype mismatch"):
        export(base, trained, tmp_path / "out", expected_saved_dtype="bfloat16")


def test_unknown_save_precision_policy_is_rejected(tmp_path):
    base = checkpoint(tmp_path / "base", {"weight": torch.ones(2)})
    trained = checkpoint(tmp_path / "trained", {"weight": torch.ones(2)})
    with pytest.raises(ValueError, match="unsupported expected saved dtype"):
        export(base, trained, tmp_path / "out", expected_saved_dtype="float16")


def qwen_serialization_pair(tmp_path):
    """Real AutoConfig parsing with tiny storage-only tensors; no model load."""
    import copy

    import transformers

    base = checkpoint(tmp_path / "base", {"weight": torch.ones(2)})
    trained = checkpoint(tmp_path / "trained", {"weight": torch.ones(2, dtype=torch.bfloat16)})
    base_config = {
        "model_type": "qwen3_5", "_commit_hash": "base123",
        "transformers_version": "4.57.0.dev0", "tie_word_embeddings": False,
        "text_config": {"model_type": "qwen3_5_text"},
        "vision_config": {"model_type": "qwen3_5"},
    }
    trained_config = copy.deepcopy(base_config)
    trained_config["transformers_version"] = transformers.__version__
    trained_config["text_config"].update(
        bos_token_id=None, pad_token_id=None, partial_rotary_factor=0.25,
        tie_word_embeddings=False,
    )
    trained_config["vision_config"]["model_type"] = "qwen3_5_vision"
    for root, config in ((base, base_config), (trained, trained_config)):
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return base, trained, trained_config


def test_explicit_qwen_config_comparison_preserves_trained_metadata(tmp_path):
    base, trained, _ = qwen_serialization_pair(tmp_path)
    original_config = (trained / "config.json").read_bytes()
    manifest = export(base, trained, tmp_path / "out", expected_saved_dtype="bfloat16",
                      config_comparison="qwen3_5")
    comparison = manifest["config_comparison"]
    assert comparison["policy"] == "qwen3_5"
    assert len(comparison["approved_raw_differences"]) == 6
    assert comparison["normalized_ignored_fields"] == ["_name_or_path"]
    assert len(comparison["normalized_config_sha256"]) == 64
    assert comparison["actual_model_load_verified"] is False
    assert (tmp_path / "out" / "config.json").read_bytes() == original_config
    assert manifest["trained_config_sha256"] == sha256_file(trained / "config.json")


def test_qwen_serialization_changes_are_rejected_by_default(tmp_path):
    base, trained, _ = qwen_serialization_pair(tmp_path)
    with pytest.raises(ValueError, match="config mismatch"):
        export(base, trained, tmp_path / "out", expected_saved_dtype="bfloat16")


@pytest.mark.parametrize("path,value", [
    (("max_length",), 23),
    (("attn_implementation",), "eager"),
    (("text_config", "max_length"), 23),
    (("text_config", "model_type"), "bogus"),
    (("vision_config", "model_type"), "bogus"),
    (("text_config", "hidden_size"), 64),
    (("text_config", "partial_rotary_factor"), 0.5),
    (("text_config", "tie_word_embeddings"), True),
    (("text_config", "pad_token_id"), 0),
    (("unknown_user_field",), "changed"),
])
def test_qwen_policy_rejects_unapproved_raw_changes_even_if_parser_discards_them(tmp_path, path, value):
    base, trained, config = qwen_serialization_pair(tmp_path)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    (trained / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unapproved raw config difference"):
        export(base, trained, tmp_path / "out", expected_saved_dtype="bfloat16",
               config_comparison="qwen3_5")
    assert not (tmp_path / "out").exists()


def test_qwen_policy_requires_matching_transformers_version(tmp_path):
    base, trained, config = qwen_serialization_pair(tmp_path)
    config["transformers_version"] = "0.0.0"
    (trained / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="version must match"):
        export(base, trained, tmp_path / "out", expected_saved_dtype="bfloat16",
               config_comparison="qwen3_5")


def test_qwen_policy_rejects_missing_trained_config(tmp_path):
    base, trained, _ = qwen_serialization_pair(tmp_path)
    (trained / "config.json").unlink()
    with pytest.raises(ValueError, match="requires the trained config"):
        export(base, trained, tmp_path / "out", expected_saved_dtype="bfloat16",
               config_comparison="qwen3_5")


@pytest.mark.parametrize("which", ["base", "trained"])
def test_config_change_after_validation_is_rejected(tmp_path, monkeypatch, which):
    import scripts.level1.report.build_complete_vlm_checkpoint as exporter

    base, trained, _ = qwen_serialization_pair(tmp_path)
    original = exporter.validate_config_comparison

    def validate_then_change(*args, **kwargs):
        result = original(*args, **kwargs)
        target = (base if which == "base" else trained) / "config.json"
        config = json.loads(target.read_text(encoding="utf-8"))
        config["text_config"]["hidden_size"] = 64
        target.write_text(json.dumps(config), encoding="utf-8")
        return result

    monkeypatch.setattr(exporter, "validate_config_comparison", validate_then_change)
    with pytest.raises(ValueError, match="source config changed"):
        export(base, trained, tmp_path / "out", expected_saved_dtype="bfloat16",
               config_comparison="qwen3_5")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutate_source", [False, True])
def test_config_mutation_during_copy_never_publishes_output(tmp_path, monkeypatch, mutate_source):
    import scripts.level1.report.build_complete_vlm_checkpoint as exporter

    base, trained, _ = qwen_serialization_pair(tmp_path)
    original = exporter.shutil.copy2

    def copy_then_change(source, destination, *args, **kwargs):
        result = original(source, destination, *args, **kwargs)
        if source.name == "config.json":
            target = source if mutate_source else destination
            config = json.loads(target.read_text(encoding="utf-8"))
            config["text_config"]["hidden_size"] = 64
            target.write_text(json.dumps(config), encoding="utf-8")
        return result

    monkeypatch.setattr(exporter.shutil, "copy2", copy_then_change)
    with pytest.raises(ValueError, match="source config changed|copied config differs"):
        export(base, trained, tmp_path / "out", expected_saved_dtype="bfloat16",
               config_comparison="qwen3_5")
    assert not (tmp_path / "out").exists()
