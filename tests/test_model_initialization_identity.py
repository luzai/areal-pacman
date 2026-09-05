"""Content-bound model initialization; fake metadata loaders are explicit."""

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.level1.dataset import write_level1_manifest as writer


@pytest.fixture
def model(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    save_file({"weight": np.zeros(3, dtype=np.float32)}, root / "model.safetensors")
    for name in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
    ):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    monkeypatch.setattr(
        writer,
        "validate_model_checkpoint",
        lambda path: {
            "architecture_verified": True,
            "offline_metadata": True,
            "actual_model_load_verified": False,
        },
    )
    return root


def test_identity_binds_runtime_assets_not_path_or_unrelated_files(model, tmp_path):
    first = writer.model_initialization_identity(model)
    clone = tmp_path / "copy"
    shutil.copytree(model, clone)
    (clone / "README.md").write_text("not used to load the model")
    (clone / "optimizer.pt").write_bytes(b"not part of model initialization")
    second = writer.model_initialization_identity(clone)
    assert first["identity_sha256"] == second["identity_sha256"]
    assert first["path"] != second["path"]
    assert first["files_sha256"] == second["files_sha256"]
    assert len(first["files_sha256"]) == 6
    content = {key: first[key] for key in ("identity_contract", "files_sha256")}
    expected = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert first["identity_sha256"] == expected
    for key in (
        "path_is_hf_revision",
        "hf_revision_verified",
        "training_lineage_verified",
        "training_completion_verified",
    ):
        assert first[key] is False


@pytest.mark.parametrize(
    "name",
    [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "chat_template.jinja",
    ],
)
def test_changing_any_runtime_asset_changes_identity(model, name):
    before = writer.model_initialization_identity(model)
    if name.endswith(".safetensors"):
        save_file({"weight": np.ones(3, dtype=np.float32)}, model / name)
    elif name.endswith(".json"):
        (model / name).write_text('{"changed": true}')
    else:
        (model / name).write_text("changed template")
    assert (
        writer.model_initialization_identity(model)["identity_sha256"]
        != before["identity_sha256"]
    )


def test_referenced_tokenizer_assets_and_template_directory_are_bound(model):
    (model / "vocabulary.data").write_text("local vocab")
    (model / "tokenizer_config.json").write_text('{"vocab_file":"vocabulary.data"}')
    (model / "chat_templates").mkdir()
    (model / "chat_templates" / "image.jinja").write_text("image template")
    identity = writer.model_initialization_identity(model)
    assert "vocabulary.data" in identity["files_sha256"]
    assert "chat_templates/image.jinja" in identity["files_sha256"]


def test_export_provenance_is_recorded_separately_from_portable_identity(model):
    before = writer.model_initialization_identity(model)
    (model / "merge_manifest.json").write_text('{"trained_checkpoint":"/old/place"}')
    (model / "merge_manifest.sha256").write_text("mocked validator checks the seal")
    after = writer.model_initialization_identity(model)
    assert before["identity_sha256"] == after["identity_sha256"]
    assert (
        after["merge_manifest_sha256"]
        == hashlib.sha256((model / "merge_manifest.json").read_bytes()).hexdigest()
    )
    assert set(after["export_provenance_sha256"]) == {
        "merge_manifest.json",
        "merge_manifest.sha256",
    }


def test_default_architecture_and_metadata_validation_is_required(model, monkeypatch):
    called = []

    def incomplete(path):
        called.append(path)
        return {"architecture_verified": False, "offline_metadata": True}

    monkeypatch.setattr(writer, "validate_model_checkpoint", incomplete)
    with pytest.raises(ValueError, match="requires verified architecture"):
        writer.model_initialization_identity(model)
    assert called == [model.resolve()]


def test_missing_shard_fails_before_identity(model):
    (model / "model.safetensors").unlink()
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}})
    )
    with pytest.raises(ValueError, match="missing initialization file"):
        writer.model_initialization_identity(model)


@pytest.mark.parametrize("target", ["model.safetensors", "tokenizer.json"])
def test_external_file_symlink_is_rejected(model, tmp_path, target):
    external = tmp_path / "external"
    shutil.copy2(model / target, external)
    (model / target).unlink()
    try:
        (model / target).symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="externally linked"):
        writer.model_initialization_identity(model)


def test_external_metadata_reference_is_rejected(model, tmp_path):
    outside = tmp_path / "outside.model"
    outside.write_text("not inside bundle")
    (model / "tokenizer_config.json").write_text(
        json.dumps({"vocab_file": str(outside)})
    )
    with pytest.raises(ValueError, match="externally linked"):
        writer.model_initialization_identity(model)


def test_validator_time_mutation_is_rejected(model, monkeypatch):
    def mutate(path):
        (path / "tokenizer.json").write_text('{"mutated":true}')
        return {"architecture_verified": True, "offline_metadata": True}

    monkeypatch.setattr(writer, "validate_model_checkpoint", mutate)
    with pytest.raises(ValueError, match="changed before hashing"):
        writer.model_initialization_identity(model)


def test_earlier_file_mutation_during_later_hash_is_rejected(model, monkeypatch):
    original = writer._stable_file_sha256

    def mutate(root, name, before):
        result = original(root, name, before)
        if name == "tokenizer_config.json":
            (root / "config.json").write_text('{"changed_after_hash":true}')
        return result

    monkeypatch.setattr(writer, "_stable_file_sha256", mutate)
    with pytest.raises(ValueError, match="changed during validation/hashing"):
        writer.model_initialization_identity(model)


def test_new_runtime_asset_during_hash_is_rejected(model, monkeypatch):
    original = writer._stable_file_sha256

    def mutate(root, name, before):
        result = original(root, name, before)
        (root / "generation_config.json").write_text("{}")
        return result

    monkeypatch.setattr(writer, "_stable_file_sha256", mutate)
    with pytest.raises(ValueError, match="inventory changed"):
        writer.model_initialization_identity(model)


def _mock_cli_inputs(tmp_path, model, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "recipe_version": "maapacman-level1-ghostdoor-v3",
                "reward_recipe_version": writer.REWARD_RECIPE_VERSION,
                "environment": {"ghost_mode": "normal", "max_steps": 512},
            }
        )
    )
    revisions = {
        name: {"commit": "1" * 40, "dirty": False}
        for name in ("areal-pacman", "pacman-python", "AReaL")
    }
    spec = SimpleNamespace(
        api_version="3.0",
        env_id="test",
        level_revision="a",
        renderer_revision="b",
        ruleset_revision="c",
        action_tokens=("U", "D", "L", "R"),
    )
    monkeypatch.setattr(
        writer,
        "PygamePacmanEnv",
        lambda _: SimpleNamespace(
            spec=spec,
            provenance={"maapacman_commit": "1" * 40, "maapacman_dirty": False},
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(writer, "repository_revisions", lambda: revisions)
    monkeypatch.setattr(writer, "environment_metadata", lambda _: {})
    monkeypatch.setattr(
        writer,
        "validate_prepared_dataset_manifest",
        lambda *a, **k: {
            "environment": {"ghost_mode": "normal", "ruleset_revision": "c"},
            "max_steps": 512,
            "splits": {"train": {"seeds": [28]}},
            "training_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        },
    )
    output = tmp_path / "run"
    monkeypatch.setattr(
        writer.sys,
        "argv",
        [
            "write_level1_manifest.py",
            "--artifact-root",
            str(output),
            "--model-path",
            str(model),
            "--dataset-manifest",
            str(tmp_path / "dataset.json"),
            "--config",
            str(config),
        ],
    )
    return output


def test_cli_publishes_content_bound_initialization_identity(
    model, tmp_path, monkeypatch, capsys
):
    output = _mock_cli_inputs(tmp_path, model, monkeypatch)
    writer.main()
    manifest = json.loads((output / "manifest.json").read_text())
    identity = manifest["model_initialization_identity"]
    assert (
        manifest["actor_initialization_identity_sha256"] == identity["identity_sha256"]
    )
    assert (
        manifest["reference_initialization_identity_sha256"]
        == identity["identity_sha256"]
    )
    assert manifest["model_revision_kind"] == "local_checkpoint_path_not_hf_revision"
    assert json.loads(capsys.readouterr().out) == manifest


def test_cli_validation_failure_writes_no_success_manifest(
    model, tmp_path, monkeypatch, capsys
):
    output = _mock_cli_inputs(tmp_path, model, monkeypatch)

    def fail(_):
        raise ValueError("missing required architecture tensor")

    monkeypatch.setattr(writer, "validate_model_checkpoint", fail)
    with pytest.raises(ValueError, match="missing required architecture"):
        writer.main()
    assert not (output / "manifest.json").exists()
    assert capsys.readouterr().out == ""


def test_atomic_publication_preserves_existing_manifest(tmp_path):
    writer._publish_manifest(tmp_path, {"original": True})
    with pytest.raises(FileExistsError):
        writer._publish_manifest(tmp_path, {"replacement": True})
    assert json.loads((tmp_path / "manifest.json").read_text()) == {"original": True}
    assert not list(tmp_path.glob(".manifest-*.tmp"))


def test_failed_atomic_publication_leaves_no_partial_manifest(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(writer.os, "link", fail)
    with pytest.raises(OSError, match="publication failure"):
        writer._publish_manifest(tmp_path, {"valid": True})
    assert not (tmp_path / "manifest.json").exists()
    assert not list(tmp_path.glob(".manifest-*.tmp"))
