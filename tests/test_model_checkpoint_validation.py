import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.level1.train.validate_model_checkpoint import (
    CheckpointValidationError,
    validate_model_checkpoint,
)


def _write_config(root: Path) -> None:
    (root / "config.json").write_text("{}\n", encoding="utf-8")


def _write_safetensors(path: Path, tensor_name: str) -> None:
    save_file({tensor_name: np.zeros((1,), dtype=np.float32)}, path)


def _validate(root: Path) -> dict[str, object]:
    # These fixtures exercise storage format only, not a real model schema.
    return validate_model_checkpoint(
        root, load_transformers_metadata=False, validate_architecture=False
    )


def test_accepts_nonempty_single_file_checkpoint(tmp_path: Path) -> None:
    _write_config(tmp_path)
    _write_safetensors(tmp_path / "model.safetensors", "model.a")
    result = _validate(tmp_path)
    assert result["weight_layout"] == "model.safetensors"
    assert result["weight_file_count"] == 1


def test_accepts_index_only_when_every_referenced_shard_is_nonempty(
    tmp_path: Path,
) -> None:
    _write_config(tmp_path)
    index = {
        "weight_map": {
            "model.a": "model-00001-of-00002.safetensors",
            "model.b": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    for tensor_name, shard in index["weight_map"].items():
        _write_safetensors(tmp_path / shard, tensor_name)
    result = _validate(tmp_path)
    assert result["weight_layout"] == "model.safetensors.index.json"
    assert result["weight_file_count"] == 2


@pytest.mark.parametrize("failure", ["missing", "empty"])
def test_rejects_missing_or_empty_index_shard(tmp_path: Path, failure: str) -> None:
    _write_config(tmp_path)
    shard = "model-00001-of-00001.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.a": shard}}), encoding="utf-8"
    )
    if failure == "empty":
        (tmp_path / shard).touch()
    with pytest.raises(CheckpointValidationError, match="missing or empty"):
        _validate(tmp_path)


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("config.json", "not-json", "invalid model config"),
        ("model.safetensors.index.json", "not-json", "invalid weight index"),
        ("model.safetensors.index.json", "{}", "weight_map"),
    ],
)
def test_rejects_invalid_json_or_empty_weight_map(
    tmp_path: Path,
    filename: str,
    content: str,
    message: str,
) -> None:
    _write_config(tmp_path)
    if filename != "config.json":
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / filename).write_text(content, encoding="utf-8")
    with pytest.raises(CheckpointValidationError, match=message):
        _validate(tmp_path)


def test_rejects_index_path_traversal(tmp_path: Path) -> None:
    _write_config(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.a": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointValidationError, match="invalid shard filename"):
        _validate(tmp_path)


def test_rejects_shard_type_that_disagrees_with_index(tmp_path: Path) -> None:
    _write_config(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.a": "pytorch_model-00001.bin"}}),
        encoding="utf-8",
    )
    (tmp_path / "pytorch_model-00001.bin").write_bytes(b"weights")
    with pytest.raises(CheckpointValidationError, match="must end with"):
        _validate(tmp_path)


def test_rejects_corrupt_safetensors_file(tmp_path: Path) -> None:
    _write_config(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"not-safetensors")
    with pytest.raises(CheckpointValidationError, match="invalid safetensors"):
        _validate(tmp_path)


def test_rejects_index_tensor_missing_from_referenced_shard(tmp_path: Path) -> None:
    _write_config(tmp_path)
    shard = "model-00001-of-00001.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.expected": shard}}), encoding="utf-8"
    )
    _write_safetensors(tmp_path / shard, "model.other")
    with pytest.raises(CheckpointValidationError, match="tensors absent"):
        _validate(tmp_path)


def test_launcher_uses_fail_closed_checkpoint_validator() -> None:
    launcher = (
        Path(__file__).parents[1] / "scripts/level1/train/run_level1_training.sh"
    ).read_text(encoding="utf-8")
    assert "validate_model_checkpoint.py" in launcher
    assert 'MODEL_PATH="$(cd "${MODEL_PATH}" && pwd -P)"' in launcher
    assert '[[ ! -f "${MODEL_PATH}/model.safetensors.index.json"' not in launcher


def _tiny_qwen35(root: Path, *, tied: bool = False) -> dict:
    """Real Transformers Qwen3.5 architecture, scaled down for CPU fixtures."""
    from transformers import AutoModelForImageTextToText, Qwen3_5Config

    config = Qwen3_5Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "tie_word_embeddings": tied,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
        },
        tie_word_embeddings=tied,
    )
    model = AutoModelForImageTextToText.from_config(config, attn_implementation="eager")
    model.tie_weights()
    config.save_pretrained(root)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _save_real_weights(root: Path, state: dict, *, sharded: bool = False) -> None:
    from safetensors.torch import save_file as save_torch

    if not sharded:
        save_torch(state, root / "model.safetensors")
        return
    parts = (dict(list(state.items())[::2]), dict(list(state.items())[1::2]))
    weight_map = {}
    for number, tensors in enumerate(parts):
        name = f"part-{number}.safetensors"
        save_torch(tensors, root / name)
        weight_map.update({key: name for key in tensors})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


@pytest.mark.parametrize("sharded", [False, True])
def test_real_qwen35_complete_architecture_is_verified_without_forward(
    tmp_path, sharded
):
    state = _tiny_qwen35(tmp_path)
    _save_real_weights(tmp_path, state, sharded=sharded)
    result = validate_model_checkpoint(tmp_path, load_transformers_metadata=False)
    assert result["architecture_verified"] is True
    assert result["architecture_validation"] == "offline-meta-keys-shapes-v1"
    assert result["model_class"] == "Qwen3_5ForConditionalGeneration"
    assert (
        result["expected_tensor_count"] == result["stored_tensor_count"] == len(state)
    )
    assert result["actual_model_load_verified"] is False


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("kind", ["missing", "extra", "shape"])
def test_real_qwen35_rejects_corrupt_architecture_even_with_consistent_index(
    tmp_path, sharded, kind
):
    import torch

    state = _tiny_qwen35(tmp_path)
    key = "model.language_model.layers.0.mlp.gate_proj.weight"
    message = {
        "missing": "missing required architecture tensor",
        "extra": "unexpected tensor",
        "shape": "shape mismatch",
    }[kind]
    if kind == "missing":
        del state[key]
    elif kind == "extra":
        state["invented.weight"] = torch.ones(1)
    else:
        state[key] = torch.zeros(1)
    _save_real_weights(tmp_path, state, sharded=sharded)
    with pytest.raises(CheckpointValidationError, match=message):
        validate_model_checkpoint(tmp_path, load_transformers_metadata=False)


@pytest.mark.parametrize(
    "omitted", ["lm_head.weight", "model.language_model.embed_tokens.weight"]
)
def test_real_tied_embeddings_may_omit_exactly_one_actual_alias(tmp_path, omitted):
    state = _tiny_qwen35(tmp_path, tied=True)
    del state[omitted]
    _save_real_weights(tmp_path, state)
    result = validate_model_checkpoint(tmp_path, load_transformers_metadata=False)
    assert result["tied_tensor_omissions"] == [omitted]


def test_missing_both_tied_embeddings_is_not_excused(tmp_path):
    state = _tiny_qwen35(tmp_path, tied=True)
    del state["lm_head.weight"]
    del state["model.language_model.embed_tokens.weight"]
    _save_real_weights(tmp_path, state)
    with pytest.raises(
        CheckpointValidationError, match="missing required architecture tensor"
    ):
        validate_model_checkpoint(tmp_path, load_transformers_metadata=False)


def test_untied_embedding_omission_is_not_excused_by_declared_tie_patterns(tmp_path):
    state = _tiny_qwen35(tmp_path, tied=False)
    del state["lm_head.weight"]
    _save_real_weights(tmp_path, state)
    with pytest.raises(
        CheckpointValidationError,
        match="missing required architecture tensor: lm_head.weight",
    ):
        validate_model_checkpoint(tmp_path, load_transformers_metadata=False)


def test_unsupported_architecture_fails_closed_by_default(tmp_path):
    _write_config(tmp_path)
    _write_safetensors(tmp_path / "model.safetensors", "arbitrary.weight")
    with pytest.raises(
        CheckpointValidationError, match="cannot verify checkpoint architecture"
    ):
        validate_model_checkpoint(tmp_path, load_transformers_metadata=False)
    assert _validate(tmp_path)["architecture_verified"] is False


def test_ambiguous_layouts_are_rejected(tmp_path):
    _write_config(tmp_path)
    _write_safetensors(tmp_path / "model.safetensors", "model.a")
    (tmp_path / "pytorch_model.bin").write_bytes(b"ambiguous")
    with pytest.raises(CheckpointValidationError, match="ambiguous weight layouts"):
        _validate(tmp_path)


def test_unmapped_shard_tensor_is_rejected(tmp_path):
    _write_config(tmp_path)
    shard = "part.safetensors"
    save_file({"model.a": np.zeros(1), "model.extra": np.zeros(1)}, tmp_path / shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.a": shard}})
    )
    with pytest.raises(CheckpointValidationError, match="not mapped to it"):
        _validate(tmp_path)


def test_pytorch_bin_architecture_is_checked_on_meta(tmp_path):
    import torch

    state = _tiny_qwen35(tmp_path)
    torch.save(state, tmp_path / "pytorch_model.bin")
    result = validate_model_checkpoint(tmp_path, load_transformers_metadata=False)
    assert result["architecture_verified"] is True
    assert result["weight_layout"] == "pytorch_model.bin"


def test_full_size_qwen35_schema_constructs_only_meta_tensors(tmp_path):
    from transformers import Qwen3_5Config
    from scripts.level1.train.validate_model_checkpoint import _expected_architecture

    # Use the library's full-size defaults, not the tiny fixture; this would be
    # far too large to materialize in an ordinary CPU unit test.
    Qwen3_5Config().save_pretrained(tmp_path)
    shapes, _, model_class = _expected_architecture(tmp_path)
    assert model_class == "Qwen3_5ForConditionalGeneration"
    assert len(shapes) > 500
    assert shapes["model.language_model.embed_tokens.weight"][0] > 200_000


def test_safetensors_validation_never_materializes_checkpoint_tensors(
    tmp_path, monkeypatch
):
    import safetensors

    state = _tiny_qwen35(tmp_path)
    _save_real_weights(tmp_path, state)
    original_open = safetensors.safe_open

    class HeaderOnly:
        def __init__(self, *args, **kwargs):
            self.handle = original_open(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def keys(self):
            return self.handle.keys()

        def get_slice(self, key):
            return self.handle.get_slice(key)

        def get_tensor(self, key):
            pytest.fail(f"validator materialized a tensor: {key}")

    monkeypatch.setattr(safetensors, "safe_open", HeaderOnly)
    result = validate_model_checkpoint(tmp_path, load_transformers_metadata=False)
    assert result["architecture_verified"] is True


def test_declared_architecture_mismatch_is_rejected(tmp_path):
    state = _tiny_qwen35(tmp_path)
    _save_real_weights(tmp_path, state)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["Qwen3_5ForCausalLM"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(CheckpointValidationError, match="config declares"):
        validate_model_checkpoint(tmp_path, load_transformers_metadata=False)


def test_old_bin_serialization_has_no_unsafe_or_allocating_fallback(tmp_path):
    import torch

    _write_config(tmp_path)
    torch.save(
        {"model.a": torch.zeros(1)},
        tmp_path / "pytorch_model.bin",
        _use_new_zipfile_serialization=False,
    )
    with pytest.raises(
        CheckpointValidationError, match="cannot safely inspect weight file"
    ):
        _validate(tmp_path)


def test_real_vlm_accepts_only_official_auxiliary_mtp_tensors(tmp_path):
    import torch

    state = _tiny_qwen35(tmp_path)
    state["mtp.fc.weight"] = torch.zeros(2, 2)
    state["mtp.layers.0.norm.weight"] = torch.zeros(2)
    _save_real_weights(tmp_path, state)
    result = validate_model_checkpoint(tmp_path, load_transformers_metadata=False)
    assert result["ignored_auxiliary_tensors"] == [
        "mtp.fc.weight",
        "mtp.layers.0.norm.weight",
    ]
    assert result["stored_tensor_count"] == result["expected_tensor_count"] + 2


def test_vlm_does_not_apply_text_only_visual_ignore_rule(tmp_path):
    import torch

    state = _tiny_qwen35(tmp_path)
    state["model.visual.invented.weight"] = torch.zeros(1)
    _save_real_weights(tmp_path, state)
    with pytest.raises(
        CheckpointValidationError, match="unexpected tensor.*model.visual"
    ):
        validate_model_checkpoint(tmp_path, load_transformers_metadata=False)


def test_auxiliary_ignore_rules_never_excuse_missing_required_weights(
    tmp_path, monkeypatch
):
    from transformers import Qwen3_5ForConditionalGeneration

    state = _tiny_qwen35(tmp_path)
    del state["model.visual.patch_embed.proj.weight"]
    _save_real_weights(tmp_path, state)
    monkeypatch.setattr(
        Qwen3_5ForConditionalGeneration, "_keys_to_ignore_on_load_unexpected", [r".*"]
    )
    with pytest.raises(
        CheckpointValidationError, match="missing required architecture tensor"
    ):
        validate_model_checkpoint(tmp_path, load_transformers_metadata=False)


def test_gpu_visible_schema_uses_isolated_offline_child_without_parent_env_mutation(
    tmp_path, monkeypatch
):
    import os
    import sys
    from types import SimpleNamespace
    import torch
    from scripts.level1.train import validate_model_checkpoint as validator

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    observed = {}

    def child(command, **kwargs):
        observed.update(command=command, **kwargs)
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "expected_shapes": {
                        "input.weight": [3, 4],
                        "output.weight": [3, 4],
                    },
                    "tied_groups": [["input.weight", "output.weight"]],
                    "model_class": "Qwen3_5ForConditionalGeneration",
                    "ignored_unexpected_patterns": ["^mtp.*"],
                }
            ),
        )

    monkeypatch.setattr(validator.subprocess, "run", child)
    shapes, groups, model_class, patterns = validator.architecture_tensor_policy(
        tmp_path
    )
    assert shapes["input.weight"] == (3, 4)
    assert groups == [{"input.weight", "output.weight"}]
    assert patterns == ("^mtp.*",)
    assert model_class == "Qwen3_5ForConditionalGeneration"
    assert observed["command"][0] == sys.executable
    assert "--internal-architecture-schema" in observed["command"]
    assert all(
        observed["env"][name] == ""
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        )
    )
    assert observed["env"]["HF_HUB_OFFLINE"] == "1"
    assert observed["timeout"] == 90
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,5"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "2"


@pytest.mark.parametrize("failure", ["exit", "json", "schema", "timeout"])
def test_cpu_schema_child_failures_never_fall_back_to_gpu(
    tmp_path, monkeypatch, failure
):
    from types import SimpleNamespace
    import torch
    from scripts.level1.train import validate_model_checkpoint as validator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def child(*args, **kwargs):
        if failure == "timeout":
            raise validator.subprocess.TimeoutExpired("schema", 90)
        return SimpleNamespace(
            returncode=1 if failure == "exit" else 0,
            stderr="non-meta parameter",
            stdout="not json" if failure == "json" else "{}",
        )

    monkeypatch.setattr(validator.subprocess, "run", child)
    monkeypatch.setattr(
        validator,
        "_architecture_tensor_policy_local",
        lambda _: pytest.fail("unsafe parent fallback"),
    )
    with pytest.raises(CheckpointValidationError, match="CPU-only schema"):
        validator.architecture_tensor_policy(tmp_path)


def test_real_cpu_schema_subprocess_roundtrip_for_qwen35(tmp_path, monkeypatch):
    import torch
    from scripts.level1.train import validate_model_checkpoint as validator

    state = _tiny_qwen35(tmp_path)
    # Simulate a GPU-visible caller, but execute the real separate interpreter.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    shapes, _, model_class, patterns = validator.architecture_tensor_policy(tmp_path)
    assert set(shapes) == set(state)
    assert model_class == "Qwen3_5ForConditionalGeneration"
    assert any(validator.re.search(pattern, "mtp.fc.weight") for pattern in patterns)
