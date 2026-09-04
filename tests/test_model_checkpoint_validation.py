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
    return validate_model_checkpoint(root, load_transformers_metadata=False)


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
        Path(__file__).parents[1]
        / "scripts/level1/train/run_level1_training.sh"
    ).read_text(encoding="utf-8")
    assert "validate_model_checkpoint.py" in launcher
    assert 'MODEL_PATH="$(cd "${MODEL_PATH}" && pwd -P)"' in launcher
    assert '[[ ! -f "${MODEL_PATH}/model.safetensors.index.json"' not in launcher
