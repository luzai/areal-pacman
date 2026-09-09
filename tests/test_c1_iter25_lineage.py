from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "scripts/repro/verify_c1_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("verify_c1_checkpoint", VERIFY_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_verify_hf_accepts_complete_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch7epochstep1globalstep15"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    result = VERIFY.verify_hf(checkpoint, 15)

    assert result["global_step"] == 15
    assert result["weight_files"] == 1


def test_verify_hf_rejects_wrong_boundary(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch7epochstep1globalstep14"
    checkpoint.mkdir()
    with pytest.raises(RuntimeError, match="expected globalstep15"):
        VERIFY.verify_hf(checkpoint, 15)


def test_verify_dcp_requires_optimizer_and_matching_step(tmp_path: Path) -> None:
    checkpoint = tmp_path / "recover_checkpoint"
    checkpoint.mkdir()
    metadata = {
        "state_dict_metadata": {
            "dcp.model.layer.weight": {},
            "dcp.optim.state.layer.exp_avg": {},
        }
    }
    with (checkpoint / ".metadata").open("wb") as handle:
        pickle.dump(metadata, handle)
    recover_info = tmp_path / "recover_info"
    recover_info.mkdir()
    (recover_info / "step_info.json").write_text(
        json.dumps({"global_step": 16}), encoding="utf-8"
    )
    for name in (
        "checkpoint_info.json",
        "evaluator_info.json",
        "saver_info.json",
        "stats_logger_info.json",
    ):
        (recover_info / name).write_text("{}", encoding="utf-8")
    (recover_info / "dataloader_info.pkl").write_bytes(b"state")

    result = VERIFY.verify_dcp(checkpoint, recover_info, 16)

    assert result["optimizer_metadata_keys"] == 1


def test_verify_dcp_rejects_weights_only_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "recover_checkpoint"
    checkpoint.mkdir()
    with (checkpoint / ".metadata").open("wb") as handle:
        pickle.dump({"state_dict_metadata": {"dcp.model.layer.weight": {}}}, handle)
    recover_info = tmp_path / "recover_info"
    recover_info.mkdir()
    with pytest.raises(RuntimeError, match="no optimizer state"):
        VERIFY.verify_dcp(checkpoint, recover_info, 16)


def test_launcher_contains_both_optimizer_handoff_gates() -> None:
    text = (ROOT / "scripts/repro/run_c1_iter25_lineage.sh").read_text(encoding="utf-8")
    assert "no_save_optim" in text
    assert "no_load_optim" in text
    assert '"$verify" dcp' in text
    assert "new_branch_dcp_model_optimizer_plus_reconstructed_scheduler_rng" in text
    assert "stage1_stop=16" in text
    assert "stage2_stop=17" in text
    assert "stage3_stop=25" in text
