from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "level1"
    / "report"
    / "summarize_level1_trajectories.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_level1_trajectories", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_trajectory(
    root: Path,
    *,
    row_id: str,
    sample_id: str,
    split: str,
    action: str,
    score: int,
) -> Path:
    path = root / f"{row_id}--sample-{sample_id}.json"
    path.write_text(
        json.dumps(
            {
                "id": row_id,
                "trajectory_sample_id": sample_id,
                "split": split,
                "decoding": {
                    "enable_thinking": False,
                    "temperature": 0.7,
                    "top_p": 0.95,
                },
                "image_prompt_style": "live_state_v3",
                "final_score": score,
                "pellet_clear_rate": score / 1000,
                "normal_pellet_clear_rate": score / 1000,
                "normal_pellets_eaten": score,
                "wall_collisions": 2,
                "oscillation_returns": 1,
                "steps": 10,
                "parse_failures": 0,
                "won": False,
                "trajectory": [
                    {
                        "action": action,
                        "reasoning_content": "",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_audit_splits_chronological_train_and_validation_batches(
    tmp_path: Path,
) -> None:
    files = [
        _write_trajectory(
            tmp_path,
            row_id="train-row",
            sample_id=f"train-{index}",
            split="train",
            action="L",
            score=10 + index,
        )
        for index in range(4)
    ]
    files += [
        _write_trajectory(
            tmp_path,
            row_id="validation-row",
            sample_id=f"validation-{index}",
            split="validation",
            action="R",
            score=20 + index,
        )
        for index in range(2)
    ]
    for index, path in enumerate(files):
        timestamp_ns = 1_700_000_000_000_000_000 + index
        path.touch()
        path.chmod(0o600)
        import os

        os.utime(path, ns=(timestamp_ns, timestamp_ns))

    result = MODULE.audit(
        tmp_path,
        train_batch_size=4,
        validation_batch_size=2,
    )

    assert result["trajectory_files"] == 6
    assert result["unique_sample_ids"] == 6
    assert result["complete_batches"] == 2
    assert result["incomplete_batches"] == 0
    assert [batch["split"] for batch in result["batches"]] == [
        "train",
        "validation",
    ]
    assert result["batches"][0]["action_counts"] == {"L": 4}
    assert result["batches"][1]["action_counts"] == {"R": 2}
    assert result["batches"][1]["decoding_contracts"] == [
        {
            "enable_thinking": False,
            "temperature": 0.7,
            "top_p": 0.95,
        }
    ]
    assert result["batches"][1]["average_normal_pellets_eaten"] == 20.5
    assert result["batches"][1]["average_wall_hit_rate"] == 0.2
    assert result["batches"][1]["average_oscillation_rate"] == 0.1


def test_audit_rejects_filename_id_mismatch(tmp_path: Path) -> None:
    path = _write_trajectory(
        tmp_path,
        row_id="row",
        sample_id="sample",
        split="train",
        action="L",
        score=10,
    )
    path.rename(tmp_path / "wrong.json")

    with pytest.raises(ValueError, match="filename does not match"):
        MODULE.audit(
            tmp_path,
            train_batch_size=1,
            validation_batch_size=1,
        )
