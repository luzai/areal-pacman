from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_level1_run.py"
SPEC = importlib.util.spec_from_file_location("compare_level1_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_summary(
    root: Path,
    *,
    label: str,
    score: float,
    clear_rate: float,
    walls: float,
    completions: int,
    filename: str = "greedy.json",
    temperature: float = 0.0,
) -> None:
    directory = root / label
    directory.mkdir(parents=True, exist_ok=True)
    metrics = {
        "episodes": 12 if filename == "sampled12.json" else 1,
        "decoding": {
            "enable_thinking": False,
            "temperature": temperature,
            "top_p": 1.0 if temperature == 0 else 0.95,
        },
        "average_final_score": score,
        "average_pellet_clear_rate": clear_rate,
        "average_wall_collisions": walls,
        "average_episode_length": 287,
        "max_no_progress_streak": 20,
        "average_base_reward": score,
        "average_shaped_reward": score - walls,
        "full_completions": completions,
        "reasoning_turns": 0,
        "action_counts": {"L": 287},
    }
    (directory / filename).write_text(
        json.dumps(metrics) + "\n",
        encoding="utf-8",
    )


def test_compare_selects_completion_before_score(tmp_path: Path) -> None:
    _write_summary(
        tmp_path,
        label="base",
        score=900,
        clear_rate=0.8,
        walls=10,
        completions=0,
    )
    _write_summary(
        tmp_path,
        label="update01",
        score=700,
        clear_rate=1.0,
        walls=20,
        completions=1,
    )
    _write_summary(
        tmp_path,
        label="update01",
        score=650,
        clear_rate=0.9,
        walls=30,
        completions=0,
        filename="sampled12.json",
        temperature=0.7,
    )
    _write_summary(
        tmp_path,
        label="base",
        score=600,
        clear_rate=0.7,
        walls=35,
        completions=0,
        filename="sampled12.json",
        temperature=0.7,
    )

    result = MODULE.compare(tmp_path, "sampled12.json")

    assert result["best_label"] == "update01"
    assert result["best_sampled_label"] == "update01"
    assert result["best_greedy_label"] == "update01"
    assert result["best_sampled12"]["average_final_score"] == 650
    assert result["base_sampled12"]["average_final_score"] == 600
    assert set(result["sampled12_by_label"]) == {"base", "update01"}
    assert result["contract"]["matched_sampled"]["labels"] == [
        "base",
        "update01",
    ]
    assert result["contract"]["thinking"] is False


def test_compare_breaks_non_completion_tie_by_clear_rate(tmp_path: Path) -> None:
    _write_summary(
        tmp_path,
        label="base",
        score=800,
        clear_rate=0.2,
        walls=5,
        completions=0,
    )
    _write_summary(
        tmp_path,
        label="update01",
        score=700,
        clear_rate=0.3,
        walls=40,
        completions=0,
    )

    result = MODULE.compare(tmp_path, "sampled12.json")

    assert result["best_label"] == "update01"
    assert result["best_sampled_label"] is None
    assert result["best_greedy_label"] == "update01"
    assert result["best_sampled12"] is None
    assert result["base_sampled12"] is None


def test_compare_uses_sampled_as_primary_and_reports_greedy_separately(
    tmp_path: Path,
) -> None:
    for label, greedy_score, sampled_score in (
        ("base", 0, 300),
        ("update01", 40, 250),
        ("update02", 20, 350),
    ):
        _write_summary(
            tmp_path,
            label=label,
            score=greedy_score,
            clear_rate=greedy_score / 1000,
            walls=200,
            completions=0,
        )
        _write_summary(
            tmp_path,
            label=label,
            score=sampled_score,
            clear_rate=sampled_score / 1000,
            walls=150,
            completions=0,
            filename="sampled12.json",
            temperature=0.7,
        )

    result = MODULE.compare(tmp_path, "sampled12.json")

    assert result["best_label"] == "update02"
    assert result["best_sampled_label"] == "update02"
    assert result["best_greedy_label"] == "update01"
    assert set(result["sampled12_by_label"]) == {
        "base",
        "update01",
        "update02",
    }
    assert result["contract"]["primary_checkpoint_selection"] == (
        "matched_sampled"
    )
    assert result["contract"]["greedy_is_separately_reported"] is True
    assert result["missing_sampled_labels"] == []


def test_complete_dual_report_rejects_missing_sampled_label(
    tmp_path: Path,
) -> None:
    _write_summary(
        tmp_path,
        label="base",
        score=0,
        clear_rate=0,
        walls=287,
        completions=0,
    )

    with pytest.raises(ValueError, match="missing sampled validation"):
        MODULE.compare(
            tmp_path,
            "sampled12.json",
            require_complete_dual=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("episodes", 1, "expected 12 episodes"),
        ("temperature", 0.0, "unexpected validation temperature"),
        ("top_p", 1.0, "unexpected validation top_p"),
        ("reasoning_turns", 1, "reasoning content was observed"),
    ),
)
def test_sampled_report_rejects_contract_drift(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    _write_summary(
        tmp_path,
        label="base",
        score=0,
        clear_rate=0,
        walls=287,
        completions=0,
    )
    _write_summary(
        tmp_path,
        label="base",
        score=100,
        clear_rate=0.1,
        walls=100,
        completions=0,
        filename="sampled12.json",
        temperature=0.7,
    )
    path = tmp_path / "base" / "sampled12.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if field in {"temperature", "top_p"}:
        payload["decoding"][field] = value
    else:
        payload[field] = value
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MODULE.compare(tmp_path, "sampled12.json")
