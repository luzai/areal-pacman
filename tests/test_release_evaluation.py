from __future__ import annotations

import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(filename):
    spec = importlib.util.spec_from_file_location(
        filename, ROOT / "scripts/level1/evaluate" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL = _module("evaluate_level1.py")
COMPARE = _module("compare_level1_run.py")


def args(tmp_path, *extra):
    return EVAL.parse_args(
        [
            "--model",
            "test-model",
            "--episodes",
            "1",
            "--samples-per-seed",
            "1",
            "--retries",
            "0",
            "--output",
            str(tmp_path / "summary.json"),
            *extra,
        ]
    )


@pytest.mark.parametrize(
    "stage,edward,ghost,clip",
    [(1, False, "disabled", "inf"), (2, True, "normal", 20.0)],
)
def test_evaluation_reads_real_stage_recipe(tmp_path, stage, edward, ghost, clip):
    parsed = args(
        tmp_path, "--config", str(ROOT / f"configs/level1/train/curriculum{stage}.yaml")
    )
    _, settings = EVAL.evaluation_settings(parsed)
    assert settings["edward_options"] is edward
    assert settings["ghost_mode"] == ghost
    assert settings["open_action_mask"] is not edward
    assert settings["action_token_choice"] is not edward
    assert settings["environment_max_steps"] == 512
    assert settings["max_completion_tokens"] == 1
    assert settings["temperature"] == 0.7 and settings["top_p"] == 1.0
    assert settings["death_penalty"] == settings["safety_refusal_penalty"] == 100.0
    assert settings["recipe_contract"]["reward"]["clip"] == clip
    json.dumps(settings["recipe_contract"], allow_nan=False)


@pytest.mark.parametrize(
    "extra",
    [
        ("--max-steps", "256"),
        ("--ghost-mode", "disabled"),
        ("--prompt-style", "minimal_v1"),
        ("--max-completion-tokens", "3"),
        ("--open-action-mask",),
    ],
)
def test_evaluation_rejects_old_flags_that_conflict_with_recipe(tmp_path, extra):
    with pytest.raises(ValueError, match="conflicts with --config"):
        EVAL.evaluation_settings(args(tmp_path, *extra))


def _episode(row, won=True):
    return {
        "id": row["id"],
        "trajectory_sample_id": row["id"],
        "seed": row["env"]["seed"],
        "won": won,
        "terminal_reason": "all_normal_pellets" if won else "max_steps",
        "normal_pellets_remaining": 0 if won else 1,
        "normal_pellets_eaten": 194 if won else 193,
        "power_pellets_remaining": 1,
        "normal_pellets_initial": 194,
        "pellet_clear_rate": 1.0 if won else 0.9,
        "normal_pellet_clear_rate": 1.0 if won else 0.9,
        "final_score": 1940,
        "steps": 512,
        "total_base_reward": 1940.0,
        "total_shaped_reward": 220.0,
        "wall_collisions": 0,
        "oscillation_returns": 0,
        "trajectory": [
            {
                "action": "L",
                "logic_frame_events": [{"event_type": "level_cleared"}] if won else [],
            }
        ],
    }


def _mock_runtime(monkeypatch, *, fail_first=False, malformed_win=False, won=True):
    calls = []

    def row(index, **kwargs):
        return {
            "id": str(index),
            "env": {
                "seed": kwargs["seed"],
                "max_steps": kwargs["max_steps"],
                "ghost_mode": kwargs["ghost_mode"],
                "name": "fixture-env",
            },
            "action_protocol": kwargs["action_protocol"],
            "source_revisions": {"fixture": "fixed"},
        }

    class Policy:
        def __init__(self, **kwargs):
            self.settings = kwargs
            self.last_episode = None

        async def run(self, row, **kwargs):
            calls.append((deepcopy(row), self.settings, kwargs))
            if fail_first and len(calls) == 1:
                raise ConnectionError("temporary network outage")
            self.last_episode = _episode(row, won=won)
            if malformed_win:
                self.last_episode["trajectory"][0]["logic_frame_events"] = []

    async def service(_):
        return {
            "requested_model_id": "test-model",
            "advertised_model_ids": ["test-model"],
        }

    async def no_delay(_):
        return None

    monkeypatch.setattr(EVAL, "make_episode_row", row)
    monkeypatch.setattr(EVAL, "workflow_class", lambda: Policy)
    monkeypatch.setattr(EVAL, "verify_served_model", service)
    monkeypatch.setattr(EVAL.asyncio, "sleep", no_delay)
    return calls


def test_evaluation_freezes_real_seed_matrix_and_reports_no_threshold(
    tmp_path, monkeypatch
):
    calls = _mock_runtime(monkeypatch)
    parsed = args(
        tmp_path,
        "--episodes",
        "2",
        "--samples-per-seed",
        "2",
        "--generation-seed-base",
        "11",
    )
    report = asyncio.run(EVAL.evaluate(parsed))
    assert [
        (row["env"]["seed"], kwargs["generation_seed"]) for row, _, kwargs in calls
    ] == [(112, 11), (112, 12), (113, 11), (113, 12)]
    assert (
        report["planned_episodes"]
        == report["completed_episodes"]
        == report["attempts"]
        == 4
    )
    assert report["full_completions"] == 4 and report["win_rate"] == 1
    assert report["minimum_win_rate"] is None
    assert report["evaluation_contract_sha256"] == EVAL.canonical_sha256(
        report["evaluation_contract"]
    )
    manifest = json.loads(parsed.output.with_suffix(".manifest.json").read_text())
    assert len(manifest["planned_trials"]) == 4
    assert report["checkpoint"]["weight_identity_verified"] is False
    assert all(settings["edward_options"] for _, settings, _ in calls)


def test_network_retry_keeps_failed_attempt_in_denominator(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch, fail_first=True)
    parsed = args(tmp_path, "--retries", "1")
    report = asyncio.run(EVAL.evaluate(parsed))
    assert report["attempts"] == 2 and report["error_attempts"] == 1
    assert report["win_rate"] == 0.5 and report["planned_trial_win_rate"] == 1
    attempts = [
        json.loads(line)
        for line in parsed.output.with_suffix(".attempts.jsonl")
        .read_text()
        .splitlines()
    ]
    assert attempts[0]["status"] == "error" and attempts[1]["status"] == "completed"
    assert attempts[0]["trial_id"] == attempts[1]["trial_id"]


def test_zero_observed_wins_is_a_valid_report_not_a_failed_threshold(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch, won=False)
    report = asyncio.run(EVAL.evaluate(args(tmp_path)))
    assert report["completed_episodes"] == 1
    assert report["error_attempts"] == 0
    assert report["full_completions"] == report["win_rate"] == 0
    assert report["minimum_win_rate"] is None
    assert report["evaluation_completed"] is True
    assert report["conclusion"] == "no_completion_observed_on_this_test_set"


def test_selection_preserves_zero_normal_pellet_clear_rate():
    assert COMPARE.selection_key({"full_completions": 0, "episodes": 1,
        "average_normal_pellet_clear_rate": 0.0, "average_pellet_clear_rate": 0.5}) == (0, 0)


def test_win_claim_without_real_terminal_event_is_failure(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch, malformed_win=True)
    report = asyncio.run(EVAL.evaluate(args(tmp_path)))
    assert report["full_completions"] == 0 and report["error_attempts"] == 1
    assert report["conclusion"] == "no_completion_observed_on_this_test_set"


def test_heldout_cannot_reuse_training_or_validation_seeds(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        asyncio.run(EVAL.evaluate(args(tmp_path, "--seed", "108")))


def test_existing_evaluation_is_not_overwritten(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch)
    parsed = args(tmp_path)
    asyncio.run(EVAL.evaluate(parsed))
    with pytest.raises(FileExistsError, match="preserve attempts"):
        asyncio.run(EVAL.evaluate(parsed))


def test_checkpoint_manifest_hash_is_not_a_false_load_verdict(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "merge_manifest.json").write_text('{"files": {}}')
    identity = EVAL.checkpoint_identity(
        args(tmp_path, "--checkpoint-path", str(checkpoint))
    )
    assert len(identity["manifest_sha256"]) == 64
    assert identity["weight_identity_verified"] is False


def test_server_model_id_is_verified(monkeypatch, tmp_path):
    class Client:
        models = SimpleNamespace()

        def __init__(self, **kwargs):
            async def models():
                return SimpleNamespace(data=[SimpleNamespace(id="different-model")])

            self.models = SimpleNamespace(list=models)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=Client))
    with pytest.raises(ValueError, match="not advertised"):
        asyncio.run(EVAL.verify_served_model(args(tmp_path)))


def test_comparison_reports_heldout_without_selecting_checkpoint(tmp_path, monkeypatch):
    _mock_runtime(monkeypatch)
    report = asyncio.run(EVAL.evaluate(args(tmp_path / "model-a")))
    (tmp_path / "model-a" / "sampled12.json").write_text(json.dumps(report))
    compared = COMPARE.compare(tmp_path, "sampled12.json", sampled_only=True)
    assert compared["best_label"] is None
    assert compared["contract"]["selection_allowed"] is False
    assert compared["contract"]["max_steps"] == 512


def test_comparison_rejects_mixed_harness_even_with_recomputed_hash(
    tmp_path, monkeypatch
):
    _mock_runtime(monkeypatch)
    report = asyncio.run(
        EVAL.evaluate(
            args(tmp_path / "model-a", "--purpose", "validation", "--seed", "108")
        )
    )
    (tmp_path / "model-a" / "sampled12.json").write_text(json.dumps(report))
    second = deepcopy(report)
    second["evaluation_contract"]["harness"]["edward_options"] = False
    second["evaluation_contract_sha256"] = EVAL.canonical_sha256(
        second["evaluation_contract"]
    )
    (tmp_path / "model-b").mkdir()
    (tmp_path / "model-b" / "sampled12.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="mixed evaluation protocols"):
        COMPARE.compare(tmp_path, "sampled12.json", sampled_only=True)
