"""CPU-only coverage of the public stage contract, without GPU dependencies."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from areal_pacman.level1.recipe import (
    json_safe_value,
    load_recipe_document,
    load_planner_audit_settings,
    recipe_contract_metadata,
)
from train_areal import (
    _build_workflow_kwargs,
    _validate_release_stage_contract,
    _validate_reward_objective_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def stage_config(stage, monkeypatch):
    monkeypatch.setenv("CURRICULUM1_CHECKPOINT", "/test/complete-c1")
    monkeypatch.setenv("AREAL_ADMIN_API_KEY", "isolated-test-placeholder")
    return OmegaConf.load(ROOT / f"configs/level1/train/curriculum{stage}.yaml")


@pytest.mark.parametrize("stage", [1, 2])
def test_both_release_stages_validate(stage, monkeypatch):
    config = stage_config(stage, monkeypatch)
    _validate_reward_objective_contract(config)
    _validate_release_stage_contract(config)
    assert config.total_train_epochs * config.dataset_generation.train_episodes // config.train_dataset.batch_size == 100


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("field,value", [
    ("environment.max_steps", 256),
    ("dataset_generation.seed", 0),
    ("actor.kl_ctl", 0.0),
    ("evaluator.freq_steps", 1),
    ("evaluator.freq_epochs", 1),
    ("evaluator.freq_secs", 60),
    ("evaluator.eval_before_train", True),
    ("gconfig.max_new_tokens", 3),
    ("seed", 28),
    ("death_penalty", 25.0),
    ("action_protocol", "unknown"),
    ("total_train_steps", 1),
    ("gconfig.greedy", True),
    ("eval_gconfig.greedy", True),
    ("actor.init_from_scratch", True),
    ("tokenizer_path", "/test/wrong-processor"),
    ("gconfig.max_tokens", 2048),
])
def test_release_drift_is_rejected(stage, field, value, monkeypatch):
    config = stage_config(stage, monkeypatch)
    OmegaConf.update(config, field, value)
    with pytest.raises(ValueError):
        _validate_release_stage_contract(config)


def test_c1_accepts_verified_local_model_location(monkeypatch):
    config = stage_config(1, monkeypatch)
    config.actor.path = "/test/offline/Qwen3.5-9B"
    _validate_release_stage_contract(config)


@pytest.mark.parametrize("stage", [1, 2])
def test_smoke_update_override_requires_explicit_cli_intent(stage, monkeypatch):
    config = stage_config(stage, monkeypatch)
    config.total_train_steps = 2
    _validate_release_stage_contract(config, smoke_updates=2)
    with pytest.raises(ValueError, match="total_train_steps"):
        _validate_release_stage_contract(config, smoke_updates=1)
    with pytest.raises(ValueError, match="total_train_steps"):
        _validate_release_stage_contract(config)


def test_c1_cannot_enable_edward(monkeypatch):
    config = stage_config(1, monkeypatch)
    config.edward_options = True
    with pytest.raises(ValueError, match="edward_options"):
        _validate_release_stage_contract(config)


@pytest.mark.parametrize("field,value", [("std_unbiased", False), ("eps", 0.01)])
def test_c2_normalization_is_exact(field, value, monkeypatch):
    config = stage_config(2, monkeypatch)
    OmegaConf.update(config, f"actor.reward_norm.{field}", value)
    with pytest.raises(ValueError, match="std_unbiased"):
        _validate_reward_objective_contract(config)


@pytest.mark.parametrize("stage", [1, 2])
def test_manifest_contract_is_strict_json(stage):
    path = ROOT / f"configs/level1/train/curriculum{stage}.yaml"
    raw = load_recipe_document(path)
    metadata = recipe_contract_metadata(raw)
    json.dumps(metadata, allow_nan=False)
    assert metadata["data"]["train_seeds"] == list(range(28, 108))
    assert metadata["data"]["validation_seeds"] == [108, 109, 110, 111]
    assert metadata["reward"]["clip"] == ("inf" if stage == 1 else 20.0)
    assert metadata["prompt"]["version"] == raw["prompt_version"]
    assert load_planner_audit_settings(path).max_steps == 512


def test_json_safe_config_bounds_are_not_nan():
    assert json_safe_value({"clip": [float("inf")]}) == {"clip": ["inf"]}
    with pytest.raises(ValueError, match="NaN"):
        json_safe_value({"clip": float("nan")})
