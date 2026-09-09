"""The four-update fixed-distance comparison is explicit and fail-closed."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import train_areal


def config(monkeypatch, stage=1):
    monkeypatch.setenv("AREAL_ADMIN_API_KEY", "isolated-test-placeholder")
    path = Path(__file__).resolve().parents[1] / f"configs/level1/train/curriculum{stage}.yaml"
    result = OmegaConf.load(path)
    result.total_train_steps = 4
    result.nearest_pellet_scale_by_cleared_ratio = False
    return result


@pytest.mark.parametrize("flag", [["--reward-ablation", "fixed-distance"], ["--reward-ablation=fixed-distance"]])
def test_parse_ablation_preserves_framework_arguments(flag):
    args = ["--config", "recipe.yaml", *flag, "actor.offload=true"]
    assert train_areal._parse_reward_ablation(args) == (
        ["--config", "recipe.yaml", "actor.offload=true"], "fixed-distance"
    )


@pytest.mark.parametrize("args", [
    ["--reward-ablation"], ["--reward-ablation="],
    ["--reward-ablation", "other"],
    ["--reward-ablation=fixed-distance", "--reward-ablation=fixed-distance"],
])
def test_parse_invalid_ablation(args):
    with pytest.raises(ValueError):
        train_areal._parse_reward_ablation(args)


def test_valid_fixed_distance_a(monkeypatch):
    train_areal._validate_release_stage_contract(
        config(monkeypatch), smoke_updates=4, reward_ablation="fixed-distance"
    )


@pytest.mark.parametrize("updates", [None, 2, 100])
def test_ablation_requires_four_updates(monkeypatch, updates):
    with pytest.raises(ValueError, match="--smoke-updates 4"):
        train_areal._validate_release_stage_contract(
            config(monkeypatch), smoke_updates=updates, reward_ablation="fixed-distance"
        )


@pytest.mark.parametrize("protocol", ["legacy", "edward"])
def test_ablation_cannot_bypass_stage_contract(monkeypatch, protocol):
    candidate = config(monkeypatch, stage=2)
    if protocol == "legacy":
        candidate.action_protocol = "legacy"
    with pytest.raises(ValueError, match="C1 direct"):
        train_areal._validate_release_stage_contract(
            candidate, smoke_updates=4, reward_ablation="fixed-distance"
        )


@pytest.mark.parametrize("field,value", [
    ("nearest_pellet_alpha", 0.2),
    ("nearest_pellet_scale_by_cleared_ratio", True),
    ("total_train_steps", 100), ("environment.max_steps", 256),
])
def test_ablation_rejects_config_drift(monkeypatch, field, value):
    candidate = config(monkeypatch)
    OmegaConf.update(candidate, field, value)
    with pytest.raises(ValueError):
        train_areal._validate_release_stage_contract(
            candidate, smoke_updates=4, reward_ablation="fixed-distance"
        )


def test_default_rejects_unscaled_reward_and_accepts_full_b(monkeypatch):
    candidate = config(monkeypatch)
    with pytest.raises(ValueError, match="scaled nearest-pellet"):
        train_areal._validate_release_stage_contract(candidate, smoke_updates=4)
    candidate.nearest_pellet_scale_by_cleared_ratio = True
    candidate.total_train_steps = None
    train_areal._validate_release_stage_contract(candidate)


def test_main_forwards_ablation_to_preflight(monkeypatch, capsys):
    received = {}

    def preflight(path, **kwargs):
        received.update(kwargs)
        return True

    monkeypatch.setattr(train_areal, "_production_dry_run", preflight)
    train_areal.main(["--config", "recipe.yaml", "--dry-run", "--validate-areal",
                      "--smoke-updates", "4", "--reward-ablation", "fixed-distance"])
    assert received["reward_ablation"] == "fixed-distance"
    assert received["smoke_updates"] == 4
    assert received["config_args"] == ["--config", "recipe.yaml", "total_train_steps=4"]
    assert "reward_ablation=fixed-distance" in capsys.readouterr().out


@pytest.mark.parametrize("cap", [[], ["--smoke-updates", "2"]])
def test_main_rejects_wrong_budget_before_imports(cap):
    with pytest.raises(ValueError, match="--smoke-updates 4"):
        train_areal.main(["--reward-ablation", "fixed-distance", *cap])
