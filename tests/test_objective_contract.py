from __future__ import annotations

from types import SimpleNamespace

import pytest

from train_areal import _validate_reward_objective_contract


def _config(*, reward_norm=None, adv_norm=None, contract="step_local_raw_v1"):
    return SimpleNamespace(
        reward_objective_contract=contract,
        workflow="areal_pacman.workflow.PacmanNativeVisionWorkflow",
        actor=SimpleNamespace(reward_norm=reward_norm, adv_norm=adv_norm),
    )


def test_step_local_raw_contract_accepts_disabled_normalization() -> None:
    _validate_reward_objective_contract(_config())


@pytest.mark.parametrize("field", ["reward_norm", "adv_norm"])
def test_step_local_raw_contract_rejects_normalization(field: str) -> None:
    kwargs = {field: SimpleNamespace(mean_level="group")}
    with pytest.raises(ValueError, match=field):
        _validate_reward_objective_contract(_config(**kwargs))


def test_step_local_raw_contract_rejects_unknown_contract() -> None:
    with pytest.raises(ValueError, match="unsupported reward_objective_contract"):
        _validate_reward_objective_contract(_config(contract="mystery"))
