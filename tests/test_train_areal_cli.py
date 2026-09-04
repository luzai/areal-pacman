import pytest

from train_areal import _apply_smoke_updates, _config_override


def test_config_override_returns_none_when_key_is_absent() -> None:
    assert _config_override(["--config", "recipe.yaml"], "train_dataset.path") is None


def test_config_override_uses_last_exact_key_value() -> None:
    args = [
        "train_dataset.path=first/train_hf",
        "valid_dataset.path=first/validation_hf",
        "train_dataset.path=second/train_hf",
        "other_train_dataset.path=wrong",
    ]
    assert _config_override(args, "train_dataset.path") == "second/train_hf"
    assert _config_override(args, "valid_dataset.path") == "first/validation_hf"


def test_smoke_updates_is_optional() -> None:
    args = ["--config", "curriculum1.yaml"]
    assert _apply_smoke_updates(args) == (args, None)


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (["--smoke-updates", "2"], 2),
        (["--smoke-updates=7"], 7),
    ],
)
def test_smoke_updates_sets_exact_training_step_limit(
    flag: list[str], expected: int
) -> None:
    cleaned, smoke_updates = _apply_smoke_updates(
        ["--config", "curriculum1.yaml", *flag]
    )
    assert smoke_updates == expected
    assert f"total_train_steps={expected}" in cleaned
    assert all(not argument.startswith("--smoke-updates") for argument in cleaned)


@pytest.mark.parametrize(
    "args",
    [
        ["--smoke-updates"],
        ["--smoke-updates", "0"],
        ["--smoke-updates=-1"],
        ["--smoke-updates=two"],
        ["--smoke-updates", "2", "--smoke-updates=3"],
        ["--smoke-updates=2", "total_train_steps=3"],
    ],
)
def test_smoke_updates_rejects_ambiguous_or_invalid_limits(
    args: list[str],
) -> None:
    with pytest.raises(ValueError):
        _apply_smoke_updates(args)
