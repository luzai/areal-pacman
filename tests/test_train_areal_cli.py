from train_areal import _config_override


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
