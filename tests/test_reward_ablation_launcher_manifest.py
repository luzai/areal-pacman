"""CPU-only argument/metadata tests, no model or GPU imports."""
import argparse
import ast
import copy
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/level1/train/run_level1_training.sh"
MANIFEST = ROOT / "scripts/level1/dataset/write_level1_manifest.py"


def metadata_functions():
    tree = ast.parse(MANIFEST.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in {"reward_ablation_metadata", "_Once"}]
    scope = {"argparse": argparse}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MANIFEST), "exec"), scope)
    return scope


def fixed_config():
    config = yaml.safe_load((ROOT / "configs/level1/train/curriculum1.yaml").read_text())
    config["nearest_pellet_scale_by_cleared_ratio"] = False
    return config


def test_explicit_a_metadata_and_no_input_mutation():
    config = fixed_config()
    before = copy.deepcopy(config)
    result = metadata_functions()["reward_ablation_metadata"](
        config, reward_ablation="fixed-distance", smoke_updates=4
    )
    assert config == before
    assert result["experiment_role"] == "A"
    assert result["requested_update_cap"] == 4
    assert result["full_recipe_schedule_updates"] == 100


def test_default_b_manifest_unchanged_and_fixed_requires_opt_in():
    helper = metadata_functions()["reward_ablation_metadata"]
    config = fixed_config()
    with pytest.raises(ValueError, match="explicit"):
        helper(config)
    config["nearest_pellet_scale_by_cleared_ratio"] = True
    assert helper(config) is None


@pytest.mark.parametrize("field,value", [
    ("action_protocol", "edward-option-token-v1"),
    ("nearest_pellet_alpha", 0.2),
    ("nearest_pellet_scale_by_cleared_ratio", True),
    ("total_train_steps", 4), ("total_train_epochs", 1),
    ("edward_options", True),
])
def test_scope_rejected(field, value):
    config = fixed_config()
    config[field] = value
    with pytest.raises(ValueError):
        metadata_functions()["reward_ablation_metadata"](
            config, reward_ablation="fixed-distance", smoke_updates=4
        )


@pytest.mark.parametrize("cap", [None, 2, 100])
def test_wrong_cap_rejected(cap):
    with pytest.raises(ValueError):
        metadata_functions()["reward_ablation_metadata"](
            fixed_config(), reward_ablation="fixed-distance", smoke_updates=cap
        )


def test_manifest_duplicate_arguments_rejected():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-ablation", action=metadata_functions()["_Once"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--reward-ablation", "fixed-distance", "--reward-ablation", "fixed-distance"])


@pytest.mark.parametrize("args", [
    ["--reward-ablation", "fixed-distance"],
    ["--reward-ablation", "wrong", "--smoke-updates", "4"],
    ["--reward-ablation", "fixed-distance", "--smoke-updates", "2"],
    ["--reward-ablation"],
    ["--reward-ablation", "fixed-distance", "--reward-ablation=fixed-distance", "--smoke-updates", "4"],
])
def test_shell_rejects_before_preflight(args):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    executable = str(bash) if bash.exists() else shutil.which("bash")
    if not executable:
        pytest.skip("bash unavailable")
    # Run the real argument-parser prefix only: no possibility of GPU work.
    prefix = LAUNCHER.read_text().split('REPO_ROOT="', 1)[0]
    result = subprocess.run([executable, "-c", prefix, "launcher", *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert "--reward-ablation" in result.stderr


def test_flag_reaches_all_three_consumers():
    text = LAUNCHER.read_text()
    assert text.count('"${REWARD_ABLATION_ARGS[@]}"') == 3
    source = MANIFEST.read_text()
    assert source.index("ablation = reward_ablation_metadata(") < source.index("env = PygamePacmanEnv(")
