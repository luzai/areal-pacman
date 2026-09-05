"""Execute the real runner with fake CPU-only tools; no server/GPU is used."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/level1/evaluate/evaluate_level1_run.sh"


def _bash():
    binary = shutil.which("bash")
    if os.name == "nt":
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file():
            binary = str(candidate)
    if binary is None:
        pytest.skip("bash is unavailable")
    return binary


@pytest.fixture
def runner_setup(tmp_path):
    workspace = tmp_path / "paths with spaces"
    workspace.mkdir()
    checkpoint = workspace / "epoch0-step2"
    checkpoint.mkdir()
    checkpoint_list = workspace / "checkpoints.txt"
    checkpoint_list.write_text(checkpoint.as_posix() + "\n", newline="\n")
    eval_root = workspace / "evaluation"
    complete = eval_root / "complete_checkpoints" / checkpoint.name
    calls = tmp_path / "calls.log"
    server_marker = tmp_path / "server-started"
    gpu_checks = tmp_path / "gpu-checks.log"
    export_args = tmp_path / "export-args.bin"
    injection_marker = tmp_path / "injected-command"
    shell_env = tmp_path / "fake-tools.sh"
    shell_env.write_text(
        "git() { return 0; }\n"
        'nvidia-smi() { printf "gpu\\n" >> "$GPU_CHECK_LOG"; return 0; }\n'
        'setsid() { printf "unexpected server" > "$SERVER_MARKER"; return 99; }\n',
        newline="\n",
    )
    python = tmp_path / "fake-python.sh"
    python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'case "$1" in\n'
        "  */build_complete_vlm_checkpoint.py)\n"
        '    printf "export\\n" >> "$CALL_LOG"\n'
        '    printf "%s\\0" "$@" > "$EXPORT_ARGS_LOG"\n'
        '    [[ "$FAKE_EXPORT_EXIT" == 0 ]] || exit "$FAKE_EXPORT_EXIT"\n'
        '    while [[ "$1" != --output-dir ]]; do shift; done\n'
        '    mkdir -p "$2"\n'
        '    printf "{}" > "$2/merge_manifest.json"\n'
        "    ;;\n"
        "  */validate_model_checkpoint.py)\n"
        '    printf "validate:%s\\n" "$2" >> "$CALL_LOG"\n'
        '    [[ -n "${PYTHONPATH:-}" ]] || exit 88\n'
        '    printf "checkpoint_validation=failed: checksum mismatch\\n" >&2\n'
        "    exit 37\n"
        "    ;;\n"
        "  *)\n"
        '    printf "unexpected:%s\\n" "$1" >> "$CALL_LOG"\n'
        "    exit 89\n"
        "    ;;\n"
        "esac\n",
        newline="\n",
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "BASH_ENV": shell_env.as_posix(),
        "PYTHON": python.as_posix(),
        "PACMAN_PYTHON_ROOT": workspace.as_posix(),
        "BASE_MODEL": (workspace / "base model").as_posix(),
        "SOURCE_RUN": workspace.as_posix(),
        "EVAL_ROOT": eval_root.as_posix(),
        "CHECKPOINT_LIST": checkpoint_list.as_posix(),
        "GPU_ID": "0",
        "CALL_LOG": calls.as_posix(),
        "SERVER_MARKER": server_marker.as_posix(),
        "GPU_CHECK_LOG": gpu_checks.as_posix(),
        "EXPORT_ARGS_LOG": export_args.as_posix(),
        "INJECTION_MARKER": injection_marker.as_posix(),
        "FAKE_EXPORT_EXIT": "0",
    }
    # The default test must not inherit an opt-in from the developer's shell.
    env.pop("EXPORT_EXPECTED_SAVED_DTYPE", None)
    env.pop("EXPORT_CONFIG_COMPARISON", None)
    return {
        "env": env,
        "checkpoint": checkpoint,
        "eval_root": eval_root,
        "complete": complete,
        "calls": calls,
        "server_marker": server_marker,
        "gpu_checks": gpu_checks,
        "export_args": export_args,
        "injection_marker": injection_marker,
    }


def _run_runner(env):
    return subprocess.run(
        [_bash(), "--noprofile", "--norc", RUNNER.as_posix()],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("explicit_policy", [False, True])
def test_bad_checkpoint_aborts_runner_before_any_server_even_when_reused(
    runner_setup, reuse, explicit_policy
):
    setup = runner_setup
    complete = setup["complete"]
    if reuse:
        complete.mkdir(parents=True)
        (complete / "merge_manifest.json").write_text("{}")
    if explicit_policy:
        setup["env"].update(
            EXPORT_EXPECTED_SAVED_DTYPE="bfloat16",
            EXPORT_CONFIG_COMPARISON="qwen3_5",
        )
    result = _run_runner(setup["env"])
    assert result.returncode == 37, (result.stdout, result.stderr)
    assert "checkpoint_validation=failed" in result.stderr
    assert setup["calls"].read_text().splitlines() == (
        ([] if reuse else ["export"]) + [f"validate:{complete.as_posix()}"]
    )
    if reuse:
        assert not setup["export_args"].exists()
    else:
        args = setup["export_args"].read_bytes().decode().split("\0")[:-1]
        assert args[0].endswith("/build_complete_vlm_checkpoint.py")
        assert args[1:] == [
            "--trained-dir",
            setup["checkpoint"].as_posix(),
            "--base-dir",
            setup["env"]["BASE_MODEL"],
            "--output-dir",
            complete.as_posix(),
            "--config-comparison",
            "qwen3_5" if explicit_policy else "strict",
            *(["--expected-saved-dtype", "bfloat16"] if explicit_policy else []),
        ]
    assert not setup["server_marker"].exists()
    assert not list((setup["eval_root"] / "servers").glob("*.pid"))
    assert not (setup["eval_root"] / "comparison.json").exists()


@pytest.mark.parametrize("explicit_policy", [False, True])
def test_export_failure_propagates_without_validator_or_server(
    runner_setup, explicit_policy
):
    setup = runner_setup
    setup["env"]["FAKE_EXPORT_EXIT"] = "43"
    if explicit_policy:
        setup["env"].update(
            EXPORT_EXPECTED_SAVED_DTYPE="bfloat16",
            EXPORT_CONFIG_COMPARISON="qwen3_5",
        )
    result = _run_runner(setup["env"])
    assert result.returncode == 43, (result.stdout, result.stderr)
    assert setup["calls"].read_text().splitlines() == ["export"]
    assert not setup["complete"].exists()
    assert not setup["server_marker"].exists()
    assert not list((setup["eval_root"] / "servers").glob("*.pid"))
    assert not (setup["eval_root"] / "comparison.json").exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("EXPORT_EXPECTED_SAVED_DTYPE", "float16"),
        ("EXPORT_EXPECTED_SAVED_DTYPE", "bfloat16 extra"),
        (
            "EXPORT_EXPECTED_SAVED_DTYPE",
            'bfloat16; printf injected > "$INJECTION_MARKER"',
        ),
        ("EXPORT_EXPECTED_SAVED_DTYPE", '$(printf injected > "$INJECTION_MARKER")'),
        ("EXPORT_CONFIG_COMPARISON", "relaxed"),
        ("EXPORT_CONFIG_COMPARISON", "qwen3_5 strict"),
        ("EXPORT_CONFIG_COMPARISON", 'qwen3_5; printf injected > "$INJECTION_MARKER"'),
        ("EXPORT_CONFIG_COMPARISON", '$(printf injected > "$INJECTION_MARKER")'),
    ],
)
def test_invalid_export_policy_rejected_before_gpu_output_or_commands(
    runner_setup, name, value
):
    setup = runner_setup
    setup["env"][name] = value
    result = _run_runner(setup["env"])
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert name in result.stderr
    for key in (
        "gpu_checks",
        "eval_root",
        "calls",
        "export_args",
        "server_marker",
        "injection_marker",
    ):
        assert not setup[key].exists(), key


def test_runner_shell_syntax():
    result = subprocess.run(
        [_bash(), "--noprofile", "--norc", "-n", RUNNER.as_posix()],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
