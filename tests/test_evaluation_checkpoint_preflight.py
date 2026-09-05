"""Execute the real runner with fake CPU-only tools; no server/GPU is used."""

import os
from pathlib import Path
import shutil
import subprocess

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


@pytest.mark.parametrize("reuse", [False, True])
def test_bad_checkpoint_aborts_runner_before_any_server_even_when_reused(
    tmp_path, reuse
):
    checkpoint = tmp_path / "epoch0-step2"
    checkpoint.mkdir()
    checkpoint_list = tmp_path / "checkpoints.txt"
    checkpoint_list.write_text(checkpoint.as_posix() + "\n", newline="\n")
    eval_root = tmp_path / "evaluation"
    complete = eval_root / "complete_checkpoints" / checkpoint.name
    if reuse:
        complete.mkdir(parents=True)
        (complete / "merge_manifest.json").write_text("{}")
    calls = tmp_path / "calls.log"
    server_marker = tmp_path / "server-started"
    shell_env = tmp_path / "fake-tools.sh"
    shell_env.write_text(
        "git() { return 0; }\n"
        "nvidia-smi() { return 0; }\n"
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
    result = subprocess.run(
        [_bash(), RUNNER.as_posix()],
        env={
            **os.environ,
            "BASH_ENV": shell_env.as_posix(),
            "PYTHON": python.as_posix(),
            "PACMAN_PYTHON_ROOT": tmp_path.as_posix(),
            "BASE_MODEL": (tmp_path / "base").as_posix(),
            "SOURCE_RUN": tmp_path.as_posix(),
            "EVAL_ROOT": eval_root.as_posix(),
            "CHECKPOINT_LIST": checkpoint_list.as_posix(),
            "GPU_ID": "0",
            "CALL_LOG": calls.as_posix(),
            "SERVER_MARKER": server_marker.as_posix(),
        },
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 37, (result.stdout, result.stderr)
    assert "checkpoint_validation=failed" in result.stderr
    assert calls.read_text().splitlines() == (
        ([] if reuse else ["export"]) + [f"validate:{complete.as_posix()}"]
    )
    assert not server_marker.exists()
    assert not list((eval_root / "servers").glob("*.pid"))
    assert not (eval_root / "comparison.json").exists()


def test_runner_shell_syntax():
    result = subprocess.run(
        [_bash(), "-n", RUNNER.as_posix()], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
