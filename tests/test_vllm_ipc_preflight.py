"""Real launcher/stdlib IPC checks with all model/GPU work blocked by fakes."""

import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/level1/train/run_level1_training.sh"


def _bash():
    binary = shutil.which("bash")
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    if os.name == "nt" and candidate.is_file():
        binary = str(candidate)
    if not binary:
        pytest.skip("bash unavailable")
    return binary


@pytest.fixture
def short_directory():
    with tempfile.TemporaryDirectory(prefix="vi-") as directory:
        root = Path(directory)
        if len(os.fsencode(root.as_posix())) > 65:
            pytest.skip("platform test temp root is too long for a short IPC fixture")
        yield root


def _run_launcher(tmp_path, *, tmpdir, rpc_path=None):
    model_marker = tmp_path / "model-called"
    gpu_marker = tmp_path / "gpu-called"
    wrapper = tmp_path / "python-wrapper.sh"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-" && "${2:-}" == --pacman-vllm-ipc-preflight ]]; then\n'
        '  exec "$REAL_PYTHON" "$@"\n'
        "fi\n"
        'printf "%s" "$1" > "$MODEL_MARKER"\n'
        "exit 42\n",
        newline="\n",
    )
    wrapper.chmod(0o755)
    shell_env = tmp_path / "shell-env.sh"
    shell_env.write_text(
        'nvidia-smi() { printf "unexpected GPU probe" > "$GPU_MARKER"; return 99; }\n',
        newline="\n",
    )
    environment = {
        **os.environ,
        "BASH_ENV": shell_env.as_posix(),
        "PYTHON": wrapper.as_posix(),
        "REAL_PYTHON": Path(sys.executable).as_posix(),
        "CONFIG": (ROOT / "configs/level1/train/curriculum1.yaml").as_posix(),
        "ARTIFACT_ROOT": (tmp_path / "artifacts").as_posix(),
        "DATASET_OUTPUT_ROOT": (tmp_path / "dataset").as_posix(),
        "MODEL_MARKER": model_marker.as_posix(),
        "GPU_MARKER": gpu_marker.as_posix(),
        "TMPDIR": str(tmpdir),
    }
    environment.pop("VLLM_RPC_BASE_PATH", None)
    if rpc_path is not None:
        environment["VLLM_RPC_BASE_PATH"] = str(rpc_path)
    result = subprocess.run(
        [_bash(), LAUNCHER.as_posix(), "--smoke-updates", "2"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert not gpu_marker.exists()
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "dataset").exists()
    return result, model_marker


def _passed(result):
    lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("vllm_ipc_preflight=ok ")
    ]
    assert len(lines) == 1, (result.stdout, result.stderr)
    return json.loads(lines[0].split(" ", 1)[1])


def test_long_tmpdir_fails_before_model_validation_and_gpu_checks(tmp_path):
    directory = tmp_path / ("long-" * 12)
    directory.mkdir()
    result, model_marker = _run_launcher(tmp_path, tmpdir=directory)
    assert result.returncode == 2
    assert "vllm_ipc_preflight=failed" in result.stderr
    assert "maximum 107" in result.stderr
    assert "VLLM_RPC_BASE_PATH" in result.stderr
    assert not model_marker.exists()
    assert not list(directory.iterdir())


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="selected Python has no AF_UNIX; Linux bind test required",
)
def test_short_rpc_override_accepts_long_tmpdir_and_binds_real_socket(
    tmp_path, short_directory
):
    long_tmp = tmp_path / ("long-" * 12)
    long_tmp.mkdir()
    before = set(short_directory.iterdir())
    result, model_marker = _run_launcher(
        tmp_path, tmpdir=long_tmp, rpc_path=short_directory
    )
    checked = _passed(result)
    assert checked["source"] == "VLLM_RPC_BASE_PATH"
    assert checked["socket_bind_verified"] is True
    assert checked["endpoint_bytes"] <= 107
    assert model_marker.exists()  # Safely blocked by the fake validator, not GPU work.
    assert set(short_directory.iterdir()) == before
    assert not list(long_tmp.iterdir())


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="selected Python has no AF_UNIX; Linux bind test required",
)
def test_short_explicit_tmpdir_uses_library_default_and_binds(
    tmp_path, short_directory
):
    result, model_marker = _run_launcher(tmp_path, tmpdir=short_directory)
    checked = _passed(result)
    assert checked["source"] == "TMPDIR"
    assert checked["socket_bind_verified"] is True
    assert model_marker.exists()
    assert not list(short_directory.iterdir())


def test_ipc_budget_counts_encoded_bytes_not_characters(tmp_path, short_directory):
    prefix = len(os.fsencode(short_directory.as_posix())) + 1
    count = (70 - prefix) // 2 + 1
    directory = short_directory / ("é" * count)
    directory.mkdir()
    assert len(str(directory)) <= 70
    assert len(os.fsencode(str(directory))) > 70
    result, model_marker = _run_launcher(
        tmp_path, tmpdir=short_directory, rpc_path=directory
    )
    assert result.returncode == 2
    assert "maximum 107" in result.stderr
    assert not model_marker.exists()


@pytest.mark.parametrize("kind", ["missing", "relative", "empty"])
def test_invalid_explicit_rpc_directory_is_not_created_or_replaced(
    tmp_path, short_directory, kind
):
    path = {
        "missing": short_directory / "missing",
        "relative": "relative-ipc",
        "empty": "",
    }[kind]
    result, model_marker = _run_launcher(
        tmp_path, tmpdir=short_directory, rpc_path=path
    )
    assert result.returncode == 2
    assert "vllm_ipc_preflight=failed" in result.stderr
    assert not model_marker.exists()
    assert not (short_directory / "missing").exists()


def test_launcher_shell_syntax():
    result = subprocess.run(
        [_bash(), "-n", LAUNCHER.as_posix()], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="selected Python has no AF_UNIX; Linux boundary bind test required",
)
def test_107_byte_endpoint_boundary_binds_successfully(tmp_path, short_directory):
    base_length = len(os.fsencode(str(short_directory)))
    directory = short_directory / ("b" * (70 - base_length - 1))
    directory.mkdir()
    assert len(os.fsencode(str(directory))) == 70
    result, model_marker = _run_launcher(
        tmp_path, tmpdir=short_directory, rpc_path=directory
    )
    checked = _passed(result)
    assert checked["endpoint_bytes"] == 107
    assert checked["socket_bind_verified"] is True
    assert model_marker.exists()
    assert not list(directory.iterdir())


def test_108_byte_endpoint_boundary_fails_before_model_or_gpu(
    tmp_path, short_directory
):
    base_length = len(os.fsencode(str(short_directory)))
    directory = short_directory / ("b" * (71 - base_length - 1))
    directory.mkdir()
    assert len(os.fsencode(str(directory))) == 71
    result, model_marker = _run_launcher(
        tmp_path, tmpdir=short_directory, rpc_path=directory
    )
    assert result.returncode == 2
    assert "108-byte" in result.stderr
    assert not model_marker.exists()
    assert not list(directory.iterdir())


def test_missing_explicit_tmpdir_never_falls_back_to_platform_default(
    tmp_path, short_directory
):
    missing = short_directory / "missing"
    result, model_marker = _run_launcher(tmp_path, tmpdir=missing)
    assert result.returncode == 2
    assert "TMPDIR directory does not exist" in result.stderr
    assert not model_marker.exists()
    assert not missing.exists()


@pytest.mark.skipif(
    hasattr(socket, "AF_UNIX"), reason="checks unsupported Windows Python diagnostic"
)
def test_unsupported_python_fails_cleanly_before_model_or_gpu(
    tmp_path, short_directory
):
    result, model_marker = _run_launcher(tmp_path, tmpdir=short_directory)
    assert result.returncode == 2
    assert "does not support AF_UNIX" in result.stderr
    assert not model_marker.exists()
