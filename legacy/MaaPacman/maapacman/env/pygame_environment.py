"""External environment wrapper for an unmodified pacman-python checkout."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from maapacman.actions import ACTION_ORDER, Action, coerce_action
from maapacman.errors import (
    EnvironmentClosedError,
    EpisodeFinishedError,
    EpisodeNotStartedError,
    InvalidConfigurationError,
    PacmanEnvError,
)

from .config import PacmanEnvSpec


def _canonical_text_sha256(path: Path) -> str:
    """Hash text content with platform line endings normalized to LF."""

    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


class PygameWorkerError(PacmanEnvError):
    """The external original-pygame worker failed or violated its protocol."""


@dataclass(frozen=True)
class PygamePacmanEnvConfig:
    pacman_python_root: str | os.PathLike[str] | None = None
    level: int = 1
    max_steps: int = 512
    timeout_seconds: float = 15.0
    video_driver: str | None = "dummy"
    audio_driver: str | None = "dummy"
    python_executable: str | os.PathLike[str] = sys.executable
    worker_base_dir: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        if self.level != 1:
            raise InvalidConfigurationError(
                "PygamePacmanEnv currently supports only pacman-python level 1"
            )
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool):
            raise InvalidConfigurationError("max_steps must be an integer")
        if self.max_steps <= 0:
            raise InvalidConfigurationError("max_steps must be positive")
        if self.timeout_seconds <= 0:
            raise InvalidConfigurationError("timeout_seconds must be positive")


class PygamePacmanEnv:
    """Gym-style wrapper around the original, unmodified pygame game process."""

    API_VERSION = "1.0"

    def __init__(self, config: PygamePacmanEnvConfig | None = None) -> None:
        self.config = config or PygamePacmanEnvConfig()
        default_root = Path(__file__).resolve().parents[3] / "pacman-python"
        configured_root = self.config.pacman_python_root or os.getenv(
            "MAAPACMAN_PACMAN_PYTHON_ROOT"
        )
        self._root = Path(configured_root or default_root).resolve()
        self._script = self._root / "pacman" / "pacman.pyw"
        self._level_path = self._root / "pacman" / "res" / "levels" / "1.txt"
        if not self._script.is_file() or not self._level_path.is_file():
            raise InvalidConfigurationError(
                f"invalid pacman-python checkout: {self._root}"
            )
        self._revision = self._git_revision()
        self._level_revision = _canonical_text_sha256(self._level_path)
        self._spec = PacmanEnvSpec(
            api_version=self.API_VERSION,
            env_id="pacman-python-level1-pygame-v1",
            level_revision=self._level_revision,
            renderer_revision=f"pacman-python:{self._revision}",
            observation_shape=(400, 336, 3),
            observation_dtype="uint8",
            action_tokens=tuple(action.value for action in ACTION_ORDER),
            deterministic=True,
        )
        self._process: subprocess.Popen[str] | None = None
        self._runtime_dir: Path | None = None
        self._runtime_script: Path | None = None
        self._resource_mode: str | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=80)
        self._frame: np.ndarray | None = None
        self._state: dict[str, Any] | None = None
        self._steps = 0
        self._request_id = 0
        self._seed = 0
        self._started = False
        self._finished = False
        self._closed = False
        self._initial_collectibles = 0
        self._last_logic_frames = 0

    @property
    def spec(self) -> PacmanEnvSpec:
        self._ensure_open()
        return self._spec

    @property
    def pacman_python_revision(self) -> str:
        return self._revision

    @property
    def worker_runtime_dir(self) -> Path | None:
        return self._runtime_dir

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._ensure_open()
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise InvalidConfigurationError("reset seed must be an integer")
        self._stop_worker()
        self._started = False
        self._frame = None
        self._state = None
        self._start_worker()
        message = self._receive(expected_type="ready")
        self._accept_message(message)
        self._steps = 0
        self._seed = seed if seed is not None else 0
        self._finished = False
        self._started = True
        self._initial_collectibles = int(self._state["collectibles_remaining"])
        return self.render(), self._build_info(score_delta=0, seed=self._seed)

    def step(
        self, action: Action | str
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._ensure_ready()
        if self._finished:
            raise EpisodeFinishedError("episode is finished; call reset()")
        canonical = coerce_action(action)
        previous = dict(self._state)
        self._request_id += 1
        self._send(
            {"op": "step", "request_id": self._request_id, "action": canonical.value}
        )
        message = self._receive(expected_type="step", request_id=self._request_id)
        self._accept_message(message)
        self._steps += 1

        score_delta = int(self._state["score"]) - int(previous["score"])
        terminated = int(self._state["mode"]) == 6
        truncated = self._steps >= self.config.max_steps and not terminated
        self._finished = terminated or truncated
        info = self._build_info(score_delta=score_delta, seed=self._seed)
        info.update(
            {
                "action": canonical.value,
                "action_applied": canonical is Action.STAY
                or [previous["row"], previous["col"]]
                != [self._state["row"], self._state["col"]],
                "previous_position": [previous["row"], previous["col"]],
                "pacman_position": [self._state["row"], self._state["col"]],
                "pellet_eaten": self._state["normal_pellets"]
                < previous["normal_pellets"],
                "power_pellet_eaten": self._state["power_pellets"]
                < previous["power_pellets"],
                "wall_collision": canonical is not Action.STAY
                and [previous["row"], previous["col"]]
                == [self._state["row"], self._state["col"]],
                "logic_frames": self._last_logic_frames,
            }
        )
        return self.render(), float(score_delta), terminated, truncated, info

    def render(self) -> np.ndarray:
        self._ensure_ready()
        return self._frame.copy()

    def legal_actions(self) -> tuple[Action, ...]:
        self._ensure_ready()
        legal = set(self._state["open"])
        legal.add("S")
        return tuple(action for action in ACTION_ORDER if action.value in legal)

    def snapshot(self) -> dict[str, Any]:
        self._ensure_ready()
        return dict(self._state)

    def close(self) -> None:
        if not self._closed:
            self._stop_worker()
            self._closed = True

    def __enter__(self) -> "PygamePacmanEnv":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _git_revision(self) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(self._root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def _start_worker(self) -> None:
        self._prepare_runtime_dir()
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        package_root = str(Path(__file__).resolve().parents[2])
        environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (package_root, environment.get("PYTHONPATH"))
            if part
        )
        if self.config.video_driver is not None:
            environment["SDL_VIDEODRIVER"] = self.config.video_driver
        else:
            environment.pop("SDL_VIDEODRIVER", None)
        if self.config.audio_driver is not None:
            environment["SDL_AUDIODRIVER"] = self.config.audio_driver
        try:
            process = subprocess.Popen(
                [
                    str(self.config.python_executable),
                    "-m",
                    "maapacman.env._pygame_worker",
                    "--script",
                    str(self._runtime_script),
                ],
                cwd=self._runtime_dir,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except BaseException:
            self._cleanup_runtime_dir()
            raise
        self._process = process
        self._messages = queue.Queue()
        self._stderr.clear()
        threading.Thread(
            target=self._read_stdout,
            args=(process.stdout,),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            args=(process.stderr,),
            daemon=True,
        ).start()

    def _prepare_runtime_dir(self) -> None:
        base_dir = None
        if self.config.worker_base_dir is not None:
            base_dir = Path(self.config.worker_base_dir).resolve()
            base_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir = Path(
            tempfile.mkdtemp(prefix="maapacman-worker-", dir=base_dir)
        ).resolve()
        runtime_script = runtime_dir / "pacman.pyw"
        source_resources = self._script.parent / "res"
        runtime_resources = runtime_dir / "res"
        try:
            shutil.copy2(self._script, runtime_script)
            if runtime_script.read_bytes() != self._script.read_bytes():
                raise PygameWorkerError("isolated pacman.pyw copy does not match source")
            try:
                os.symlink(
                    source_resources,
                    runtime_resources,
                    target_is_directory=True,
                )
                resource_mode = "source-symlink"
            except OSError:
                # Windows may deny directory symlink creation when Developer
                # Mode is disabled.  A private resource copy preserves worker
                # isolation; Linux/H100 is expected to use the read-only link.
                shutil.copytree(source_resources, runtime_resources)
                resource_mode = "private-copy"
        except BaseException:
            shutil.rmtree(runtime_dir, ignore_errors=True)
            raise
        self._runtime_dir = runtime_dir
        self._runtime_script = runtime_script
        self._resource_mode = resource_mode

    def _cleanup_runtime_dir(self) -> None:
        runtime_dir = self._runtime_dir
        self._runtime_dir = None
        self._runtime_script = None
        self._resource_mode = None
        if runtime_dir is not None:
            shutil.rmtree(runtime_dir, ignore_errors=False)

    def _read_stdout(self, stream: TextIO | None) -> None:
        if stream is None:
            return
        for line in stream:
            try:
                self._messages.put(json.loads(line))
            except json.JSONDecodeError:
                # pygame/SDL or site customizations may print a banner before
                # the worker redirects ordinary stdout.  Keep it as diagnostic
                # context without confusing it for an IPC response.
                self._stderr.append(f"worker stdout: {line.rstrip()}")

    def _read_stderr(self, stream: TextIO | None) -> None:
        if stream is None:
            return
        for line in stream:
            self._stderr.append(line.rstrip())

    def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise self._worker_failure("worker is not running")
        process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        process.stdin.flush()

    def _receive(
        self, *, expected_type: str, request_id: int | None = None
    ) -> dict[str, Any]:
        try:
            message = self._messages.get(timeout=self.config.timeout_seconds)
        except queue.Empty as exc:
            raise self._worker_failure(
                f"timed out waiting for worker {expected_type!r} response"
            ) from exc
        if message.get("type") == "error":
            raise self._worker_failure(str(message.get("error")))
        if message.get("type") != expected_type:
            raise self._worker_failure(
                f"expected {expected_type!r}, received {message.get('type')!r}"
            )
        if request_id is not None and message.get("request_id") != request_id:
            raise self._worker_failure(
                f"response id {message.get('request_id')!r} != {request_id!r}"
            )
        return message

    def _accept_message(self, message: dict[str, Any]) -> None:
        descriptor = message["frame"]
        raw = zlib.decompress(base64.b64decode(descriptor["data"]))
        if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
            raise self._worker_failure("worker frame checksum mismatch")
        frame = np.frombuffer(raw, dtype=np.dtype(descriptor["dtype"]))
        self._frame = frame.reshape(tuple(descriptor["shape"])).copy()
        self._state = dict(message["state"])
        self._last_logic_frames = int(message.get("logic_frames", 0))

    def _build_info(self, *, score_delta: int, seed: int) -> dict[str, Any]:
        remaining = int(self._state["collectibles_remaining"])
        eaten = self._initial_collectibles - remaining
        terminated = int(self._state["mode"]) == 6
        truncated = self._steps >= self.config.max_steps and not terminated
        return {
            "env_api_version": self._spec.api_version,
            "env_id": self._spec.env_id,
            "backend": "original-pygame",
            "video_driver": self.config.video_driver or "platform-default",
            "worker_runtime_id": self._runtime_dir.name,
            "resource_mode": self._resource_mode,
            "pacman_python_revision": self._revision,
            "level_revision": self._level_revision,
            "renderer_revision": self._spec.renderer_revision,
            "level": int(self._state["level"]),
            "seed": seed,
            "step": self._steps,
            "pacman_position": [
                int(self._state["row"]),
                int(self._state["col"]),
            ],
            "score": int(self._state["score"]),
            "lives": int(self._state["lives"]),
            "score_delta": score_delta,
            "pellets_initial": self._initial_collectibles,
            "pellets_eaten": eaten,
            "pellets_remaining": remaining,
            "normal_pellets_remaining": int(self._state["normal_pellets"]),
            "power_pellets_remaining": int(self._state["power_pellets"]),
            "pellet_clear_rate": eaten / self._initial_collectibles,
            "legal_actions": [action.value for action in self.legal_actions()],
            "pygame_mode": int(self._state["mode"]),
            "terminated": terminated,
            "truncated": truncated,
            "terminal_reason": (
                "all_normal_pellets"
                if terminated
                else "max_steps" if truncated else None
            ),
        }

    def _stop_worker(self) -> None:
        process = self._process
        self._process = None
        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        if process.stdin is not None:
                            process.stdin.write('{"op":"close"}\n')
                            process.stdin.flush()
                        process.wait(timeout=3)
                    except (OSError, subprocess.TimeoutExpired):
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=3)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        finally:
            self._cleanup_runtime_dir()

    def _worker_failure(self, message: str) -> PygameWorkerError:
        details = "\n".join(self._stderr)
        suffix = f"\nworker stderr:\n{details}" if details else ""
        return PygameWorkerError(message + suffix)

    def _ensure_open(self) -> None:
        if self._closed:
            raise EnvironmentClosedError("environment is closed")

    def _ensure_ready(self) -> None:
        self._ensure_open()
        if not self._started or self._frame is None or self._state is None:
            raise EpisodeNotStartedError("call reset() before using the environment")
