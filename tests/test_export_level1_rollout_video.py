import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from scripts.level1.report import export_level1_rollout_video as exporter


def run_export(tmp_path, monkeypatch, records, replay_flags, *, reason):
    episode = {
        "id": "seed28",
        "level": 1,
        "ghost_mode": "normal",
        "max_steps": 512,
        "seed": 28,
        "terminal_reason": reason,
        "final_score": 10,
        "normal_pellets_remaining": 191,
        "trajectory": records,
    }
    source = tmp_path / "trajectory.json"
    source.write_text(json.dumps(episode), encoding="utf-8")
    image = Image.new("RGB", (2, 2))
    flags = iter(replay_flags)

    class ReplayEnv:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def reset(self, *, seed):
            return image, {"score": 0}

        def step(self, action):
            terminated, truncated = next(flags)
            return image, 0.0, terminated, truncated, {"score": 10}

        def snapshot(self):
            return {"open": [exporter.Action.RIGHT.value]}

    compose = Mock(return_value=image)
    encoder = Mock(
        return_value=SimpleNamespace(stdin=io.BytesIO(), wait=lambda: 0)
    )
    monkeypatch.setattr(exporter, "PygamePacmanEnv", ReplayEnv)
    monkeypatch.setattr(exporter, "compose_frame", compose)
    monkeypatch.setattr(exporter.subprocess, "Popen", encoder)
    monkeypatch.setattr(
        "sys.argv",
        ["export", "--trajectory", str(source), "--output", str(tmp_path / "out.mp4")],
    )
    return compose, encoder


def record(*, refusal=False, terminated=False, truncated=False):
    return {
        "action": exporter.Action.RIGHT.value,
        "safety_refusal": refusal,
        "terminal_reason": "safety_refusal" if refusal else "max_steps",
        "terminated": terminated,
        "truncated": truncated,
    }


@pytest.mark.parametrize("refusal", [False, True])
def test_exports_normal_and_planner_truncated_rollouts(tmp_path, monkeypatch, refusal):
    reason = "safety_refusal" if refusal else "max_steps"
    compose, encoder = run_export(
        tmp_path,
        monkeypatch,
        [record(refusal=refusal, truncated=True)],
        [(False, not refusal)],
        reason=reason,
    )
    exporter.main()
    encoder.assert_called_once()
    for call in compose.call_args_list:
        assert call.kwargs["title"] == f"seed28 | {reason} | score 10 | pellets left 191"


@pytest.mark.parametrize(
    "records,replay_flags,reason",
    [
        ([record(refusal=True, truncated=True)], [(True, False)], "safety_refusal"),
        ([record(refusal=True, truncated=True)], [(False, True)], "safety_refusal"),
        ([record(refusal=True)], [(False, False)], "safety_refusal"),
        ([record(refusal=True, terminated=True)], [(False, False)], "safety_refusal"),
        ([record(refusal=True, truncated=True)], [(False, False)], "max_steps"),
        (
            [record(refusal=True), record(truncated=True)],
            [(False, False), (False, True)],
            "max_steps",
        ),
        ([record(truncated=True)], [(False, False)], "max_steps"),
        ([record()], [(True, False)], "max_steps"),
    ],
)
def test_rejects_invalid_terminal_replays(
    tmp_path, monkeypatch, records, replay_flags, reason
):
    _, encoder = run_export(
        tmp_path, monkeypatch, records, replay_flags, reason=reason
    )
    with pytest.raises(RuntimeError, match="replay terminal-state mismatch"):
        exporter.main()
    encoder.assert_not_called()
