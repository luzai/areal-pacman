from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_level1_sampled_run.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_level1_sampled_run", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write(
    root: Path,
    label: str,
    *,
    clear: float,
    pellets: float,
    reward: float,
    completions: int = 0,
    wall_rate: float = 0.1,
) -> None:
    output = root / label / "sampled12.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": 12,
        "decoding": {
            "temperature": 0.7,
            "top_p": 0.95,
            "enable_thinking": False,
        },
        "image_prompt_style": "live_state_v3",
        "observation_contract": (
            "screenshot_plus_live_state_and_navigation_history"
        ),
        "reward_contract": {"nearest_pellet_alpha": 0.0},
        "reasoning_turns": 0,
        "parse_failures": 0,
        "average_base_reward": reward + 287,
        "average_shaped_reward": reward,
        "average_normal_pellets_eaten": pellets,
        "average_normal_pellet_clear_rate": clear,
        "full_completions": completions,
        "average_final_score": reward + 287,
        "average_wall_collisions": wall_rate * 287,
        "average_wall_hit_rate": wall_rate,
        "average_oscillation_returns": 10,
        "average_oscillation_rate": 10 / 287,
        "average_episode_length": 287,
        "action_counts": {"U": 0, "D": 0, "L": 0, "R": 0, "S": 3444},
        "episode_results": [
            {
                "normal_pellet_clear_rate": clear,
                "normal_pellets_eaten": pellets,
            }
            for _ in range(12)
        ],
    }
    output.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_uniform_sampled_comparator_reports_overfit_pass(tmp_path: Path) -> None:
    _write(tmp_path, "base", clear=0.10, pellets=19.4, reward=-200)
    _write(tmp_path, "update01", clear=0.15, pellets=29.1, reward=-180)
    _write(tmp_path, "update02", clear=0.25, pellets=48.5, reward=-100)
    _write(tmp_path, "update03", clear=0.18, pellets=34.9, reward=-130)
    _write(tmp_path, "update04", clear=0.12, pellets=23.3, reward=-190)

    result = MODULE.compare(tmp_path, 12)

    assert result["best_checkpoint"] == "update02"
    assert result["overfit_pass"] is True
    assert (
        result["comparisons_to_base"]["update02"][
            "delta_average_normal_pellet_clear_rate"
        ]
        == 0.15
    )
    assert result["contract"]["temperature"] == 0.7
    assert result["contract"]["nearest_pellet_alpha"] == 0.0


def test_uniform_sampled_comparator_rejects_wrong_decode(tmp_path: Path) -> None:
    _write(tmp_path, "base", clear=0.1, pellets=19.4, reward=-200)
    path = tmp_path / "base" / "sampled12.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decoding"]["temperature"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        MODULE.compare(tmp_path, 12)
    except ValueError as exc:
        assert "temperature" in str(exc)
    else:
        raise AssertionError("wrong decode contract was accepted")
