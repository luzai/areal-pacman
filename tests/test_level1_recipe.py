from __future__ import annotations

import asyncio
import base64
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from maapacman.env import (
    Action,
    Position,
    PygamePacmanEnv,
    load_bundled_level,
    route_to_nearest,
)

from areal_pacman.actions import ActionParseError, parse_action
from areal_pacman.level1_dataset import (
    ENV_BACKEND,
    LONG_HORIZON_MAX_STEPS,
    PRODUCTION_MAX_STEPS,
    SHORT_HORIZON_MAX_STEPS,
    environment_metadata,
    generate_balanced_corridor_rows,
    generate_episode_rows,
    generate_single_step_rows,
    make_episode_row,
    validate_episode_row,
    write_jsonl,
)
from areal_pacman.prompts import (
    LIVE_STATE_V3_SYSTEM_PROMPT,
    LIVE_STATIC_V2_SYSTEM_PROMPT,
    LIVE_STATIC_V2_USER_INSTRUCTION,
    SYSTEM_PROMPT,
    USER_INSTRUCTION,
    build_image_messages,
    crop_pacman_local_view,
    encode_png,
    image_count,
    png_sha256,
)
from areal_pacman.rewards import RewardConfig, audit_reward, shape_reward
from areal_pacman.trajectories import audit_trajectory, summarize_episodes
from areal_pacman.level1.workflow import (
    _nearest_reachable_distance_with_diagnostics,
)
from areal_pacman.workflow import (
    ModelTurn,
    PacmanImageOnlyWorkflow,
    PacmanNativeVisionWorkflow,
    install_vllm_allowed_token_ids_adapter,
)
from train_areal import _build_workflow_kwargs


def fake_action_tokenizer() -> SimpleNamespace:
    return SimpleNamespace(
        encode=lambda token, **_: [
            {"U": 40, "D": 41, "L": 42, "R": 43}[token]
        ]
    )


def oracle_actions() -> list[str]:
    level = load_bundled_level()
    env = PygamePacmanEnv()
    _, info = env.reset(seed=0)
    remaining = set(level.pellets)
    actions: list[str] = []
    terminated = truncated = False
    while not (terminated or truncated):
        state = env.snapshot()
        position = Position(int(state["row"]), int(state["col"]))
        remaining.discard(position)
        path = route_to_nearest(
            level,
            position,
            remaining,
        )
        action = path[0]
        _, _, terminated, truncated, info = env.step(action)
        if info["pellet_eaten"]:
            remaining.discard(
                Position(
                    int(info["pacman_position"][0]),
                    int(info["pacman_position"][1]),
                )
            )
        actions.append(action.value)
    env.close()
    return actions


class OneStepEnv(PygamePacmanEnv):
    """Real original-pygame reset/step with a test-only one-step terminal."""

    def step(self, action):
        image, reward, _, _, info = super().step(action)
        info = dict(info)
        info.update(
            terminated=True,
            truncated=False,
            terminal_reason="test_complete",
        )
        return image, reward, True, False, info


class TwoStepEnv(PygamePacmanEnv):
    """Real original-pygame behavior with a test-only two-step terminal."""

    def step(self, action):
        image, reward, _, _, info = super().step(action)
        done = int(info["step"]) >= 2
        info = dict(info)
        info.update(
            terminated=done,
            truncated=False,
            terminal_reason="test_complete" if done else None,
        )
        return image, reward, done, False, info


class DatasetContractTests(unittest.TestCase):
    def test_long_horizon_256_step_rows_are_supported(self) -> None:
        row = make_episode_row(
            1,
            split="train",
            max_steps=LONG_HORIZON_MAX_STEPS,
        )
        validate_episode_row(row)
        self.assertEqual(row["env"]["max_steps"], 256)

    def test_balanced_corridor_splits_are_disjoint_and_defeat_constant_actions(
        self,
    ) -> None:
        train = list(
            generate_balanced_corridor_rows(32, split="train", seed=0)
        )
        validation = list(
            generate_balanced_corridor_rows(16, split="validation", seed=0)
        )
        self.assertFalse(
            {row["id"] for row in train} & {row["id"] for row in validation}
        )
        for rows in (train, validation):
            for action in ("U", "D", "L", "R"):
                self.assertEqual(
                    sum(
                        action in row["state_open_actions_for_audit"]
                        for row in rows
                    ),
                    len(rows) // 2,
                )
            for start in range(0, len(rows), 4):
                batch = rows[start : start + 4]
                self.assertEqual(
                    [row["state_open_actions_for_audit"] for row in batch],
                    [["L", "R"], ["U", "D"], ["L", "R"], ["U", "D"]],
                )

    def test_single_step_rows_use_distinct_collision_free_prefix_states(self) -> None:
        rows = list(
            generate_single_step_rows(4, split="train", seed=0, offset=0)
        )
        self.assertTrue(all(row["decision_steps"] == 1 for row in rows))
        self.assertEqual(
            [len(row["state_prefix_actions"]) for row in rows],
            [0, 1, 2, 3],
        )
        self.assertEqual(len({row["id"] for row in rows}), 4)

    def test_rows_are_deterministic_and_validate_revision(self) -> None:
        first = list(generate_episode_rows(3, split="train", seed=0))
        second = list(generate_episode_rows(3, split="train", seed=0))
        self.assertEqual(first, second)
        self.assertEqual(first[0]["id"], "level1-seed0-train-0001")
        self.assertEqual(
            first[0]["env"]["name"], "pacman-python-level1-pygame-v1"
        )
        self.assertEqual(first[0]["env"]["backend"], ENV_BACKEND)
        self.assertEqual(first[0]["env"]["max_steps"], PRODUCTION_MAX_STEPS)
        self.assertEqual(
            first[0]["env"]["pacman_python_revision"],
            environment_metadata()["pacman_python_revision"],
        )
        with tempfile.TemporaryDirectory() as directory:
            one = Path(directory) / "one.jsonl"
            two = Path(directory) / "two.jsonl"
            self.assertEqual(write_jsonl(first, one), write_jsonl(second, two))
            self.assertEqual(one.read_bytes(), two.read_bytes())

    def test_short_horizon_rows_are_valid(self) -> None:
        rows = list(
            generate_episode_rows(
                2,
                split="train",
                seed=0,
                max_steps=SHORT_HORIZON_MAX_STEPS,
            )
        )
        self.assertEqual(
            [row["env"]["max_steps"] for row in rows],
            [SHORT_HORIZON_MAX_STEPS, SHORT_HORIZON_MAX_STEPS],
        )

    def test_invalid_rows_are_rejected(self) -> None:
        row = make_episode_row(1, split="train")
        for key, value in (
            ("api_version", "2.0"),
            ("backend", "synthetic"),
            ("pacman_python_revision", "wrong"),
            ("level", 2),
            ("seed", True),
            ("max_steps", PRODUCTION_MAX_STEPS + 1),
            ("observation_mode", "text"),
        ):
            broken = copy.deepcopy(row)
            broken["env"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_episode_row(broken)


class PromptAndActionTests(unittest.TestCase):
    def test_prompt_contains_one_png_and_no_privileged_dynamic_state(self) -> None:
        env = PygamePacmanEnv()
        image, _ = env.reset()
        png = encode_png(image)
        messages = build_image_messages(png)
        self.assertEqual(image_count(messages), 1)
        self.assertEqual(messages[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(messages[1]["content"][0]["text"], USER_INSTRUCTION)
        combined = json.dumps(messages)
        for forbidden in ("legal_actions", "pellets_remaining", "pacman_position", "route"):
            self.assertNotIn(forbidden, combined)
        url = messages[1]["content"][1]["image_url"]["url"]
        self.assertEqual(base64.b64decode(url.split(",", 1)[1]), png)
        env.close()

    def test_png_encoding_is_deterministic(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.assertEqual(encode_png(image), encode_png(image.copy()))

    def test_local_wall_view_is_visual_only_and_centers_pacman(self) -> None:
        env = PygamePacmanEnv()
        image, _ = env.reset(seed=0)
        local = crop_pacman_local_view(image)
        env.close()
        self.assertEqual(local.shape, (288, 288, 3))
        center = local[120:168, 120:168]
        yellow = (
            (center[:, :, 0] > 240)
            & (center[:, :, 1] > 220)
            & (center[:, :, 2] < 40)
        )
        self.assertGreater(int(yellow.sum()), 100)
        messages = build_image_messages(
            encode_png(local),
            prompt_style="wall_avoidance_local_v2",
        )
        combined = json.dumps(messages)
        self.assertEqual(image_count(messages), 1)
        for forbidden in ("open_actions", "blocked_actions", "pacman_position"):
            self.assertNotIn(forbidden, combined)
        axis_messages = build_image_messages(
            encode_png(local),
            prompt_style="wall_avoidance_axis_v3",
        )
        axis_combined = json.dumps(axis_messages)
        self.assertEqual(image_count(axis_messages), 1)
        self.assertIn("if it continues horizontally", axis_combined)
        for forbidden in ("open_actions", "blocked_actions", "pacman_position"):
            self.assertNotIn(forbidden, axis_combined)

    def test_live_static_v2_is_image_only_and_borrows_visual_landmarks(self) -> None:
        png = encode_png(np.zeros((8, 8, 3), dtype=np.uint8))
        messages = build_image_messages(png, prompt_style="live_static_v2")
        self.assertEqual(
            messages[0],
            {"role": "system", "content": LIVE_STATIC_V2_SYSTEM_PROMPT},
        )
        self.assertEqual(
            messages[1]["content"][0]["text"],
            LIVE_STATIC_V2_USER_INSTRUCTION,
        )
        combined = json.dumps(messages)
        for expected in (
            "yellow circle with a mouth",
            "Blue lines are walls",
            "screen-absolute",
            "nearby pellet",
        ):
            self.assertIn(expected, combined)
        for forbidden in (
            "legal_actions",
            "pellets_remaining",
            "pacman_position",
            "Teacher action",
            "directions already taken",
            "OPEN dir",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertEqual(image_count(messages), 1)

    def test_live_state_v3_contains_authoritative_navigation_context(self) -> None:
        png = encode_png(np.zeros((8, 8, 3), dtype=np.uint8))
        context = {
            "pacman_position": [21, 14],
            "facing": "R",
            "pellets_remaining": 93,
            "open_actions": ["L", "R"],
            "legal_actions": ["L", "R", "S"],
            "blocked_actions": ["U", "D"],
            "current_cell_exit_history": ["L"],
            "current_cell_exit_counts": {"L": 17, "R": 2},
            "last_action": "R",
            "immediate_reverse_action": "L",
            "recent_actions": ["L", "R"],
            "recent_positions": [[21, 13], [21, 14]],
            "preferred_open_actions": ["R"],
        }
        messages = build_image_messages(
            png,
            prompt_style="live_state_v3",
            state_context=context,
        )
        self.assertEqual(
            messages[0],
            {"role": "system", "content": LIVE_STATE_V3_SYSTEM_PROMPT},
        )
        text = messages[1]["content"][0]["text"]
        for expected in (
            "row=21, col=14",
            "Facing: R",
            "Remaining pellets: 93",
            "OPEN dirs here: L, R",
            "BLOCKED dirs here: U, D",
            "directions already taken before: L",
            "Prefer a DIFFERENT OPEN dir than cell history: [R]",
            "Do NOT reverse the last move (R)",
            "Choose ONE ACTION from [R]",
            "Only output one ACTION letter: <one of U/D/L/R>",
        ):
            self.assertIn(expected, text)
        self.assertEqual(image_count(messages), 1)
        with self.assertRaisesRegex(ValueError, "requires state_context"):
            build_image_messages(png, prompt_style="live_state_v3")

    def test_action_parser_is_strict(self) -> None:
        for token in ("U", "D", "L", "R", "S"):
            self.assertEqual(parse_action(f" \n{token}\t").value, token)
        for invalid in ("RIGHT", "Action: R", '{"action":"R"}', "R R", "r", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ActionParseError):
                parse_action(invalid)


class RewardAndTrajectoryTests(unittest.TestCase):
    def test_unreachable_bfs_logs_debug_state_and_fails(self) -> None:
        level = load_bundled_level(1)
        start = Position(12, 10)
        targets = {Position(16, 9), Position(16, 11)}
        with (
            patch(
                "areal_pacman.level1.workflow.nearest_reachable_distance",
                side_effect=ValueError(
                    "no target is reachable from the requested position"
                ),
            ),
            self.assertLogs(
                "areal_pacman.level1.workflow",
                level="ERROR",
            ) as captured,
            self.assertRaisesRegex(
                RuntimeError,
                '"phase": "after_step"',
            ),
        ):
            _nearest_reachable_distance_with_diagnostics(
                level,
                start,
                targets,
                phase="after_step",
                previous_position=(12, 9),
                action="R",
                live_legal_actions=["L", "R"],
            )
        message = captured.output[0]
        self.assertIn('"start_position": [12, 10]', message)
        self.assertIn('"remaining_normal_pellet_count": 2', message)
        self.assertIn('"previous_position": [12, 9]', message)
        self.assertIn('"action": "R"', message)
        self.assertIn('"live_legal_actions": ["L", "R"]', message)

    def test_reward_formula_and_audit(self) -> None:
        result = shape_reward(
            10.0,
            {"pellet_clear_rate": 0.0},
            {"pellet_clear_rate": 1 / 196, "wall_collision": False},
            RewardConfig(step_penalty=1.0, wall_penalty=1.0),
        )
        self.assertAlmostEqual(result.step_penalty, 1.0)
        self.assertAlmostEqual(result.shaped_reward, 9.0)
        audit_reward(result.as_dict())
        wall = shape_reward(
            0.0,
            {"pellet_clear_rate": 0.0},
            {"pellet_clear_rate": 0.0, "wall_collision": True},
            RewardConfig(step_penalty=1.0, wall_penalty=1.0),
        )
        self.assertAlmostEqual(wall.shaped_reward, -2.0)
        progress = shape_reward(
            0.0,
            {"pellet_clear_rate": 0.0},
            {"pellet_clear_rate": 0.0, "wall_collision": False},
            RewardConfig(
                step_penalty=1.0,
                wall_penalty=1.0,
                nearest_pellet_alpha=1.0,
            ),
            nearest_pellet_distance_before=3,
            nearest_pellet_distance_after=2,
        )
        self.assertEqual(progress.nearest_pellet_progress_reward, 1.0)
        self.assertEqual(progress.shaped_reward, 0.0)
        audit_reward(progress.as_dict())
        broken = result.as_dict()
        broken["shaped_reward"] = 99.0
        with self.assertRaises(ValueError):
            audit_reward(broken)

    def test_explicit_task_and_late_bfs_distance_reward_formula(self) -> None:
        config = RewardConfig(
            use_base_reward=False,
            normal_pellet_reward=1.0,
            power_pellet_reward=1.0,
            completion_reward=50.0,
            step_penalty=0.05,
            wall_penalty=0.5,
            nearest_pellet_alpha=0.1,
            nearest_pellet_remaining_ratio_threshold=0.25,
            nearest_pellet_skip_on_eat=True,
        )
        ordinary = shape_reward(
            123.0,
            {},
            {"wall_collision": False, "pellet_eaten": False},
            config,
            normal_pellet_remaining_ratio=0.5,
            nearest_pellet_distance_before=3,
            nearest_pellet_distance_after=2,
        )
        self.assertAlmostEqual(ordinary.shaped_reward, -0.05)
        self.assertEqual(ordinary.base_reward_contribution, 0.0)
        self.assertFalse(ordinary.nearest_pellet_shaping_active)

        closer = shape_reward(
            0.0,
            {},
            {"wall_collision": False, "pellet_eaten": False},
            config,
            normal_pellet_remaining_ratio=0.25,
            nearest_pellet_distance_before=3,
            nearest_pellet_distance_after=2,
        )
        self.assertAlmostEqual(closer.shaped_reward, 0.05)
        self.assertTrue(closer.nearest_pellet_shaping_active)

        farther = shape_reward(
            0.0,
            {},
            {"wall_collision": False, "pellet_eaten": False},
            config,
            normal_pellet_remaining_ratio=0.2,
            nearest_pellet_distance_before=2,
            nearest_pellet_distance_after=3,
        )
        self.assertAlmostEqual(farther.shaped_reward, -0.15)
        audit_reward(farther.as_dict())

        wall = shape_reward(
            0.0,
            {},
            {"wall_collision": True, "pellet_eaten": False},
            config,
            normal_pellet_remaining_ratio=0.5,
            nearest_pellet_distance_before=2,
            nearest_pellet_distance_after=2,
        )
        self.assertAlmostEqual(wall.shaped_reward, -0.55)

        pellet = shape_reward(
            10.0,
            {},
            {"wall_collision": False, "pellet_eaten": True},
            config,
            normal_pellet_remaining_ratio=0.2,
            nearest_pellet_distance_before=1,
            nearest_pellet_distance_after=12,
        )
        self.assertAlmostEqual(pellet.shaped_reward, 0.95)
        self.assertEqual(pellet.normal_pellet_reward, 1.0)
        self.assertFalse(pellet.nearest_pellet_shaping_active)
        self.assertEqual(pellet.nearest_pellet_progress_reward, 0.0)

        power_pellet = shape_reward(
            50.0,
            {},
            {
                "wall_collision": False,
                "pellet_eaten": False,
                "power_pellet_eaten": True,
            },
            config,
            normal_pellet_remaining_ratio=0.5,
            nearest_pellet_distance_before=2,
            nearest_pellet_distance_after=2,
        )
        self.assertAlmostEqual(power_pellet.shaped_reward, 0.95)
        self.assertTrue(power_pellet.power_pellet_eaten)
        self.assertEqual(power_pellet.power_pellet_reward, 1.0)
        audit_reward(power_pellet.as_dict())

        completion = shape_reward(
            10.0,
            {},
            {
                "wall_collision": False,
                "pellet_eaten": True,
                "terminal_reason": "all_normal_pellets",
                "normal_pellets_remaining": 0,
            },
            config,
            normal_pellet_remaining_ratio=0.0,
            nearest_pellet_distance_before=1,
            nearest_pellet_distance_after=0,
        )
        self.assertAlmostEqual(completion.shaped_reward, 50.95)
        self.assertTrue(completion.level_completed)
        audit_reward(completion.as_dict())

    def test_reward_config_rejects_invalid_shaping_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "normal_pellet_reward"):
            RewardConfig(normal_pellet_reward=-1.0)
        with self.assertRaisesRegex(ValueError, "power_pellet_reward"):
            RewardConfig(power_pellet_reward=-1.0)
        with self.assertRaisesRegex(ValueError, "completion_reward"):
            RewardConfig(completion_reward=-1.0)
        with self.assertRaisesRegex(ValueError, "ratio_threshold"):
            RewardConfig(
                nearest_pellet_remaining_ratio_threshold=1.1
            )

    def test_step256_reward_config_contract(self) -> None:
        config = (
            Path(__file__).parents[1]
            / "configs"
            / "level1"
            / "train"
            / "level1_live_state_step256_100update_group12_8gpu.yaml"
        ).read_text(encoding="utf-8")
        for expected in (
            "total_train_epochs: 50",
            "validation_contract: sampled12_uniform_shaped",
            "use_base_reward: false",
            "normal_pellet_reward: 1.0",
            "power_pellet_reward: 1.0",
            "completion_reward: 50.0",
            "step_penalty: 0.05",
            "wall_penalty: 0.5",
            "nearest_pellet_alpha: 0.1",
            "nearest_pellet_remaining_ratio_threshold: 0.25",
            "nearest_pellet_skip_on_eat: true",
            "run_artifacts/level1_dataset_step256/train_hf",
            "run_artifacts/level1_dataset_step256/validation_hf",
        ):
            self.assertIn(expected, config)
        self.assertNotIn("revisit_penalty:", config)

    def test_summary_counts_acceptance_metrics(self) -> None:
        summary = summarize_episodes(
            [
                {
                    "pellet_clear_rate": 0.5,
                    "won": False,
                    "parse_failures": 0,
                    "canonical_action_violations": 0,
                    "steps": 10,
                    "total_base_reward": 5,
                    "total_shaped_reward": 9,
                    "final_score": 50,
                    "wall_collisions": 2,
                    "oscillation_returns": 1,
                    "normal_pellets_eaten": 97,
                    "normal_pellet_clear_rate": 0.5,
                    "normal_pellets_remaining": 97,
                    "power_pellets_remaining": 2,
                    "trajectory": [
                        {"action": "R", "base_reward": 0.0},
                        {"action": "R", "base_reward": 10.0},
                        {"action": "S", "base_reward": 0.0},
                    ],
                }
            ]
        )
        self.assertEqual(summary["average_pellet_clear_rate"], 0.5)
        self.assertEqual(summary["parse_failures"], 0)
        self.assertEqual(summary["average_final_score"], 50)
        self.assertEqual(summary["average_normal_pellets_eaten"], 97)
        self.assertEqual(summary["average_normal_pellet_clear_rate"], 0.5)
        self.assertEqual(summary["average_wall_hit_rate"], 0.2)
        self.assertEqual(summary["average_oscillation_rate"], 0.1)
        self.assertEqual(summary["action_counts"], {"U": 0, "D": 0, "L": 0, "R": 2, "S": 1})
        self.assertEqual(summary["max_no_progress_streak"], 1)


class WorkflowContractTests(unittest.TestCase):
    def test_native_workflow_is_direct_areal_rollout_workflow(self) -> None:
        from areal.api import RolloutWorkflow

        self.assertTrue(issubclass(PacmanNativeVisionWorkflow, RolloutWorkflow))

    def test_single_step_row_replays_prefix_then_scores_one_action(self) -> None:
        row = list(
            generate_single_step_rows(1, split="train", seed=0, offset=3)
        )[0]
        workflow = PacmanImageOnlyWorkflow(
            env_factory=PygamePacmanEnv,
            enable_thinking=False,
            image_prompt_style="minimal_v1",
            scripted_actions=["U"],
            scripted_completion_ids=["single-step-id"],
        )
        result = asyncio.run(workflow.run(row))
        self.assertEqual(set(result), {"single-step-id"})
        self.assertEqual(workflow.last_episode["steps"], 1)
        self.assertEqual(
            workflow.last_episode["state_prefix_actions"],
            row["state_prefix_actions"],
        )
        self.assertEqual(
            workflow.last_episode["terminal_reason"],
            "single_step_complete",
        )

    @staticmethod
    def _fake_native_processor():
        class FakeProcessor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.template_kwargs = kwargs
                return "processed prompt"

            def __call__(self, *, text, images, **kwargs):
                pixel_mean = float(np.asarray(images[0]).mean())
                return {
                    "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
                    "mm_token_type_ids": torch.tensor(
                        [[0, 1]], dtype=torch.long
                    ),
                    "pixel_values": torch.tensor(
                        [[pixel_mean]], dtype=torch.float32
                    ),
                    "image_grid_thw": torch.tensor(
                        [[1, 1, 1]], dtype=torch.long
                    ),
                }

        return FakeProcessor()

    def test_native_vision_processor_preserves_real_image_tensors(self) -> None:
        processor = self._fake_native_processor()
        workflow = PacmanNativeVisionWorkflow(
            gconfig=SimpleNamespace(),
            tokenizer=SimpleNamespace(
                encode=lambda token, **_: [
                    {"U": 40, "D": 41, "L": 42, "R": 43}[token]
                ]
            ),
            processor=processor,
            env_factory=OneStepEnv,
        )
        black = build_image_messages(
            encode_png(np.zeros((4, 4, 3), dtype=np.uint8))
        )
        white = build_image_messages(
            encode_png(np.full((4, 4, 3), 255, dtype=np.uint8))
        )
        _, _, black_processed, black_ids = workflow._process_messages(black)
        _, _, white_processed, white_ids = workflow._process_messages(white)

        self.assertEqual(black_ids, [10, 11])
        self.assertEqual(white_ids, [10, 11])
        self.assertFalse(
            torch.equal(
                black_processed["pixel_values"],
                white_processed["pixel_values"],
            )
        )
        self.assertIs(processor.template_kwargs["enable_thinking"], False)

    def test_native_vision_episode_returns_official_tensor_contract(self) -> None:
        processor = self._fake_native_processor()

        class FakeTokenizer:
            def encode(self, token, **kwargs):
                return [{"U": 40, "D": 41, "L": 42, "R": 43}[token]]

            def decode(self, token_ids, **kwargs):
                return "L"

        class FakeGConfig:
            def new(self, **kwargs):
                self.last_kwargs = kwargs
                return self

        class FakeModelRequest(SimpleNamespace):
            pass

        class FakeResponse:
            input_tokens = [10, 11]
            output_tokens = [42]
            output_logprobs = [-0.25]
            output_versions = [7]
            input_len = 2
            output_len = 1
            stop_reason = "stop"

        class FakeEngine:
            async def agenerate(self, request):
                self.request = request
                return FakeResponse()

        fake_areal_api = SimpleNamespace(ModelRequest=FakeModelRequest)
        fake_areal_image = SimpleNamespace(
            image2base64=lambda image: ["encoded-image"]
        )
        fake_areal_data = SimpleNamespace(
            concat_padded_tensors=lambda samples: samples[0]
        )
        workflow = PacmanNativeVisionWorkflow(
            gconfig=FakeGConfig(),
            tokenizer=FakeTokenizer(),
            processor=processor,
            env_factory=OneStepEnv,
            enable_thinking=False,
            image_prompt_style="minimal_v1",
            action_token_choice=True,
        )
        engine = FakeEngine()
        with patch.dict(
            sys.modules,
            {
                "areal.api": fake_areal_api,
                "areal.utils.image": fake_areal_image,
                "areal.utils.data": fake_areal_data,
            },
        ):
            result = asyncio.run(
                workflow.arun_episode(
                    engine,
                    make_episode_row(1, split="train"),
                )
            )

        self.assertEqual(
            set(result),
            {
                "input_ids",
                "mm_token_type_ids",
                "loss_mask",
                "pacman_action_mask_bits",
                "logprobs",
                "versions",
                "attention_mask",
                "rewards",
                "multi_modal_input",
            },
        )
        self.assertEqual(result["input_ids"].tolist(), [[10, 11, 42]])
        self.assertEqual(
            result["mm_token_type_ids"].tolist(), [[0, 1, 0]]
        )
        self.assertEqual(result["loss_mask"].tolist(), [[0, 0, 1]])
        self.assertEqual(
            result["pacman_action_mask_bits"].tolist(), [[0, 0, 15]]
        )
        self.assertEqual(result["logprobs"].tolist(), [[0.0, 0.0, -0.25]])
        self.assertEqual(result["versions"].tolist(), [[-1, -1, 7]])
        self.assertEqual(engine.request.image_data, ["encoded-image"])
        self.assertEqual(engine.request.input_ids, [10, 11])
        self.assertEqual(
            engine.request.metadata["allowed_token_ids"],
            [40, 41, 42, 43],
        )
        self.assertEqual(
            engine.request.metadata["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        json.dumps(engine.request.vision_msg_vllm)
        self.assertEqual(
            engine.request.vision_msg_vllm[0][1]["content"][1]["image_url"],
            {"url": "placeholder"},
        )
        self.assertIn(
            "pixel_values", result["multi_modal_input"][0]
        )

    def test_native_vllm_adapter_forwards_no_thinking_and_action_mask(
        self,
    ) -> None:
        class FakeVLLMBackend:
            def build_generation_request(self, req, with_lora, version):
                return SimpleNamespace(payload={"temperature": 0.7})

        fake_vllm_remote = SimpleNamespace(VLLMBackend=FakeVLLMBackend)
        with patch.dict(
            sys.modules,
            {
                "areal": SimpleNamespace(),
                "areal.engine": SimpleNamespace(),
                "areal.engine.vllm_remote": fake_vllm_remote,
            },
        ):
            install_vllm_allowed_token_ids_adapter()

        req = SimpleNamespace(
            metadata={
                "allowed_token_ids": [40, 43],
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        request = FakeVLLMBackend().build_generation_request(
            req,
            with_lora=False,
            version=0,
        )

        self.assertEqual(request.payload["allowed_token_ids"], [40, 43])
        self.assertEqual(
            request.payload["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_workflow_defaults_to_thinking_disabled_and_rejects_true(self) -> None:
        captured = {}

        class CapturingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                captured.update(options)
                return ModelTurn("S", "capture-id", messages)

        workflow = CapturingWorkflow(env_factory=OneStepEnv)
        asyncio.run(workflow.run(make_episode_row(1, split="test")))
        self.assertIs(captured["enable_thinking"], False)
        self.assertIs(workflow.last_episode["decoding"]["enable_thinking"], False)

        with self.assertRaisesRegex(ValueError, "enable_thinking=false"):
            asyncio.run(
                CapturingWorkflow(env_factory=OneStepEnv).run(
                    make_episode_row(1, split="test"),
                    enable_thinking=True,
                )
            )

    def test_model_request_explicitly_disables_thinking(self) -> None:
        captured = {}

        class FakeResponse:
            id = "response-1"
            choices = [
                SimpleNamespace(
                    message=SimpleNamespace(content="S", reasoning_content=None)
                )
            ]

            def model_dump(self, mode):
                return {"id": self.id, "mode": mode}

        class FakeCompletions:
            async def create(self, **request):
                captured.update(request)
                return FakeResponse()

        class FakeClient:
            def __init__(self, **kwargs):
                captured["client"] = kwargs
                self.chat = SimpleNamespace(completions=FakeCompletions())

            async def close(self):
                captured["closed"] = True

        fake_openai = SimpleNamespace(AsyncOpenAI=FakeClient)
        messages = [{"role": "user", "content": "test"}]
        with (
            patch.dict(sys.modules, {"openai": fake_openai}),
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=fake_action_tokenizer(),
            ),
        ):
            workflow = PacmanImageOnlyWorkflow(
                open_action_mask=True,
                tokenizer_path="test-tokenizer",
            )
            turn = asyncio.run(
                workflow._call_model(
                    messages,
                    model="test-model",
                    base_url="http://example.invalid/v1",
                    api_key="test",
                    temperature=0.7,
                    top_p=0.95,
                    max_completion_tokens=3,
                    enable_thinking=False,
                    open_action_mask=True,
                    current_open_actions=["U", "R"],
                )
            )

        self.assertEqual(turn.completion, "S")
        self.assertEqual(captured["temperature"], 0.7)
        self.assertEqual(captured["top_p"], 0.95)
        self.assertEqual(
            captured["extra_body"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertEqual(
            captured["extra_body"]["structured_outputs"],
            {"choice": ["U", "R"]},
        )
        self.assertEqual(
            captured["extra_body"]["allowed_token_ids"],
            [40, 43],
        )
        self.assertTrue(captured["closed"])

    def test_workflow_passes_current_open_actions_to_generation(self) -> None:
        captured = {}

        class CapturingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                captured["open_actions"] = options["current_open_actions"]
                return ModelTurn(
                    captured["open_actions"][0],
                    "capture-id",
                    messages,
                )

        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=fake_action_tokenizer(),
        ):
            workflow = CapturingWorkflow(
                env_factory=OneStepEnv,
                open_action_mask=True,
                tokenizer_path="test-tokenizer",
            )
        asyncio.run(workflow.run(make_episode_row(1, split="test")))

        self.assertTrue(captured["open_actions"])
        self.assertNotIn("S", captured["open_actions"])
        self.assertTrue(
            set(captured["open_actions"]).issubset({"U", "D", "L", "R"})
        )
        step = workflow.last_episode["trajectory"][0]
        self.assertEqual(step["open_action_mask"], captured["open_actions"])
        self.assertIn(step["action"], step["open_action_mask"])
        self.assertFalse(step["wall_collision"])

    def test_duplicate_dataset_row_writes_distinct_trajectory_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = make_episode_row(1, split="validation")
            for _ in range(2):
                workflow = PacmanImageOnlyWorkflow(env_factory=OneStepEnv)
                asyncio.run(
                    workflow.run(
                        row,
                        scripted_actions=["S"],
                        trajectory_dir=directory,
                    )
                )
            files = sorted(Path(directory).glob("*.json"))
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]

        self.assertEqual(len(files), 2)
        self.assertEqual({payload["id"] for payload in payloads}, {row["id"]})
        self.assertEqual(
            len({payload["trajectory_sample_id"] for payload in payloads}),
            2,
        )

    def test_workflow_uses_real_maapacman_and_records_completion_reward(self) -> None:
        row = make_episode_row(1, split="train")
        workflow = PacmanImageOnlyWorkflow(env_factory=OneStepEnv)
        result = asyncio.run(
            workflow.run(
                row,
                scripted_actions=["L"],
                scripted_completion_ids=["completion-1"],
            )
        )
        self.assertEqual(result, {"completion-1": workflow.last_episode["trajectory"][0]["shaped_reward"]})
        payload = workflow.last_episode
        assert payload is not None
        self.assertEqual(payload["env_api_version"], "1.0")
        self.assertEqual(payload["terminal_reason"], "test_complete")
        self.assertEqual(payload["trajectory"][0]["action"], "L")
        self.assertEqual(payload["backend"], "original-pygame")
        self.assertEqual(
            payload["pacman_python_revision"],
            row["env"]["pacman_python_revision"],
        )
        self.assertTrue(payload["renderer_revision"].startswith("pacman-python:"))
        self.assertIn("score", payload["trajectory"][0])
        self.assertIn("pygame_mode", payload["trajectory"][0])
        audit_trajectory(payload)

    def test_nearest_pellet_shaping_uses_hidden_maapacman_topology(self) -> None:
        row = make_episode_row(1, split="train")
        level = load_bundled_level(1)
        route = route_to_nearest(level, level.pacman_start, level.pellets)
        workflow = PacmanImageOnlyWorkflow(env_factory=OneStepEnv)
        asyncio.run(
            workflow.run(
                row,
                scripted_actions=[route[0].value],
                nearest_pellet_alpha=1.0,
            )
        )
        payload = workflow.last_episode
        assert payload is not None
        step = payload["trajectory"][0]
        self.assertEqual(payload["nearest_pellet_alpha"], 1.0)
        self.assertEqual(
            payload["nearest_pellet_topology_revision"], level.revision
        )
        self.assertIsNotNone(step["nearest_pellet_distance_before"])
        self.assertIsNotNone(step["nearest_pellet_distance_after"])
        self.assertEqual(
            step["nearest_pellet_progress_reward"],
            step["nearest_pellet_distance_before"]
            - step["nearest_pellet_distance_after"],
        )
        combined_prompt = payload["system_prompt"] + payload["user_instruction"]
        self.assertNotIn("distance", combined_prompt.lower())
        audit_trajectory(payload)

    def test_parse_failure_is_terminal_and_never_forwarded(self) -> None:
        class CountingEnv(PygamePacmanEnv):
            steps_called = 0

            def step(self, action):
                type(self).steps_called += 1
                return super().step(action)

        workflow = PacmanImageOnlyWorkflow(env_factory=CountingEnv)
        reward = asyncio.run(
            workflow.run(
                make_episode_row(1, split="train"),
                scripted_actions=["Action: R"],
            )
        )
        self.assertEqual(reward, -50.0)
        self.assertEqual(CountingEnv.steps_called, 0)
        self.assertEqual(workflow.last_episode["terminal_reason"], "parse_failed")

    def test_environment_closes_on_success_and_exception(self) -> None:
        created = []

        class TrackingEnv(PygamePacmanEnv):
            def __init__(self, config):
                super().__init__(config)
                self.close_called = False
                created.append(self)

            def close(self):
                self.close_called = True
                super().close()

        class TrackingOneStepEnv(TrackingEnv):
            def step(self, action):
                image, reward, _, _, info = super().step(action)
                info = dict(info)
                info.update(
                    terminated=True,
                    truncated=False,
                    terminal_reason="test_complete",
                )
                return image, reward, True, False, info

        workflow = PacmanImageOnlyWorkflow(env_factory=TrackingOneStepEnv)
        asyncio.run(
            workflow.run(
                make_episode_row(1, split="train"),
                scripted_actions=["S"],
            )
        )
        self.assertTrue(created[-1].close_called)
        workflow = PacmanImageOnlyWorkflow(env_factory=TrackingEnv)
        with self.assertRaises(RuntimeError):
            asyncio.run(
                workflow.run(
                    make_episode_row(2, split="train"),
                    scripted_actions=["S"],
                )
            )
        self.assertTrue(created[-1].close_called)

    def test_model_receives_same_png_hash_recorded_in_trajectory(self) -> None:
        captured = {}

        class CapturingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                captured["messages"] = messages
                return ModelTurn("S", "capture-id", messages)

        workflow = CapturingWorkflow(env_factory=OneStepEnv)
        asyncio.run(workflow.run(make_episode_row(1, split="test")))
        url = captured["messages"][1]["content"][1]["image_url"]["url"]
        sent_png = base64.b64decode(url.split(",", 1)[1])
        recorded = workflow.last_episode["trajectory"][0]["observation_png_sha256"]
        self.assertEqual(png_sha256(sent_png), recorded)

    def test_live_state_workflow_records_prompt_context_and_history(self) -> None:
        workflow = PacmanImageOnlyWorkflow(env_factory=TwoStepEnv)
        asyncio.run(
            workflow.run(
                make_episode_row(1, split="test"),
                scripted_actions=["U", "S"],
                image_prompt_style="live_state_v3",
            )
        )
        payload = workflow.last_episode
        assert payload is not None
        first, second = payload["trajectory"]
        self.assertEqual(
            payload["observation_contract"],
            "screenshot_plus_live_state_and_navigation_history",
        )
        initial_position = first["observation_context"]["pacman_position"]
        self.assertEqual(len(initial_position), 2)
        self.assertIn("U", first["observation_context"]["blocked_actions"])
        self.assertEqual(
            second["observation_context"]["current_cell_exit_history"],
            ["U"],
        )
        self.assertEqual(
            second["observation_context"]["current_cell_exit_counts"],
            {"U": 1},
        )
        self.assertEqual(second["observation_context"]["recent_actions"], ["U"])
        self.assertEqual(
            second["observation_context"]["immediate_reverse_action"],
            "D",
        )
        self.assertIn(
            (
                f"At this cell ({initial_position[0]},{initial_position[1]}), "
                "directions already taken before: U."
            ),
            second["model_user_instruction"],
        )

    def test_live_state_cell_history_remains_bounded_for_full_horizon(self) -> None:
        workflow = PacmanImageOnlyWorkflow()
        asyncio.run(
            workflow.run(
                make_episode_row(1, split="test"),
                scripted_actions=["U"] * PRODUCTION_MAX_STEPS,
                image_prompt_style="live_state_v3",
            )
        )
        payload = workflow.last_episode
        assert payload is not None
        instructions = [
            step["model_user_instruction"] for step in payload["trajectory"]
        ]
        self.assertLess(max(map(len, instructions)), 2200)
        last_context = payload["trajectory"][-1]["observation_context"]
        self.assertEqual(last_context["current_cell_exit_history"], ["U"])
        self.assertEqual(
            last_context["current_cell_exit_counts"],
            {"U": PRODUCTION_MAX_STEPS - 1},
        )

    def test_trajectory_preserves_verbatim_model_response(self) -> None:
        raw_response = {
            "id": "response-raw-1",
            "choices": [
                {
                    "message": {
                        "content": "S",
                        "reasoning_content": None,
                    }
                }
            ],
        }
        request_extra_body = {
            "structured_outputs": {"choice": ["U", "D", "L", "R", "S"]},
            "chat_template_kwargs": {"enable_thinking": False},
        }

        class CapturingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                return ModelTurn(
                    "S",
                    "response-raw-1",
                    messages,
                    reasoning_content=None,
                    raw_response=raw_response,
                    request_extra_body=request_extra_body,
                )

        with tempfile.TemporaryDirectory() as directory:
            workflow = CapturingWorkflow(env_factory=OneStepEnv)
            asyncio.run(
                workflow.run(
                    make_episode_row(1, split="test"),
                    trajectory_dir=directory,
                    enable_thinking=False,
                )
            )
            payload = json.loads(next(Path(directory).glob("*.json")).read_text())

        step = payload["trajectory"][0]
        self.assertEqual(step["completion"], "S")
        self.assertIsNone(step["reasoning_content"])
        self.assertEqual(step["raw_model_response"], raw_response)
        self.assertEqual(step["request_extra_body"], request_extra_body)

    def test_oracle_clears_level_through_production_workflow(self) -> None:
        actions = oracle_actions()
        self.assertEqual(len(actions), PRODUCTION_MAX_STEPS)
        workflow = PacmanImageOnlyWorkflow()
        asyncio.run(
            workflow.run(
                make_episode_row(1, split="test"),
                scripted_actions=actions,
            )
        )
        payload = workflow.last_episode
        assert payload is not None
        self.assertTrue(payload["won"])
        self.assertEqual(payload["terminal_reason"], "all_normal_pellets")
        self.assertEqual(payload["normal_pellets_remaining"], 0)
        self.assertEqual(payload["power_pellets_remaining"], 2)
        self.assertAlmostEqual(payload["pellet_clear_rate"], 194 / 196)

    def test_nearest_pellet_shaping_telescopes_over_full_oracle(self) -> None:
        actions = oracle_actions()
        workflow = PacmanImageOnlyWorkflow()
        asyncio.run(
            workflow.run(
                make_episode_row(1, split="test"),
                scripted_actions=actions,
                nearest_pellet_alpha=1.0,
            )
        )
        payload = workflow.last_episode
        assert payload is not None
        trajectory = payload["trajectory"]
        self.assertEqual(len(trajectory), PRODUCTION_MAX_STEPS)
        self.assertTrue(payload["won"])
        self.assertTrue(
            all(
                step["nearest_pellet_distance_before"] is not None
                and step["nearest_pellet_distance_after"] is not None
                for step in trajectory
            )
        )
        progress_total = sum(
            step["nearest_pellet_progress_reward"] for step in trajectory
        )
        self.assertEqual(
            progress_total,
            trajectory[0]["nearest_pellet_distance_before"]
            - trajectory[-1]["nearest_pellet_distance_after"],
        )
        self.assertEqual(
            trajectory[-1]["nearest_pellet_distance_after"], 0
        )
        audit_trajectory(payload)

    def test_cancellation_closes_worker_and_removes_runtime(self) -> None:
        created = []
        model_started = asyncio.Event()

        class TrackingEnv(PygamePacmanEnv):
            def __init__(self, config):
                super().__init__(config)
                self.close_called = False
                created.append(self)

            def close(self):
                self.close_called = True
                super().close()

        class BlockingWorkflow(PacmanImageOnlyWorkflow):
            async def _call_model(self, messages, **options):
                model_started.set()
                await asyncio.Future()

        async def scenario() -> None:
            workflow = BlockingWorkflow(env_factory=TrackingEnv)
            task = asyncio.create_task(
                workflow.run(make_episode_row(1, split="test"))
            )
            await asyncio.wait_for(model_started.wait(), timeout=15)
            runtime = created[-1].worker_runtime_dir
            self.assertIsNotNone(runtime)
            self.assertTrue(runtime.is_dir())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(created[-1].close_called)
            self.assertFalse(runtime.exists())

        asyncio.run(scenario())

    def test_production_workflow_does_not_import_private_environment(self) -> None:
        source = (
            Path(__file__).parents[1]
            / "areal_pacman"
            / "level1"
            / "workflow.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from maapacman.env import", source)
        self.assertNotIn("from .env import", source)
        self.assertNotIn("from areal_pacman.env import", source)

    def test_workflow_import_shim_preserves_public_classes(self) -> None:
        from areal_pacman.level1.workflow import (
            PacmanImageOnlyWorkflow as CanonicalImageOnlyWorkflow,
        )
        from areal_pacman.level1.workflow import (
            PacmanNativeVisionWorkflow as CanonicalNativeVisionWorkflow,
        )

        self.assertIs(PacmanImageOnlyWorkflow, CanonicalImageOnlyWorkflow)
        self.assertIs(PacmanNativeVisionWorkflow, CanonicalNativeVisionWorkflow)


class TrainerGenerationContractTests(unittest.TestCase):
    @staticmethod
    def _config() -> SimpleNamespace:
        return SimpleNamespace(
            enable_thinking=False,
            image_prompt_style="live_static_v2",
            tokenizer_path="test-tokenizer",
            legal_action_mask=False,
            open_action_mask=True,
            guided_action_choice=False,
            legal_action_choice=False,
            non_stay_legal_action_choice=False,
            non_backtracking_legal_action_choice=False,
            action_token_choice=True,
            completion_api="chat",
            parse_failure_penalty=-50,
            reward_mode="sparse",
            route_shaping_scale=1.0,
            safe_progress_alpha=1.0,
            step_penalty=1.0,
            wall_penalty=1.0,
            nearest_pellet_alpha=0.0,
            observation_mode="rgb",
            vision_tile_size=32,
            store_observation_images=False,
            trajectory_dir="run_artifacts/trajectories",
        )

    def test_train_and_eval_generation_kwargs_are_independent(self) -> None:
        config = self._config()
        train_generation = SimpleNamespace(
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,
            max_new_tokens=3,
        )
        eval_generation = SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            max_new_tokens=3,
        )

        training = _build_workflow_kwargs(config, train_generation)
        evaluation = _build_workflow_kwargs(config, eval_generation)

        self.assertEqual(training["temperature"], 0.7)
        self.assertEqual(evaluation["temperature"], 0.0)
        self.assertEqual(training["top_p"], 0.95)
        self.assertEqual(evaluation["top_p"], 1.0)
        self.assertIs(training["enable_thinking"], False)
        self.assertIs(evaluation["enable_thinking"], False)
        self.assertEqual(training["image_prompt_style"], "live_static_v2")
        self.assertEqual(evaluation["image_prompt_style"], "live_static_v2")
        self.assertIs(training["open_action_mask"], True)
        self.assertIs(evaluation["open_action_mask"], True)

    def test_group12_config_declares_true_greedy_validation(self) -> None:
        config = (
            Path(__file__).parents[1]
            / "configs"
            / "level1"
            / "archive"
            / "level1_image_overfit_4epoch_group12_8gpu.yaml"
        ).read_text(encoding="utf-8")
        eval_block = config.split("eval_gconfig:", 1)[1].split("actor:", 1)[0]
        self.assertIn("n_samples: 1", eval_block)
        self.assertIn("greedy: true", eval_block)
        self.assertIn("temperature: 0.0", eval_block)
        self.assertIn("top_p: 1.0", eval_block)

    def test_anticollapse_config_has_exact_four_update_contract(self) -> None:
        config = (
            Path(__file__).parents[1]
            / "configs"
            / "level1"
            / "archive"
            / "level1_image_anticollapse_4update_group12_8gpu.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("total_train_epochs: 2", config)
        self.assertIn("image_prompt_style: minimal_v1", config)
        self.assertEqual(
            config.count("admin_api_key: ${oc.env:AREAL_ADMIN_API_KEY}"), 2
        )
        train_generation = config.split("gconfig:", 1)[1].split(
            "eval_gconfig:", 1
        )[0]
        self.assertIn("n_samples: 12", train_generation)
        self.assertIn("temperature: 0.7", train_generation)
        eval_generation = config.split("eval_gconfig:", 1)[1].split(
            "actor:", 1
        )[0]
        self.assertIn("n_samples: 1", eval_generation)
        self.assertIn("greedy: true", eval_generation)
        self.assertIn("temperature: 0.0", eval_generation)
        actor = config.split("actor:", 1)[1].split("ref:", 1)[0]
        self.assertIn("lr: 1.5e-6", actor)
        self.assertIn("kl_ctl: 0.01", actor)
        self.assertIn("optimizer_dtype: float32", actor)
        self.assertIn("enable_offload: false", config)
        ref = config.split("ref:", 1)[1].split("vllm:", 1)[0]
        self.assertIn('backend: "vllm:d4p1t1"', config)
        self.assertIn('backend: "fsdp:d4p1t1"', actor)
        self.assertIn("backend: ${actor.backend}", ref)
        self.assertIn("optimizer_dtype: bfloat16", ref)
        self.assertIn("optimizer: null", ref)
        self.assertIn("offload: false", ref)
        self.assertIn("offload_params: true", ref)
        self.assertIn("type: colocation", ref)
        self.assertIn("target: actor", ref)
        vllm = config.split("vllm:", 1)[1].split("train_dataset:", 1)[0]
        self.assertIn("gpu_memory_utilization: 0.65", vllm)
        self.assertEqual(config.count("freq_steps: 1"), 2)

    def test_progress_config_is_alpha_one_isolated_followup(self) -> None:
        config = (
            Path(__file__).parents[1]
            / "configs"
            / "level1"
            / "archive"
            / "level1_image_progress_4update_group12_8gpu.yaml"
        ).read_text(encoding="utf-8")
        for expected in (
            "nearest_pellet_alpha: 1.0",
            "validation_contract: sampled12_and_greedy1",
            "total_train_epochs: 2",
            "image_prompt_style: minimal_v1",
            "n_samples: 12",
            "temperature: 0.7",
            "lr: 1.5e-6",
            "kl_ctl: 0.01",
            'backend: "vllm:d4p1t1"',
            'backend: "fsdp:d4p1t1"',
            "offload_params: true",
        ):
            self.assertIn(expected, config)
        self.assertEqual(
            config.count("admin_api_key: ${oc.env:AREAL_ADMIN_API_KEY}"), 2
        )
        eval_generation = config.split("eval_gconfig:", 1)[1].split(
            "actor:", 1
        )[0]
        self.assertIn("n_samples: 12", eval_generation)
        self.assertIn("greedy: false", eval_generation)
        self.assertIn("temperature: 0.7", eval_generation)
        self.assertIn("top_p: 0.95", eval_generation)

    def test_live_state_config_is_uniform_sampled_without_distance_shaping(self) -> None:
        config = (
            Path(__file__).parents[1]
            / "configs"
            / "level1"
            / "archive"
            / "level1_live_state_4update_group12_8gpu.yaml"
        ).read_text(encoding="utf-8")
        for expected in (
            "validation_contract: sampled12_uniform",
            "image_prompt_style: live_state_v3",
            "nearest_pellet_alpha: 0.0",
            "total_train_epochs: 8",
            "n_samples: 12",
            "temperature: 0.7",
            "top_p: 1.0",
            "logprobs_mode: processed_logprobs",
        ):
            self.assertIn(expected, config)
        train_generation = config.split("gconfig:", 1)[1].split(
            "eval_gconfig:", 1
        )[0]
        eval_generation = config.split("eval_gconfig:", 1)[1].split(
            "actor:", 1
        )[0]
        for generation in (train_generation, eval_generation):
            self.assertIn("n_samples: 12", generation)
            self.assertIn("greedy: false", generation)
            self.assertIn("temperature: 0.7", generation)
            self.assertIn("top_p: 1.0", generation)

    def test_open_action_mask_logprob_patch_is_preflighted(self) -> None:
        root = Path(__file__).parents[1]
        patch = (
            root / "patches" / "areal_pacman_action_logprobs.patch"
        ).read_text(encoding="utf-8")
        launcher = (
            root / "scripts" / "level1" / "train" / "run_level1_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("_apply_pacman_action_mask", patch)
        self.assertIn('logprobs_mode: str = "raw_logprobs"', patch)
        self.assertIn("areal_pacman_action_logprobs_patch=ok", launcher)

    def test_fsdp_cpu_offload_patch_is_reproducible_and_preflighted(self) -> None:
        root = Path(__file__).parents[1]
        patch = (
            root / "patches" / "areal_fsdp_cpu_offload_empty_cache.patch"
        ).read_text(encoding="utf-8")
        launcher = (
            root / "scripts" / "level1" / "train" / "run_level1_training.sh"
        ).read_text(encoding="utf-8")
        marker = "CPUOffloadPolicy has already moved the persistent FSDP parameter"
        self.assertIn(marker, patch)
        self.assertIn("areal_fsdp_cpu_offload_empty_cache_patch=ok", launcher)
        self.assertIn(marker, launcher)

    def test_training_launcher_uses_ephemeral_nondefault_admin_key(self) -> None:
        launcher = (
            Path(__file__).parents[1]
            / "scripts"
            / "level1"
            / "train"
            / "run_level1_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("secrets.token_urlsafe(32)", launcher)
        self.assertIn('export AREAL_ADMIN_API_KEY', launcher)
        self.assertIn(
            '[[ "${AREAL_ADMIN_API_KEY}" == "areal-admin-key" ]]', launcher
        )
        self.assertNotIn("echo \"${AREAL_ADMIN_API_KEY}\"", launcher)

    def test_training_launcher_prefers_official_areal_worktree(self) -> None:
        launcher = (
            Path(__file__).parents[1]
            / "scripts"
            / "level1"
            / "train"
            / "run_level1_training.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('AREAL_ROOT="${AREAL_ROOT:-${OWNER_ROOT}/xinglu/AReaL}"', launcher)
        self.assertIn(
            'export PYTHONPATH="${AREAL_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"',
            launcher,
        )
        self.assertIn("AReaL import escaped official checkout", launcher)

    def test_official_areal_single_step_smoke_contract(self) -> None:
        config = (
            Path(__file__).parents[1]
            / "configs"
            / "level1"
            / "archive"
            / "level1_official_areal_smoke_3b_4gpu.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("total_train_epochs: 2", config)
        self.assertIn("n_samples: 4", config)
        self.assertIn("group_size: ${gconfig.n_samples}", config)
        self.assertIn("offload_params: false", config)
        self.assertIn(
            "allow_unoffloaded_actor_colocated_ref_for_smoke: true", config
        )
        train = config.split("train_dataset:", 1)[1].split("valid_dataset:", 1)[0]
        self.assertIn("batch_size: 4", train)
        self.assertIn(
            "image_prompt_style: wall_avoidance_axis_v3", config
        )
        self.assertIn("action_token_choice: false", config)
        self.assertIn("n_gpus_per_node: 4", config)
        self.assertIn('backend: "vllm:d2p1t1"', config)
        self.assertIn('backend: "fsdp:d2p1t1"', config)

    def test_run_evaluator_can_resume_and_matches_sampled_served_name(self) -> None:
        evaluator = (
            Path(__file__).parents[1]
            / "scripts"
            / "level1"
            / "evaluate"
            / "evaluate_level1_run.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'if [[ -f "${EVAL_ROOT}/${label}/greedy.json" ]]', evaluator
        )
        self.assertIn('local served_model_name="${4:-${label}}"', evaluator)
        self.assertIn('--served-model-name "${served_model_name}"', evaluator)
        self.assertIn(
            '"${sampled_port}" \\\n    "${sampled_label}"',
            evaluator,
        )
        self.assertIn('SAMPLED_LABELS=("${LABELS[@]}")', evaluator)
        self.assertNotIn("comparison.partial.json", evaluator)
        self.assertIn(
            'sampled12_already_complete=${sampled_label}', evaluator
        )
        self.assertIn("best_sampled_label=${BEST_SAMPLED_LABEL}", evaluator)
        self.assertIn("best_greedy_label=${BEST_GREEDY_LABEL}", evaluator)
        self.assertIn("--require-complete-dual", evaluator)


if __name__ == "__main__":
    unittest.main()
