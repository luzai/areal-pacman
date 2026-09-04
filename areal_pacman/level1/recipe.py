"""CPU-only configuration contract shared by data preparation and training."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from maapacman.env.ghost_modes import validate_ghost_mode


@dataclass
class EnvironmentConfig:
    ghost_mode: str = "normal"
    max_steps: int = 256

    def __post_init__(self):
        validate_ghost_mode(self.ghost_mode)
        if type(self.max_steps) is not int or self.max_steps not in {
            32,
            256,
            512,
            2000,
        }:
            raise ValueError("environment.max_steps must be 32, 256, 512, or 2000")


@dataclass
class DatasetGenerationConfig:
    train_episodes: int = 8
    validation_episodes: int = 2
    seed: int = 0

    def __post_init__(self):
        for name in ("train_episodes", "validation_episodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(
                    f"dataset_generation.{name} must be a positive integer"
                )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("dataset_generation.seed must be a non-negative integer")


def load_recipe_settings(
    path: Path,
) -> tuple[EnvironmentConfig, DatasetGenerationConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return (
        EnvironmentConfig(**raw.get("environment", {})),
        DatasetGenerationConfig(**raw.get("dataset_generation", {})),
    )
