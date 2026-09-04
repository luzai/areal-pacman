from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("AREAL_ALLOW_DEFAULT_ADMIN_KEY", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


def _load_workflow(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _yaml_scalar(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*([^#\r\n]+)", text)
    if not match:
        raise ValueError(f"missing required config field: {key}")
    return match.group(1).strip().strip('"\'')


def _config_override(args: list[str], key: str) -> str | None:
    """Return the final exact-key CLI override, matching AReaL precedence."""

    prefix = f"{key}="
    matches = [argument[len(prefix) :] for argument in args if argument.startswith(prefix)]
    return matches[-1] if matches else None


def _apply_smoke_updates(args: list[str]) -> tuple[list[str], int | None]:
    """Translate the release smoke flag into AReaL's exact step limit."""

    cleaned: list[str] = []
    raw_updates: str | None = None
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--smoke-updates":
            if raw_updates is not None:
                raise ValueError("--smoke-updates may be specified only once")
            if index + 1 >= len(args):
                raise ValueError("--smoke-updates requires a positive integer")
            raw_updates = args[index + 1]
            index += 2
            continue
        if argument.startswith("--smoke-updates="):
            if raw_updates is not None:
                raise ValueError("--smoke-updates may be specified only once")
            raw_updates = argument.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(argument)
        index += 1

    if raw_updates is None:
        return cleaned, None
    if re.fullmatch(r"[1-9][0-9]*", raw_updates) is None:
        raise ValueError("--smoke-updates requires a positive integer")
    if _config_override(cleaned, "total_train_steps") is not None:
        raise ValueError(
            "--smoke-updates cannot be combined with total_train_steps"
        )
    smoke_updates = int(raw_updates)
    cleaned.append(f"total_train_steps={smoke_updates}")
    return cleaned, smoke_updates


def _backend_degree(backend: str) -> int:
    match = re.search(r":d(\d+)p\d+t\d+$", backend)
    if match is None:
        raise ValueError(f"unsupported production backend allocation: {backend}")
    return int(match.group(1))


def _validate_reward_objective_contract(config) -> None:
    contract = config.reward_objective_contract
    if contract == "legacy":
        return
    if contract not in {
        "step_local_raw_v1",
        "option_return_raw_v1",
        "episode_return_group_v1",
    }:
        raise ValueError(f"unsupported reward_objective_contract: {contract}")
    if (
        contract in {"option_return_raw_v1", "episode_return_group_v1"}
        and not getattr(config, "edward_options", False)
    ):
        raise ValueError(f"{contract} requires edward_options=true")
    if not config.workflow.endswith(".PacmanNativeVisionWorkflow"):
        raise ValueError(
            f"{contract} requires PacmanNativeVisionWorkflow"
        )
    if contract in {"option_return_raw_v1", "episode_return_group_v1"}:
        if getattr(config.actor, "use_sapo_loss", False) or getattr(
            config.actor, "use_cispo_loss", False
        ):
            raise ValueError(
                f"{contract} requires the PPO/GRPO surrogate; SAPO and CISPO "
                "do not support equal-episode reduction"
            )
        if int(getattr(config.actor, "ppo_n_minibatches", 1)) != 1:
            raise ValueError(
                f"{contract} requires actor.ppo_n_minibatches=1 so one "
                "optimizer update averages all complete episodes together"
            )
    if contract == "episode_return_group_v1":
        reward_norm = config.actor.reward_norm
        if reward_norm is None:
            raise ValueError(
                "episode_return_group_v1 requires actor.reward_norm"
            )
        n_samples = int(config.gconfig.n_samples)
        if n_samples != 12:
            raise ValueError(
                "episode_return_group_v1 requires exactly 12 complete episodes "
                "per initial maze state"
            )
        if (
            reward_norm.mean_level != "group"
            or reward_norm.std_level != "group"
            or int(reward_norm.group_size) != n_samples
            or bool(reward_norm.mean_leave1out)
        ):
            raise ValueError(
                "episode_return_group_v1 requires group mean/std normalization, "
                "group_size=gconfig.n_samples, and mean_leave1out=false"
            )
        if getattr(config.actor, "overlong_reward_penalty", False):
            raise ValueError(
                "episode_return_group_v1 does not support per-completion "
                "overlong reward penalties"
            )
        if getattr(config, "critic", None) is not None or getattr(
            config, "teacher", None
        ) is not None:
            raise ValueError(
                "episode_return_group_v1 requires critic=null and teacher=null"
            )
    elif config.actor.reward_norm is not None:
        raise ValueError(
            f"{contract} requires actor.reward_norm=null; normalization across "
            "unrelated maze states changes the sign of state-local returns"
        )
    if config.actor.adv_norm is not None:
        raise ValueError(
            f"{contract} requires actor.adv_norm=null; batch centering creates "
            "a global action-token baseline across unrelated maze states"
        )


def _build_workflow_kwargs(
    config, generation_config, *, training: bool = True
) -> dict[str, object]:
    kwargs = dict(
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        max_tokens=generation_config.max_tokens,
        max_completion_tokens=generation_config.max_new_tokens,
        tokenizer_path=config.tokenizer_path,
        enable_thinking=config.enable_thinking,
        image_prompt_style=config.image_prompt_style,
        legal_action_mask=config.legal_action_mask,
        open_action_mask=config.open_action_mask,
        edward_options=getattr(config, "edward_options", False),
        objective_encoding=getattr(config, "objective_encoding", "legacy"),
        reward_objective_contract=(
            getattr(config, "reward_objective_contract", "legacy")
            if training
            else "evaluation_only_v1"
        ),
        guided_action_choice=config.guided_action_choice,
        legal_action_choice=config.legal_action_choice,
        non_stay_legal_action_choice=config.non_stay_legal_action_choice,
        non_backtracking_legal_action_choice=config.non_backtracking_legal_action_choice,
        action_token_choice=config.action_token_choice,
        completion_api=config.completion_api,
        parse_failure_penalty=getattr(config, "parse_failure_penalty", -50),
        contract_violation_return=getattr(
            config, "contract_violation_return", -1.0
        ),
        reward_mode=config.reward_mode,
        route_shaping_scale=config.route_shaping_scale,
        safe_progress_alpha=config.safe_progress_alpha,
        step_penalty=config.step_penalty,
        step_penalty_cleared_ratio_scale=getattr(
            config,
            "step_penalty_cleared_ratio_scale",
            0.0,
        ),
        wall_penalty=config.wall_penalty,
        use_base_reward=getattr(config, "use_base_reward", True),
        normal_pellet_reward=getattr(
            config,
            "normal_pellet_reward",
            0.0,
        ),
        power_pellet_reward=getattr(
            config,
            "power_pellet_reward",
            0.0,
        ),
        ghost_reward=getattr(config, "ghost_reward", 0.0),
        fruit_reward=getattr(config, "fruit_reward", 0.0),
        reward_recipe_version=getattr(
            config,
            "reward_recipe_version",
            "maapacman-level1-event-reward-v3",
        ),
        death_penalty=getattr(config, "death_penalty", 0.0),
        completion_reward=getattr(config, "completion_reward", 0.0),
        safety_refusal_penalty=getattr(
            config, "safety_refusal_penalty", 0.0
        ),
        nearest_pellet_alpha=config.nearest_pellet_alpha,
        nearest_pellet_remaining_ratio_threshold=(
            getattr(
                config,
                "nearest_pellet_remaining_ratio_threshold",
                1.0,
            )
        ),
        nearest_pellet_scale_by_cleared_ratio=getattr(
            config,
            "nearest_pellet_scale_by_cleared_ratio",
            False,
        ),
        nearest_pellet_skip_on_eat=getattr(
            config,
            "nearest_pellet_skip_on_eat",
            False,
        ),
        observation_mode=config.observation_mode,
        vision_tile_size=config.vision_tile_size,
        store_observation_images=config.store_observation_images,
        trajectory_dir=(
            os.getenv("PACMAN_TRAJECTORY_DIR") or config.trajectory_dir
        ),
    )
    if getattr(config, "workflow", "").endswith(
        ".PacmanNativeVisionWorkflow"
    ):
        kwargs.update(
            gconfig=generation_config,
            tokenizer=config.tokenizer_path,
            processor=config.tokenizer_path,
        )
    return kwargs


def _production_dry_run(
    config_path: Path,
    *,
    validate_areal: bool = False,
    config_args: list[str] | None = None,
) -> bool:
    text = config_path.read_text(encoding="utf-8")
    recipe_version = _yaml_scalar(text, "recipe_version")
    if recipe_version not in {
        "maapacman-level1-v1",
        "maapacman-level1-ghost-v2",
        "maapacman-level1-ghostdoor-v3",
    }:
        return False
    from areal_pacman.level1.level1_dataset import validate_episode_row
    from maapacman.env import PygamePacmanEnv

    epochs = int(_yaml_scalar(text, "total_train_epochs"))
    workflow_path = _yaml_scalar(text, "workflow")
    if epochs < 2:
        raise ValueError("production level-1 training must run at least two epochs")
    workflow_cls = _load_workflow(workflow_path)
    if workflow_cls.__module__ not in {
        "areal_pacman.workflow",
        "areal_pacman.level1.workflow",
    }:
        raise ValueError("production config must use areal_pacman.workflow")

    effective_args = config_args or ["--config", str(config_path)]
    dataset_overrides = [
        _config_override(effective_args, "train_dataset.path"),
        _config_override(effective_args, "valid_dataset.path"),
    ]
    override_present = [value is not None for value in dataset_overrides]
    if any(override_present) and not all(override_present):
        raise ValueError(
            "train_dataset.path and valid_dataset.path must be overridden together"
        )
    if all(override_present):
        if any(not value for value in dataset_overrides):
            raise ValueError("dataset path overrides must be non-empty")
        dataset_matches = [str(path) for path in dataset_overrides]
    else:
        dataset_matches = re.findall(r"(?m)^\s+path:\s*([^#\r\n]+)", text)
        if len(dataset_matches) < 2:
            raise ValueError("config must declare train and validation dataset paths")
    rows = 0
    for raw_path in dataset_matches[-2:]:
        dataset_path = Path(raw_path.strip().strip('"\''))
        if not dataset_path.is_absolute():
            dataset_path = (Path(__file__).resolve().parent / dataset_path).resolve()
        if dataset_path.suffix == ".jsonl":
            jsonl = dataset_path
        elif dataset_path.name.endswith("_hf"):
            jsonl = dataset_path.with_name(dataset_path.name.removesuffix("_hf") + ".jsonl")
        else:
            jsonl = dataset_path.with_suffix(".jsonl")
        if not jsonl.is_file():
            raise FileNotFoundError(jsonl)
        import json

        for line in jsonl.read_text(encoding="utf-8").splitlines():
            validate_episode_row(json.loads(line))
            rows += 1
    env = PygamePacmanEnv()
    try:
        spec = env.spec
    finally:
        env.close()
    print("dry_run=ok")
    print(f"workflow={workflow_cls.__module__}.{workflow_cls.__name__}")
    print(f"config={config_path.resolve()}")
    print(f"dataset_rows={rows}")
    print(f"total_train_epochs={epochs}")
    print(f"env_api_version={spec.api_version}")
    print(f"level_revision={spec.level_revision}")
    print(f"action_tokens={','.join(spec.action_tokens)}")
    if validate_areal:
        from areal.api.cli_args import load_expr_config
        from areal_pacman.synthetic.configs import PacmanAgentConfig

        config, _ = load_expr_config(effective_args, PacmanAgentConfig)
        _validate_reward_objective_contract(config)
        gpu_count = config.cluster.n_gpus_per_node
        if gpu_count not in (4, 6, 8):
            raise ValueError(
                "production config must request an accepted 4-, 6-, or "
                "8-GPU topology"
            )
        rollout_degree = _backend_degree(config.rollout.backend)
        actor_degree = _backend_degree(config.actor.backend)
        if not config.rollout.backend.startswith("vllm:"):
            raise ValueError("production rollout backend must use vLLM")
        if not config.actor.backend.startswith("fsdp:"):
            raise ValueError("production actor backend must use FSDP")
        allocated_gpus = rollout_degree + actor_degree
        if config.gconfig.n_samples < rollout_degree:
            raise ValueError("GRPO group size must be at least the rollout degree")
        if config.gconfig.n_samples % rollout_degree:
            raise ValueError("GRPO group size must divide evenly across rollout workers")
        if config.train_dataset.batch_size % actor_degree:
            raise ValueError("train batch size must divide evenly across actor workers")
        if config.enable_thinking is not False:
            raise ValueError("production image-only config must disable thinking")
        if recipe_version == "maapacman-level1-ghostdoor-v3":
            if not config.edward_options:
                raise ValueError("Edward v3 recipe must enable edward_options")
            if config.open_action_mask or config.action_token_choice:
                raise ValueError(
                    "Edward v3 replaces atomic action-token constraints"
                )
            if config.objective_encoding != "edward-option-code-v1":
                raise ValueError(
                    "Edward v3 requires objective_encoding=edward-option-code-v1"
                )
            if (
                config.gconfig.min_new_tokens != 1
                or config.gconfig.max_new_tokens != 1
                or config.eval_gconfig.min_new_tokens != 1
                or config.eval_gconfig.max_new_tokens != 1
            ):
                raise ValueError(
                    "Edward option-code train and evaluation generation must "
                    "use exactly one new token"
                )
            if config.gconfig.top_p != 1.0:
                raise ValueError("Edward objective training requires top_p=1.0")
            if config.actor.temperature != config.gconfig.temperature:
                raise ValueError(
                    "Edward rollout and actor temperatures must match"
                )
        if config.validation_contract == "greedy1":
            if not config.eval_gconfig.greedy:
                raise ValueError("greedy1 validation must use greedy decoding")
            if config.eval_gconfig.temperature != 0.0:
                raise ValueError("greedy1 validation temperature must be zero")
            if config.eval_gconfig.top_p != 1.0:
                raise ValueError("greedy1 validation top_p must be one")
            if config.eval_gconfig.n_samples != 1:
                raise ValueError(
                    "greedy1 validation must use exactly one sample"
                )
        elif config.validation_contract in (
            "sampled12_and_greedy1",
            "sampled12_uniform",
            "sampled12_uniform_shaped",
        ):
            if config.eval_gconfig.greedy:
                raise ValueError(
                    "sampled validation must not use greedy decoding"
                )
            if config.eval_gconfig.n_samples != 12:
                raise ValueError(
                    "sampled12 validation must use exactly 12 samples"
                )
            if config.eval_gconfig.n_samples != config.gconfig.n_samples:
                raise ValueError(
                    "sampled validation count must match the GRPO group size"
                )
            if (
                config.eval_gconfig.temperature
                != config.gconfig.temperature
                or config.eval_gconfig.top_p != config.gconfig.top_p
            ):
                raise ValueError(
                    "sampled validation temperature/top_p must match training"
                )
            if config.validation_contract in (
                "sampled12_uniform",
                "sampled12_uniform_shaped",
            ):
                if config.image_prompt_style != "live_state_v3":
                    raise ValueError(
                        "uniform sampled validation requires live_state_v3"
                    )
            if config.validation_contract == "sampled12_uniform":
                if config.nearest_pellet_alpha != 0.0:
                    raise ValueError(
                        "sampled12_uniform disables nearest-pellet shaping"
                    )
        else:
            raise ValueError(
                "unsupported production validation_contract: "
                f"{config.validation_contract}"
            )
        if config.actor.kl_ctl > 0:
            if config.ref is None:
                raise ValueError("positive KL requires a reference engine")
            if config.actor.optimizer_dtype not in ("float32", "bfloat16"):
                raise ValueError(
                    "the trainable actor optimizer dtype must be float32 or bfloat16"
                )
            if config.ref.optimizer is not None:
                raise ValueError(
                    "the KL reference engine must remain frozen"
                )
            if config.ref.optimizer_dtype != "bfloat16":
                raise ValueError(
                    "positive-KL level-1 training requires bfloat16 reference "
                    "parameter storage for the accepted split 4+4/3+3 topology"
                )
            if config.ref.temperature != config.actor.temperature:
                raise ValueError(
                    "reference and actor temperatures must match"
                )
            if config.enable_offload != config.ref.offload:
                raise ValueError(
                    "positive-KL reference offload requires enable_offload and "
                    "ref.offload to be enabled or disabled together"
                )
            ref_degree = _backend_degree(config.ref.backend)
            if config.ref.scheduling_strategy.type == "colocation":
                if config.ref.scheduling_strategy.target not in ("actor", "rollout"):
                    raise ValueError(
                        "a colocated reference engine must target actor or rollout"
                    )
                if (
                    config.ref.scheduling_strategy.target == "actor"
                    and not config.ref.fsdp.offload_params
                    and not config.allow_unoffloaded_actor_colocated_ref_for_smoke
                ):
                    raise ValueError(
                        "an actor-colocated reference must use native FSDP "
                        "parameter offload unless the explicit small-model "
                        "smoke override is enabled"
                    )
            else:
                allocated_gpus += ref_degree
        if allocated_gpus != gpu_count:
            raise ValueError(
                "production backend allocation must exactly consume the "
                f"{gpu_count} requested GPUs, got {allocated_gpus}"
            )
        print("areal_config=ok")
        print(f"cluster_gpus={config.cluster.n_gpus_per_node}")
        print(f"rollout_backend={config.rollout.backend}")
        print(f"actor_backend={config.actor.backend}")
        print(f"allocated_gpus={allocated_gpus}")
        print(f"actor_optimizer_dtype={config.actor.optimizer_dtype}")
        print(f"reference_optimizer_dtype={config.ref.optimizer_dtype if config.ref else None}")
        print(f"reference_offload={config.ref.offload if config.ref else False}")
        print(f"validation_contract={config.validation_contract}")
        print(
            "validation_decoding="
            f"n{config.eval_gconfig.n_samples},"
            f"temperature{config.eval_gconfig.temperature},"
            f"top_p{config.eval_gconfig.top_p},"
            f"thinking{config.enable_thinking}"
        )
    return True


def main(args: list[str]) -> None:
    args, smoke_updates = _apply_smoke_updates(args)
    dry_run = "--dry-run" in args
    args = [arg for arg in args if arg != "--dry-run"]
    validate_areal = "--validate-areal" in args
    args = [arg for arg in args if arg != "--validate-areal"]

    config_path = None
    if "--config" in args:
        index = args.index("--config")
        if index + 1 >= len(args):
            raise ValueError("--config requires a path")
        config_path = Path(args[index + 1])
    if dry_run and config_path is not None and _production_dry_run(
        config_path,
        validate_areal=validate_areal,
        config_args=args,
    ):
        if smoke_updates is not None:
            print(f"smoke_updates={smoke_updates}")
        return

    from areal import PPOTrainer
    from areal.api.cli_args import load_expr_config
    from areal.dataset import get_custom_dataset
    from areal.utils.hf_utils import load_hf_tokenizer
    from areal_pacman.synthetic.configs import PacmanAgentConfig
    from datasets import load_from_disk

    config, _ = load_expr_config(args, PacmanAgentConfig)
    if smoke_updates is not None:
        print(f"smoke_updates={smoke_updates}")

    if dry_run:
        dataset_path = Path(config.train_dataset.path)
        dataset = load_from_disk(str(dataset_path))
        workflow_cls = _load_workflow(config.workflow)
        print("dry_run=ok")
        print(f"workflow={workflow_cls.__module__}.{workflow_cls.__name__}")
        print(f"train_dataset={dataset_path}")
        print(f"train_rows={len(dataset)}")
        print(f"actor_path={config.actor.path}")
        print(f"scheduler={config.scheduler.type}")
        print(f"reward_mode={config.reward_mode}")
        print(f"safe_progress_alpha={config.safe_progress_alpha}")
        print(f"nearest_pellet_alpha={config.nearest_pellet_alpha}")
        return

    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = _build_workflow_kwargs(
        config, config.gconfig, training=True
    )
    eval_workflow_kwargs = _build_workflow_kwargs(
        config, config.eval_gconfig, training=False
    )

    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
