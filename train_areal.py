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


def _backend_degree(backend: str) -> int:
    match = re.search(r":d(\d+)p\d+t\d+$", backend)
    if match is None:
        raise ValueError(f"unsupported production backend allocation: {backend}")
    return int(match.group(1))


def _build_workflow_kwargs(config, generation_config) -> dict[str, object]:
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
        guided_action_choice=config.guided_action_choice,
        legal_action_choice=config.legal_action_choice,
        non_stay_legal_action_choice=config.non_stay_legal_action_choice,
        non_backtracking_legal_action_choice=config.non_backtracking_legal_action_choice,
        action_token_choice=config.action_token_choice,
        completion_api=config.completion_api,
        parse_failure_penalty=config.parse_failure_penalty,
        reward_mode=config.reward_mode,
        route_shaping_scale=config.route_shaping_scale,
        safe_progress_alpha=config.safe_progress_alpha,
        step_penalty=config.step_penalty,
        wall_penalty=config.wall_penalty,
        nearest_pellet_alpha=config.nearest_pellet_alpha,
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
    config_path: Path, *, validate_areal: bool = False
) -> bool:
    text = config_path.read_text(encoding="utf-8")
    if "recipe_version: maapacman-level1-v1" not in text:
        return False
    from areal_pacman.level1.level1_dataset import validate_episode_row
    from maapacman.env import PygamePacmanEnv

    epochs = int(_yaml_scalar(text, "total_train_epochs"))
    workflow_path = _yaml_scalar(text, "workflow")
    if epochs < 2:
        raise ValueError("production level-1 training must run at least two epochs")
    workflow_cls = _load_workflow(workflow_path)
    if workflow_cls.__module__ != "areal_pacman.workflow":
        raise ValueError("production config must use areal_pacman.workflow")

    dataset_matches = re.findall(r"(?m)^\s+path:\s*([^#\r\n]+)", text)
    if len(dataset_matches) < 2:
        raise ValueError("config must declare train and validation dataset paths")
    rows = 0
    for raw_path in dataset_matches[-2:]:
        dataset_path = Path(raw_path.strip().strip('"\''))
        if not dataset_path.is_absolute():
            dataset_path = (config_path.parent.parent / dataset_path).resolve()
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

        config, _ = load_expr_config(
            ["--config", str(config_path)], PacmanAgentConfig
        )
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
            if config.validation_contract == "sampled12_uniform":
                if config.image_prompt_style != "live_state_v3":
                    raise ValueError(
                        "sampled12_uniform requires live_state_v3"
                    )
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
            if config.enable_offload or config.ref.offload:
                raise ValueError(
                    "the accepted positive-KL topology keeps the BF16 reference "
                    "resident; TMS reference offload is not supported"
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
        config_path, validate_areal=validate_areal
    ):
        return

    from areal import PPOTrainer
    from areal.api.cli_args import load_expr_config
    from areal.dataset import get_custom_dataset
    from areal.utils.hf_utils import load_hf_tokenizer
    from areal_pacman.synthetic.configs import PacmanAgentConfig
    from datasets import load_from_disk

    config, _ = load_expr_config(args, PacmanAgentConfig)

    if dry_run:
        dataset_path = Path(config.train_dataset.path)
        dataset = load_from_disk(str(dataset_path))
        workflow_cls = _load_workflow(config.workflow)
        print(f"dry_run=ok")
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

    workflow_kwargs = _build_workflow_kwargs(config, config.gconfig)
    eval_workflow_kwargs = _build_workflow_kwargs(config, config.eval_gconfig)

    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
