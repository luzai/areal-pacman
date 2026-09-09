from __future__ import annotations

import importlib
import copy
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("AREAL_ALLOW_DEFAULT_ADMIN_KEY", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


class _EpochOffsetSampler:
    def __init__(self, sampler, offset: int):
        self._sampler = sampler
        self._offset = offset

    def set_epoch(self, epoch: int) -> None:
        self._sampler.set_epoch(epoch + self._offset)

    def __getattr__(self, name):
        return getattr(self._sampler, name)


class _EpochOffsetDataLoader:
    """Start a fresh weights-only branch at the matching logical dataset epoch."""

    def __init__(self, dataloader, epoch_offset: int):
        self._dataloader = dataloader
        self.sampler = _EpochOffsetSampler(dataloader.sampler, epoch_offset)
        self.batch_size = dataloader.batch_size

    def __iter__(self):
        return iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader)

    def state_dict(self):
        return self._dataloader.state_dict()

    def load_state_dict(self, state):
        return self._dataloader.load_state_dict(state)

    def __getattr__(self, name):
        return getattr(self._dataloader, name)


def _validate_reward_objective_contract(config) -> None:
    contract = config.reward_objective_contract
    if contract == "legacy":
        return
    if contract != "step_local_raw_v1":
        raise ValueError(f"unsupported reward_objective_contract: {contract}")
    if not config.workflow.endswith(".PacmanNativeVisionWorkflow"):
        raise ValueError(
            "step_local_raw_v1 is restricted to PacmanNativeVisionWorkflow"
        )
    if config.actor.reward_norm is not None:
        raise ValueError(
            "step_local_raw_v1 requires actor.reward_norm=null because each "
            "tensor sample is one environment decision, not one rollout"
        )
    if config.actor.adv_norm is not None:
        raise ValueError(
            "step_local_raw_v1 requires actor.adv_norm=null so dense rewards "
            "retain their state-local sign"
        )


def _jsonable(value):
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _probe_token_count(value) -> int:
    """Return a token count without fetching an RTensor from remote storage."""
    if hasattr(value, "numel"):
        return int(value.numel())
    data = getattr(value, "data", None)
    if hasattr(data, "numel"):
        return int(data.numel())
    return len(value)


def _configure_explicit_resume(trainer, requested_start_step: int) -> dict:
    from areal.api import StepInfo

    steps_per_epoch = len(trainer.train_dataloader)
    if requested_start_step <= 0:
        raise ValueError("requested cold-resume start must be positive")
    if trainer.recover_info is None:
        last_global_step = requested_start_step - 1
        last = StepInfo(
            epoch=last_global_step // steps_per_epoch,
            epoch_step=last_global_step % steps_per_epoch,
            global_step=last_global_step,
            steps_per_epoch=steps_per_epoch,
        )
        epoch_offset = requested_start_step // steps_per_epoch
        trainer.train_dataloader = _EpochOffsetDataLoader(
            trainer.train_dataloader,
            epoch_offset=epoch_offset,
        )
        trainer.recover_info = SimpleNamespace(last_step_info=last)
        source = "iter16_hf_weights_plus_reconstructed_state"
        print(
            "COLD_RESUME_ACTOR_WEIGHTS_ONLY "
            + json.dumps(
                {
                    "last_global_step": last.global_step,
                    "next_global_step": last.next().global_step,
                    "next_iteration": last.next().global_step + 1,
                    "epoch_offset": epoch_offset,
                    "optimizer": "fresh",
                    "scheduler": "constant_reconstructed",
                    "rng": "fresh_deterministic_seed_not_original",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        last = trainer.recover_info.last_step_info
        if last.next().global_step < requested_start_step:
            raise RuntimeError(
                "existing recover checkpoint predates requested resume: "
                f"last={last.global_step}, requested={requested_start_step}"
            )
        source = "new_branch_dcp_model_optimizer_plus_reconstructed_scheduler_rng"

    completed_updates = trainer.recover_info.last_step_info.global_step + 1
    trainer._onload_model(trainer.actor, role="actor")
    for _ in range(completed_updates):
        trainer.actor.step_lr_scheduler()
    trainer.actor.set_version(completed_updates)
    trainer.rollout.set_version(completed_updates)
    if trainer.eval_rollout is not None:
        trainer.eval_rollout.set_version(completed_updates)
    audit = trainer.actor._custom_function_call(
        "runtime_state_audit",
        expected_phase="cuda",
        rpc_meta={"broadcast": False},
    )
    trainer._offload_model(trainer.actor, role="actor")

    if hasattr(trainer.rollout, "staleness_manager"):
        manager = trainer.rollout.staleness_manager
    else:
        manager = trainer.rollout.workflow_executor.staleness_manager
    if manager is not None:
        manager.on_version_recovered(completed_updates)

    result = {
        "source": source,
        "completed_updates": completed_updates,
        "next_global_step": trainer.recover_info.last_step_info.next().global_step,
        "next_iteration": trainer.recover_info.last_step_info.next().global_step + 1,
        "actor_audit": _jsonable(audit),
    }
    print("EXPLICIT_RESUME_AUDIT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def _expected_safety_probe_max_steps() -> int:
    value = int(os.getenv("MAAPACMAN_SAFETY_PROBE_MAX_STEPS", "700"))
    if value <= 0:
        raise ValueError("MAAPACMAN_SAFETY_PROBE_MAX_STEPS must be positive")
    return value


def _run_safety_probe(trainer, config, workflow_kwargs: dict) -> dict:
    # A recovered StatefulDataLoader may be parked exactly at an epoch boundary.
    # Production rollout preparation cycles finite loaders, so the probe must do
    # the same instead of treating the boundary's StopIteration as a failure.
    from areal.utils.data import cycle_dataloader
    batch = next(cycle_dataloader(trainer.train_dataloader))
    if not isinstance(batch, list) or not batch:
        raise RuntimeError(f"unexpected safety-probe batch type: {type(batch)!r}")
    row = batch[0]
    env = row.get("env", {})
    expected_max_steps = _expected_safety_probe_max_steps()
    if int(env.get("max_steps", -1)) != expected_max_steps:
        raise RuntimeError(
            "probe row does not match expected max_steps "
            f"{expected_max_steps}: {env!r}"
        )

    trajectories = []
    submitted = 0
    # RolloutController returns one trajectory per submitted data item; group_size
    # describes a packed workflow result rather than the number of submissions.
    # Production prepare_batch transparently replaces rejected workflows. Mirror
    # that behavior here so a single workflow rejection cannot yield a zero-sample
    # safety test.
    for probe_round in range(1, 5):
        needed = 4 - len(trajectories)
        if needed == 0:
            break
        probe_rows = []
        for offset in range(needed):
            probe_row = copy.deepcopy(row)
            probe_row["id"] = (
                f"{row.get('id', 'probe')}-safety-r{probe_round}-n{offset}"
            )
            probe_rows.append(probe_row)
        submitted += len(probe_rows)
        accepted = trainer.rollout.rollout_batch(
            probe_rows,
            workflow=config.workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=1,
        )
        trajectories.extend(accepted)
        print(
            "SAFETY_PROBE_ROLLOUT_ROUND "
            + json.dumps(
                {
                    "round": probe_round,
                    "submitted": len(probe_rows),
                    "accepted": len(accepted),
                    "accepted_total": len(trajectories),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if len(trajectories) != 4:
        raise RuntimeError(
            "safety probe requires exactly four accepted long rollouts; "
            f"accepted={len(trajectories)} submitted={submitted}"
        )

    stress_copies = int(os.getenv("MAAPACMAN_SAFETY_LOGP_STRESS_COPIES", "4"))
    if stress_copies < 4 or stress_copies > 48 or stress_copies % 4 != 0:
        raise RuntimeError(
            "MAAPACMAN_SAFETY_LOGP_STRESS_COPIES must be a multiple of four "
            f"between 4 and 48; got {stress_copies}"
        )
    logp_trajectories = [
        copy.deepcopy(trajectories[index % len(trajectories)])
        for index in range(stress_copies)
    ]

    trainer._onload_model(trainer.ref, role="ref")
    ref_logps = trainer._compute_logp_in_rpc_chunks(
        trainer.ref, logp_trajectories, role="ref_probe"
    )
    if ref_logps is None or len(ref_logps) != len(logp_trajectories):
        raise RuntimeError("reference compute_logp returned the wrong result count")
    for trajectory, logp in zip(logp_trajectories, ref_logps):
        trajectory["ref_logp"] = logp
    for trajectory, stress_trajectory in zip(trajectories, logp_trajectories):
        trajectory["ref_logp"] = stress_trajectory["ref_logp"]
    trainer._offload_model(trainer.ref, role="ref")

    trainer._onload_model(trainer.actor, role="actor")
    prox_logps = trainer._compute_logp_in_rpc_chunks(
        trainer.actor, logp_trajectories, role="actor_probe"
    )
    if prox_logps is None or len(prox_logps) != len(logp_trajectories):
        raise RuntimeError("actor compute_logp returned the wrong result count")
    for trajectory, logp in zip(logp_trajectories, prox_logps):
        trajectory["prox_logp"] = logp
    for trajectory, stress_trajectory in zip(trajectories, logp_trajectories):
        trajectory["prox_logp"] = stress_trajectory["prox_logp"]
    advantages = trainer.actor.compute_advantages(trajectories)
    backward_audit = trainer.actor.safety_probe_backward(advantages)
    state_audit = trainer.actor._custom_function_call(
        "runtime_state_audit",
        expected_phase="cuda",
        rpc_meta={"broadcast": False},
    )
    trainer._offload_model(trainer.actor, role="actor")

    rollout_summaries = []
    for index, trajectory in enumerate(trajectories):
        input_ids = trajectory.get("input_ids")
        token_count = _probe_token_count(input_ids)
        rollout_summaries.append(
            {
                "index": index,
                "token_count": token_count,
                "reward": _jsonable(trajectory.get("rewards")),
                "version": _jsonable(trajectory.get("versions")),
            }
        )

    report = {
        "status": "ok",
        "rollout_count": len(trajectories),
        "rollout_submitted": submitted,
        "max_steps": expected_max_steps,
        "logp_stress_trajectory_copies": len(logp_trajectories),
        "logp_rpc_chunk_size": int(
            os.getenv("MAAPACMAN_LOGP_RPC_CHUNK_SIZE", "0")
        ),
        "reference_compute_logp": "ok",
        "actor_compute_logp": "ok",
        "backward_without_optimizer_step": "ok",
        "rollouts": rollout_summaries,
        "backward_audit": _jsonable(backward_audit),
        "state_audit": _jsonable(state_audit),
    }
    report_path = Path(config.artifact_root) / "safety_probe" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("SAFETY_PROBE_OK " + json.dumps(report, sort_keys=True), flush=True)
    return report

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
        completion_reward=getattr(config, "completion_reward", 0.0),
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
    if workflow_cls.__module__ not in {
        "areal_pacman.workflow",
        "areal_pacman.level1.workflow",
    }:
        raise ValueError("production config must use areal_pacman.workflow")

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

        config, _ = load_expr_config(
            ["--config", str(config_path)], PacmanAgentConfig
        )
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
    safety_probe = os.getenv("MAAPACMAN_SAFETY_PROBE") == "1"
    explicit_resume_step = int(
        os.getenv("MAAPACMAN_EXPLICIT_RESUME_GLOBAL_STEP", "0")
    )
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
        if explicit_resume_step:
            _configure_explicit_resume(trainer, explicit_resume_step)
        if safety_probe:
            _run_safety_probe(trainer, config, workflow_kwargs)
            return
        trainer.train(
            workflow=config.workflow,
            eval_workflow=config.eval_workflow,
            workflow_kwargs=workflow_kwargs,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
