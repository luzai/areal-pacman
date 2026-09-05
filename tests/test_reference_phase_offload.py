"""CPU-only policy checks; these do not claim a GPU memory/throughput gate."""

from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from train_areal import _validate_actor_colocated_reference_offload

ROOT = Path(__file__).resolve().parents[1]


def config(*, enable=True, actor=True, ref=True, native=False, override=False):
    return SimpleNamespace(
        enable_offload=enable,
        allow_unoffloaded_actor_colocated_ref_for_smoke=override,
        actor=SimpleNamespace(backend="fsdp:d4p1t1", offload=actor),
        ref=SimpleNamespace(
            backend="fsdp:d4p1t1", offload=ref,
            fsdp=SimpleNamespace(offload_params=native),
            scheduling_strategy=SimpleNamespace(type="colocation", target="actor"),
        ),
    )


@pytest.mark.parametrize("enable,actor,ref", list(product((False, True), repeat=3)))
def test_phase_only_requires_all_three_switches(enable, actor, ref):
    value = config(enable=enable, actor=actor, ref=ref)
    if enable and actor and ref:
        _validate_actor_colocated_reference_offload(value)
    else:
        with pytest.raises(ValueError, match="actor-colocated reference"):
            _validate_actor_colocated_reference_offload(value)


@pytest.mark.parametrize("stage", [1, 2])
def test_actual_release_yaml_uses_phase_only_without_smoke_bypass(stage, monkeypatch):
    monkeypatch.setenv("CURRICULUM1_CHECKPOINT", "/test/complete-c1")
    value = OmegaConf.load(ROOT / f"configs/level1/train/curriculum{stage}.yaml")
    assert not value.actor.fsdp.offload_params
    assert not value.ref.fsdp.offload_params
    assert not value.get("allow_unoffloaded_actor_colocated_ref_for_smoke", False)
    _validate_actor_colocated_reference_offload(value)


@pytest.mark.parametrize("actor_backend,ref_backend", [
    ("megatron:d4p1t1", "megatron:d4p1t1"),
    ("fsdp:d4p1t1", "megatron:d4p1t1"),
    ("megatron:d4p1t1", "fsdp:d4p1t1"),
    ("fsdp:d4p1t1", "fsdp:d3p1t1"),
    ("fsdp:d4p1t1", "fsdp:d4p1t2"),
])
def test_phase_only_rejects_unverified_or_mismatched_backends(actor_backend, ref_backend):
    value = config()
    value.actor.backend, value.ref.backend = actor_backend, ref_backend
    with pytest.raises(ValueError, match="matching FSDP backends"):
        _validate_actor_colocated_reference_offload(value)


def test_native_fsdp_policy_still_has_the_existing_acceptance_path():
    _validate_actor_colocated_reference_offload(
        config(enable=False, actor=False, ref=False, native=True)
    )


def test_explicit_legacy_small_model_override_is_preserved():
    _validate_actor_colocated_reference_offload(
        config(enable=False, actor=False, ref=False, override=True)
    )


@pytest.mark.parametrize("kind,target", [("separation", None), ("colocation", "rollout")])
def test_other_reference_placements_are_outside_this_guard(kind, target):
    value = config(enable=False, actor=False, ref=False)
    value.ref.scheduling_strategy.type = kind
    value.ref.scheduling_strategy.target = target
    _validate_actor_colocated_reference_offload(value)


def test_no_reference_is_outside_this_guard():
    value = config()
    value.ref = None
    _validate_actor_colocated_reference_offload(value)
