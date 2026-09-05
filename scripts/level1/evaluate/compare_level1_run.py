"""Compare checkpoints only within matched, recorded evaluation protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

METRICS = (
    "average_final_score",
    "average_pellet_clear_rate",
    "average_normal_pellet_clear_rate",
    "average_wall_collisions",
    "average_episode_length",
    "max_no_progress_streak",
    "average_base_reward",
    "average_shaped_reward",
    "full_completions",
    "reasoning_turns",
    "action_counts",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampled-filename", default="sampled12.json")
    parser.add_argument("--require-complete-dual", action="store_true")
    parser.add_argument("--sampled-only", action="store_true")
    return parser.parse_args(argv)


def contract_hash(contract):
    return hashlib.sha256(
        json.dumps(
            contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read_summary(
    path, *, expected_episodes=None, expected_temperature=None, expected_top_p=None
):
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("evaluation_contract")
    if (
        not isinstance(protocol, dict)
        or protocol.get("version") != "pacman-release-evaluation-v1"
    ):
        raise ValueError(
            f"missing versioned evaluation protocol in {path}; regenerate old summaries"
        )
    if payload.get("evaluation_contract_sha256") != contract_hash(protocol):
        raise ValueError(f"evaluation protocol hash mismatch in {path}")
    for field in (
        "environment",
        "source_revisions",
        "harness",
        "prompt",
        "reward",
        "seeds",
        "generation_seeds",
    ):
        if field not in protocol:
            raise ValueError(f"missing evaluation protocol {field} in {path}")
    decoding = payload.get("decoding", {})
    declared = protocol["decoding"]
    if decoding.get("enable_thinking") is not False:
        raise ValueError(f"thinking was not disabled in {path}")
    if int(payload.get("reasoning_turns", 0)) != 0:
        raise ValueError(f"reasoning content was observed in {path}")
    count = len(protocol["seeds"]) * int(protocol["samples_per_seed"])
    if expected_episodes is not None and count != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episodes in {path}, protocol has {count}"
        )
    if int(payload.get("episodes", -1)) != count:
        raise ValueError(
            f"expected {count} episodes in {path}, got {payload.get('episodes')}"
        )
    for field, supplied in (
        ("temperature", expected_temperature),
        ("top_p", expected_top_p),
    ):
        expected = declared[field] if supplied is None else supplied
        if decoding.get(field) != expected or decoding.get(field) != declared[field]:
            raise ValueError(
                f"unexpected validation {field} in {path}: {decoding.get(field)}"
            )
    if decoding != declared:
        raise ValueError(f"decoding differs from recorded protocol in {path}")
    if payload.get("evaluation_completed") is not True:
        raise ValueError(f"unfinished evaluation in {path}")
    if not isinstance(payload.get("checkpoint"), dict) or not payload.get("model"):
        raise ValueError(f"missing checkpoint identity metadata in {path}")
    return payload


def compact(payload):
    return {
        "episodes": payload["episodes"],
        "decoding": payload["decoding"],
        "evaluation_contract_sha256": payload["evaluation_contract_sha256"],
        "win_rate": payload.get("win_rate"),
        "error_attempts": payload.get("error_attempts", 0),
        "completed_episodes": payload.get("completed_episodes"),
        **{metric: payload.get(metric) for metric in METRICS},
    }


def selection_key(payload):
    clear_rate = payload.get("average_normal_pellet_clear_rate")
    if clear_rate is None:
        clear_rate = payload["average_pellet_clear_rate"]
    return (
        float(payload["full_completions"]) / int(payload["episodes"]),
        float(clear_rate),
    )


def _natural_label(label):
    return tuple(
        int(item) if item.isdigit() else item for item in re.split(r"(\d+)", label)
    )


def _base_protocol(protocol):
    return {
        key: value
        for key, value in protocol.items()
        if key not in {"decoding", "samples_per_seed", "generation_seeds"}
    }


def compare(
    eval_root, sampled_filename, *, require_complete_dual=False, sampled_only=False
):
    greedy_raw = (
        {
            path.parent.name: read_summary(path)
            for path in sorted(eval_root.glob("*/greedy.json"))
        }
        if not sampled_only
        else {}
    )
    sampled_raw = {
        path.parent.name: read_summary(path)
        for path in sorted(eval_root.glob(f"*/{sampled_filename}"))
    }
    if not greedy_raw and not sampled_raw:
        raise ValueError(f"no evaluation summaries found under {eval_root}")
    labels = sorted(set(greedy_raw) | set(sampled_raw), key=_natural_label)
    missing_sampled = [label for label in labels if label not in sampled_raw]
    missing_greedy = [label for label in labels if label not in greedy_raw]
    if require_complete_dual and (missing_sampled or missing_greedy):
        raise ValueError(
            "missing sampled validation or greedy validation for labels: "
            + ", ".join(sorted(set(missing_sampled + missing_greedy)))
        )
    mode_protocols = {}
    common = None
    for mode, summaries in (("greedy", greedy_raw), ("matched_sampled", sampled_raw)):
        for label, payload in summaries.items():
            protocol = payload["evaluation_contract"]
            temperature = protocol["decoding"]["temperature"]
            if (mode == "greedy" and temperature != 0) or (
                mode == "matched_sampled" and temperature <= 0
            ):
                raise ValueError(
                    f"unexpected validation temperature for {mode}: {label}"
                )
            if mode in mode_protocols and protocol != mode_protocols[mode]:
                raise ValueError(
                    f"mixed evaluation protocols in {mode}; cannot compare {label}"
                )
            mode_protocols[mode] = protocol
            base = _base_protocol(protocol)
            if common is not None and base != common:
                raise ValueError(
                    "mixed environment/harness/reward/seed protocols across comparison"
                )
            common = base
    greedy = {
        label: compact(greedy_raw[label]) for label in labels if label in greedy_raw
    }
    sampled = {
        label: compact(sampled_raw[label]) for label in labels if label in sampled_raw
    }
    # Never select a release checkpoint using held-out test results.
    selection_allowed = common["purpose"] == "validation"

    def best(summaries, *, include_base=False):
        eligible = [
            label
            for label in labels
            if label in summaries
            and (include_base or label != "base")
            and summaries[label].get("error_attempts", 0) == 0
            and summaries[label].get("average_pellet_clear_rate") is not None
        ]
        return (
            max(eligible, key=lambda label: selection_key(summaries[label]))
            if selection_allowed and eligible
            else None
        )

    best_greedy = best(greedy)
    best_sampled = best(sampled)
    return {
        "contract": {
            **common,
            "max_steps": common["environment"]["max_steps"],
            "thinking": False,
            "greedy": mode_protocols.get("greedy"),
            "matched_sampled": {
                **(mode_protocols.get("matched_sampled") or {}),
                "labels": list(sampled),
            },
            "primary_checkpoint_selection": "matched_sampled",
            "greedy_is_separately_reported": True,
            "selection_order": [
                "win_rate",
                "average_normal_pellet_clear_rate",
                "earlier_checkpoint",
            ],
            "selection_allowed": selection_allowed,
            "minimum_win_rate": None,
        },
        "checkpoint_sources": {
            label: (sampled_raw.get(label) or greedy_raw[label])["checkpoint"]
            for label in labels
        },
        "greedy": greedy,
        "best_label": best_sampled or best_greedy,
        "best_sampled_label": best_sampled,
        "best_greedy_label": best_greedy,
        "best_overall_sampled_label": best(sampled, include_base=True),
        "base_is_release_candidate": False,
        "sampled12_by_label": sampled,
        "missing_sampled_labels": missing_sampled,
        "missing_greedy_labels": missing_greedy,
        "base_sampled12": sampled.get("base"),
        "best_sampled12": sampled.get(best_sampled),
        "note": "Ranks use validation only; no minimum win-rate gate. Infrastructure-error runs are reported but not selected.",
    }


def main():
    args = parse_args()
    payload = compare(
        args.eval_root,
        args.sampled_filename,
        require_complete_dual=args.require_complete_dual,
        sampled_only=args.sampled_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
