from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export representative level-1 trajectory JSON as Markdown."
    )
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_episodes(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    episodes = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    ]
    if not episodes:
        raise ValueError(f"no trajectories found in {directory}")
    return episodes


def _rank(item: tuple[Path, dict[str, Any]]) -> tuple[float, ...]:
    payload = item[1]
    return (
        float(payload["normal_pellet_clear_rate"]),
        float(payload["total_shaped_reward"]),
        -float(payload["wall_collisions"]),
        -float(payload.get("oscillation_returns", 0)),
    )


def _tokens(values: Any) -> str:
    return ",".join(str(value) for value in (values or [])) or "-"


def _markdown(
    label: str,
    source: Path,
    payload: dict[str, Any],
) -> str:
    lines = [
        f"# {label}",
        "",
        f"- Source JSON: `{source}`",
        f"- Model split: `{payload['split']}`",
        f"- Final score: `{payload['final_score']}`",
        (
            "- Normal pellets eaten: "
            f"`{payload['normal_pellets_eaten']}/"
            f"{payload['normal_pellets_initial']}`"
        ),
        (
            "- Normal-pellet clear rate: "
            f"`{100 * float(payload['normal_pellet_clear_rate']):.2f}%`"
        ),
        f"- Level cleared: `{payload['won']}`",
        f"- Total shaped reward: `{payload['total_shaped_reward']}`",
        f"- Wall hits: `{payload['wall_collisions']}`",
        f"- Oscillation returns: `{payload.get('oscillation_returns', 0)}`",
        f"- Episode steps: `{payload['steps']}`",
        (
            "- Decoding: "
            f"`temperature={payload['decoding']['temperature']}, "
            f"top_p={payload['decoding']['top_p']}, "
            f"enable_thinking={payload['decoding']['enable_thinking']}`"
        ),
        "",
        "## Prompt",
        "",
        "System prompt:",
        "",
        "```text",
        str(payload["system_prompt"]),
        "```",
        "",
        "First-step dynamic user observation:",
        "",
        "```text",
        str(payload["trajectory"][0]["model_user_instruction"]),
        "```",
        "",
        "## Full action trajectory",
        "",
        (
            "| Step | Position | Open | Blocked | Tried here | Try counts | "
            "Preferred | Recent actions | Output | Normal eaten | Reward | Wall | "
            "Oscillation |"
        ),
        (
            "|---:|---|---|---|---|---|---|---|---|---:|---:|:---:|:---:|"
        ),
    ]
    for step in payload["trajectory"]:
        context = step.get("observation_context") or {}
        position = context.get("pacman_position") or ["?", "?"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(step["step"]),
                    f"({position[0]},{position[1]})",
                    _tokens(context.get("open_actions")),
                    _tokens(context.get("blocked_actions")),
                    _tokens(context.get("current_cell_exit_history")),
                    _tokens(
                        f"{key}={value}"
                        for key, value in (
                            context.get("current_cell_exit_counts") or {}
                        ).items()
                    ),
                    _tokens(context.get("preferred_open_actions")),
                    _tokens(context.get("recent_actions")),
                    str(step.get("completion", "")).replace("|", "\\|"),
                    str(step.get("normal_pellets_eaten", "")),
                    f"{float(step.get('shaped_reward', 0.0)):g}",
                    "Y" if step.get("wall_collision") else "",
                    "Y" if step.get("oscillation_return") else "",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    comparison_path = args.eval_root / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    best_label = str(comparison["best_checkpoint"])
    base = sorted(
        _load_episodes(args.eval_root / "base" / "sampled12_trajectories"),
        key=_rank,
    )
    trained = sorted(
        _load_episodes(
            args.eval_root / best_label / "sampled12_trajectories"
        ),
        key=_rank,
    )
    selections = {
        "base_typical": base[len(base) // 2],
        f"{best_label}_best": trained[-1],
        f"{best_label}_typical": trained[len(trained) // 2],
        f"{best_label}_worst": trained[0],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "best_checkpoint": best_label,
        "examples": {},
    }
    for label, (source, payload) in selections.items():
        output = args.output_dir / f"{label}.md"
        output.write_text(
            _markdown(label, source, payload),
            encoding="utf-8",
            newline="\n",
        )
        manifest["examples"][label] = {
            "markdown": str(output),
            "source_json": str(source),
            "normal_pellet_clear_rate": payload[
                "normal_pellet_clear_rate"
            ],
            "normal_pellets_eaten": payload["normal_pellets_eaten"],
            "total_shaped_reward": payload["total_shaped_reward"],
            "level_cleared": payload["won"],
        }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
