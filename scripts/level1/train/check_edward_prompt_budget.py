"""Fail closed when the worst Edward request exceeds the Qwen context budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from maapacman.planner import PlannerCandidate

from areal_pacman.level1.prompts import (
    EDWARD_OPTION_CODE_V1_SYSTEM_PROMPT,
    build_image_messages,
    encode_png,
)
from areal_pacman.level1.token_constraints import (
    OPTION_IDS,
    ObjectiveTokenConstraint,
)
from areal_pacman.level1.workflow import (
    PacmanNativeVisionWorkflow,
    _compact_edward_decision_prompt,
)


OBSERVATION_SHAPE = (400, 336, 3)


def worst_case_messages(tokenizer: object) -> tuple[list[dict], str]:
    candidates = tuple(
        PlannerCandidate(
            option_id=option_id,
            strategy={"C": "COLLECT", "A": "AVOID", "E": "ELIMINATE"}[
                option_id[0]
            ],
            target=(24, 20),
            first_action=("U", "D", "L", "R")[index % 4],
            route_distance=999,
            commit_moves={"C": 8, "A": 3, "E": 6}[option_id[0]],
            safety_margin=1_000_000,
            future_safe_exits=4,
            entity_id=index % 4 if option_id.startswith("E") else None,
        )
        for index, option_id in enumerate(OPTION_IDS)
    )
    state_context = {
        "pacman_position": [24, 20],
        "facing": "R",
        "pellets_remaining": 196,
        "maze_size": [25, 21],
        "ghosts": [
            {"id": index, "state": "vulnerable", "position": [24, 20]}
            for index in range(4)
        ],
        "edible_ticks": 360,
        "last_action": "R",
    }
    constraint = ObjectiveTokenConstraint.build(
        tokenizer, (candidate.option_id for candidate in candidates)
    )
    user_prompt = _compact_edward_decision_prompt(
        state_context, candidates, constraint
    )
    image = np.zeros(OBSERVATION_SHAPE, dtype=np.uint8)
    messages = build_image_messages(
        encode_png(image),
        prompt_style="live_state_v3",
        state_context=state_context,
    )
    messages[0]["content"] = EDWARD_OPTION_CODE_V1_SYSTEM_PROMPT
    messages[1]["content"][0]["text"] = user_prompt
    return messages, user_prompt


def measure(processor: object) -> dict[str, int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    messages, user_prompt = worst_case_messages(tokenizer)
    image, chat_messages = PacmanNativeVisionWorkflow._pil_and_chat_messages(
        messages
    )
    text = processor.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    processed = processor(
        text=[text], images=[image], padding=False, truncation=False, return_tensors="pt"
    )
    return {
        "input_tokens": int(processed["input_ids"].shape[-1]),
        "system_characters": len(EDWARD_OPTION_CODE_V1_SYSTEM_PROMPT),
        "user_characters": len(user_prompt),
        "candidate_count": len(OPTION_IDS),
        "ghost_count": 4,
        "image_height": OBSERVATION_SHAPE[0],
        "image_width": OBSERVATION_SHAPE[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    result = measure(processor)
    print(json.dumps(result, sort_keys=True))
    if result["input_tokens"] >= args.max_input_tokens:
        raise SystemExit(
            "Edward worst-case prompt exceeds the model budget: "
            f"{result['input_tokens']} >= {args.max_input_tokens}"
        )


if __name__ == "__main__":
    main()
