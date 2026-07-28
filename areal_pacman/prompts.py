"""Pacman image and live-state prompt construction."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from typing import Any, Mapping, Sequence

import numpy as np


MINIMAL_V1_SYSTEM_PROMPT = (
    "You control Pacman in the supplied game image. Collect pellets and avoid "
    "wasting moves. Respond with exactly one action token: U, D, L, R, or S. "
    "Do not explain the action."
)
MINIMAL_V1_USER_INSTRUCTION = (
    "Choose the next action. Return exactly U, D, L, R, or S."
)

WALL_AVOIDANCE_V1_SYSTEM_PROMPT = (
    "You control Pacman from one screenshot. Pacman is the yellow circle and "
    "blue lines are walls. Choose one movement direction that does not hit a "
    "wall. Respond with exactly one token: U, D, L, or R. Do not explain."
)
WALL_AVOIDANCE_V1_USER_INSTRUCTION = (
    "Choose one open movement direction. Return exactly U, D, L, or R."
)

WALL_AVOIDANCE_LOCAL_V2_SYSTEM_PROMPT = (
    "You control Pacman from a local screenshot centered on Pacman. Pacman is "
    "the large yellow circle near the center and cyan lines are walls. Choose "
    "one movement direction with an open corridor immediately next to Pacman. "
    "Directions are screen-absolute: U=up, D=down, L=left, R=right. Respond "
    "with exactly one token: U, D, L, or R. Do not explain."
)
WALL_AVOIDANCE_LOCAL_V2_USER_INSTRUCTION = (
    "Inspect the walls immediately around the centered Pacman. Choose one open "
    "direction and return exactly U, D, L, or R."
)

WALL_AVOIDANCE_AXIS_V3_SYSTEM_PROMPT = (
    "You control Pacman from a local screenshot centered on Pacman. Pacman is "
    "the large yellow circle near the center and cyan lines are walls. Read the "
    "corridor through Pacman's center: if it continues horizontally, choose L "
    "or R; if it continues vertically, choose U or D. Never choose a direction "
    "that crosses a cyan wall. Directions are screen-absolute: U=up, D=down, "
    "L=left, R=right. Respond with exactly one token: U, D, L, or R. Do not "
    "explain."
)
WALL_AVOIDANCE_AXIS_V3_USER_INSTRUCTION = (
    "Use the visible corridor axis through centered Pacman to choose an open "
    "direction. Return exactly U, D, L, or R."
)

LIVE_STATIC_V2_SYSTEM_PROMPT = (
    "You are playing Classic Pacman from one current screenshot. Pacman is the "
    "yellow circle with a mouth. Blue lines are walls. Small gold dots are "
    "pellets to eat. Directions are screen-absolute: U=up, D=down, L=left, "
    "R=right. Choose a direction that is visually open and moves Pacman along "
    "a path toward a nearby pellet; never move through a blue wall. Re-evaluate "
    "from every new screenshot. Prefer U, D, L, or R; use S only if no movement "
    "direction appears safe. Respond with exactly one action token: U, D, L, R, "
    "or S. Do not explain the action."
)
LIVE_STATIC_V2_USER_INSTRUCTION = (
    "Inspect only this screenshot and choose the next action toward a reachable "
    "nearby pellet. Return exactly U, D, L, R, or S."
)

LIVE_STATE_V3_SYSTEM_PROMPT = (
    "You are playing Classic Pacman. Follow the current screenshot and "
    "authoritative game-state instructions. Output exactly one action letter: "
    "U, D, L, or R. Do not explain the action."
)
LIVE_STATE_V3_USER_INSTRUCTION = (
    "You are playing Classic Pacman. One current screenshot.\n"
    "Yellow circle with a mouth = Pac-Man. Blue lines = walls. Small gold "
    "coins = pellets to eat.\n"
    "Directions are screen-absolute: U=top, D=bottom, L=left, R=right."
)

PROMPT_STYLES = (
    "minimal_v1",
    "wall_avoidance_v1",
    "wall_avoidance_local_v2",
    "wall_avoidance_axis_v3",
    "live_static_v2",
    "live_state_v3",
)
SYSTEM_PROMPT = MINIMAL_V1_SYSTEM_PROMPT
USER_INSTRUCTION = MINIMAL_V1_USER_INSTRUCTION


def prompt_text(prompt_style: str) -> tuple[str, str]:
    if prompt_style == "minimal_v1":
        return MINIMAL_V1_SYSTEM_PROMPT, MINIMAL_V1_USER_INSTRUCTION
    if prompt_style == "wall_avoidance_v1":
        return WALL_AVOIDANCE_V1_SYSTEM_PROMPT, WALL_AVOIDANCE_V1_USER_INSTRUCTION
    if prompt_style == "wall_avoidance_local_v2":
        return (
            WALL_AVOIDANCE_LOCAL_V2_SYSTEM_PROMPT,
            WALL_AVOIDANCE_LOCAL_V2_USER_INSTRUCTION,
        )
    if prompt_style == "wall_avoidance_axis_v3":
        return (
            WALL_AVOIDANCE_AXIS_V3_SYSTEM_PROMPT,
            WALL_AVOIDANCE_AXIS_V3_USER_INSTRUCTION,
        )
    if prompt_style == "live_static_v2":
        return LIVE_STATIC_V2_SYSTEM_PROMPT, LIVE_STATIC_V2_USER_INSTRUCTION
    if prompt_style == "live_state_v3":
        return LIVE_STATE_V3_SYSTEM_PROMPT, LIVE_STATE_V3_USER_INSTRUCTION
    raise ValueError(
        f"unsupported image prompt style {prompt_style!r}; "
        f"expected one of {PROMPT_STYLES}"
    )


def _action_text(actions: Sequence[Any]) -> str:
    tokens = [str(action) for action in actions]
    return ", ".join(tokens) if tokens else "NONE"


def live_state_instruction(context: Mapping[str, Any]) -> str:
    """Build the bounded dynamic block used by the live-demo-style policy."""
    position = context.get("pacman_position")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, (str, bytes))
        or len(position) != 2
    ):
        raise ValueError("live-state prompt requires a two-item pacman_position")
    row, col = (int(position[0]), int(position[1]))
    facing = str(context.get("facing") or "S")
    pellets_remaining = int(context.get("pellets_remaining", -1))
    open_actions = list(context.get("open_actions") or [])
    blocked_actions = list(context.get("blocked_actions") or [])
    exits = list(context.get("current_cell_exit_history") or [])
    preferred = list(context.get("preferred_open_actions") or [])
    last_action = context.get("last_action")

    history_text = _action_text(exits)
    preferred_text = _action_text(preferred or open_actions)
    pellets_line = (
        f" - Remaining pellets: {pellets_remaining}\n"
        if pellets_remaining >= 0
        else ""
    )
    reverse_line = (
        f"Do NOT reverse the last move ({last_action}) unless it is "
        "the only remaining preferred OPEN dir.\n"
        if last_action
        else ""
    )
    return (
        f"{LIVE_STATE_V3_USER_INSTRUCTION}\n\n"
        "GAME STATE (authoritative from Pacman engine):\n"
        f" - Grid position: row={row}, col={col}\n"
        f" - Facing: {facing}\n"
        f"{pellets_line}"
        f" - BLOCKED dirs here: {_action_text(blocked_actions)}\n"
        f" - OPEN dirs here: {_action_text(open_actions)}\n"
        " - Do NOT choose a BLOCKED direction.\n"
        " - Prefer an OPEN direction with nearest gold coins (pellets).\n\n"
        f"At this cell ({row},{col}), directions already taken before: "
        f"{history_text}.\n"
        f"Prefer a DIFFERENT OPEN dir than cell history: [{preferred_text}].\n"
        "Avoid repeating a past exit from this same cell when another OPEN dir "
        "remains.\n"
        f"{reverse_line}"
        "Do NOT oscillate like L<->R or U<->D.\n\n"
        "TASK - answer in EXACTLY this format (one line):\n"
        f"Choose ONE ACTION from [{preferred_text}]. Prefer the direction that "
        "on the path to eat the nearest coin based on the image; never choose "
        "a BLOCKED dir.\n\n"
        "Only output one ACTION letter: <one of U/D/L/R> where U=up, D=down, "
        "L=left, R=right"
    )


def encode_png(image: np.ndarray) -> bytes:
    """Encode one MaaPacman RGB observation as deterministic PNG bytes."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image-only rollouts") from exc
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return buffer.getvalue()


def crop_pacman_local_view(
    image: np.ndarray,
    *,
    radius: int = 48,
    upscale: int = 3,
) -> np.ndarray:
    """Find Pacman from yellow pixels and return a centered local RGB view."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB uint8 array")
    if radius < 8 or upscale < 1:
        raise ValueError("invalid local-view dimensions")

    # Pacman and pellets are pure yellow. Pacman is the largest connected
    # yellow component; orange ghosts are excluded by the strict green cutoff.
    yellow = (
        (image[:, :, 0] > 240)
        & (image[:, :, 1] > 220)
        & (image[:, :, 2] < 40)
    )
    # Ignore the small yellow score/lives glyphs at the very bottom.
    yellow[max(0, image.shape[0] - 20) :, :] = False
    seen = np.zeros(yellow.shape, dtype=bool)
    largest: list[tuple[int, int]] = []
    height, width = yellow.shape
    for row, col in zip(*np.where(yellow), strict=True):
        row = int(row)
        col = int(col)
        if seen[row, col]:
            continue
        component: list[tuple[int, int]] = []
        stack = [(row, col)]
        seen[row, col] = True
        while stack:
            current_row, current_col = stack.pop()
            component.append((current_row, current_col))
            for row_delta in (-1, 0, 1):
                for col_delta in (-1, 0, 1):
                    neighbor_row = current_row + row_delta
                    neighbor_col = current_col + col_delta
                    if (
                        0 <= neighbor_row < height
                        and 0 <= neighbor_col < width
                        and yellow[neighbor_row, neighbor_col]
                        and not seen[neighbor_row, neighbor_col]
                    ):
                        seen[neighbor_row, neighbor_col] = True
                        stack.append((neighbor_row, neighbor_col))
        if len(component) > len(largest):
            largest = component
    if len(largest) < 100:
        raise ValueError("could not locate Pacman in RGB observation")

    center_row = int(round(sum(row for row, _ in largest) / len(largest)))
    center_col = int(round(sum(col for _, col in largest) / len(largest)))
    side = radius * 2
    local = np.zeros((side, side, 3), dtype=np.uint8)
    source_top = max(0, center_row - radius)
    source_bottom = min(height, center_row + radius)
    source_left = max(0, center_col - radius)
    source_right = min(width, center_col + radius)
    target_top = source_top - (center_row - radius)
    target_left = source_left - (center_col - radius)
    local[
        target_top : target_top + source_bottom - source_top,
        target_left : target_left + source_right - source_left,
    ] = image[source_top:source_bottom, source_left:source_right]
    if upscale > 1:
        local = np.repeat(np.repeat(local, upscale, axis=0), upscale, axis=1)
    return local


def png_sha256(png: bytes) -> str:
    return hashlib.sha256(png).hexdigest()


def png_data_url(png: bytes) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_image_messages(
    png: bytes,
    *,
    prompt_style: str = "minimal_v1",
    state_context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the two-message contract containing exactly one image."""
    system_prompt, user_instruction = prompt_text(prompt_style)
    if prompt_style == "live_state_v3":
        if state_context is None:
            raise ValueError("live_state_v3 requires state_context")
        user_instruction = live_state_instruction(state_context)
    elif state_context is not None:
        raise ValueError(f"{prompt_style} does not accept state_context")
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": png_data_url(png)},
                },
            ],
        },
    ]


def image_count(messages: list[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            count += sum(
                1
                for item in content
                if isinstance(item, dict) and item.get("type") == "image_url"
            )
    return count
