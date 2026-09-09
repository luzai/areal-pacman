"""Run an unmodified pacman-python game behind a line-oriented IPC bridge.

This module belongs to MaaPacman.  It instruments pygame at runtime and never
writes to the sibling pacman-python checkout.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import queue
import runpy
import sys
import threading
import zlib
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    return parser.parse_args()


class _PygameBridge:
    def __init__(self, pygame: Any, protocol_output: Any) -> None:
        self._pygame = pygame
        self._protocol_output = protocol_output
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._original_event_get = pygame.event.get
        self._original_flip = pygame.display.flip
        self._next_action: str | None = None
        self._pending_request: dict[str, Any] | None = None
        self._boot_started = False
        self._ready_sent = False

    def start(self) -> None:
        self._pygame.event.get = self._event_get
        self._pygame.display.flip = self._flip
        threading.Thread(target=self._read_commands, daemon=True).start()

    def _read_commands(self) -> None:
        for line in sys.stdin:
            try:
                command = json.loads(line)
            except json.JSONDecodeError as exc:
                self._emit({"type": "error", "error": f"invalid command: {exc}"})
                continue
            self._commands.put(command)
        self._commands.put({"op": "close"})

    def _event_get(self, *args: Any, **kwargs: Any) -> list[Any]:
        events = list(self._original_event_get(*args, **kwargs))
        action = self._next_action
        self._next_action = None
        key_by_action = {
            "U": self._pygame.K_UP,
            "D": self._pygame.K_DOWN,
            "L": self._pygame.K_LEFT,
            "R": self._pygame.K_RIGHT,
        }
        if action in key_by_action:
            events.append(
                self._pygame.event.Event(
                    self._pygame.KEYDOWN,
                    key=key_by_action[action],
                    repeat=False,
                )
            )
        return events

    def _flip(self) -> None:
        self._original_flip()
        globals_dict = sys._getframe(1).f_globals
        game = globals_dict.get("thisGame")
        player = globals_dict.get("player")
        level = globals_dict.get("thisLevel")
        if game is None or player is None or level is None:
            return

        if not self._boot_started:
            game.StartNewGame()
            game.SetMode(1)
            player.SnapToGrid()
            self._boot_started = True
            return

        payload = self._capture(globals_dict)
        state_writer = globals_dict.get("WriteAgentState")
        if callable(state_writer):
            # The original loop normally writes immediately after flip().
            # Because the bridge pauses inside flip(), invoke the unchanged
            # original writer here so the worker-local file matches this frame.
            state_writer()
        if not self._ready_sent:
            payload.update({"type": "ready", "request_id": 0})
            self._emit(payload)
            self._ready_sent = True
        elif self._pending_request is not None:
            self._pending_request["logic_frames"] += 1
            if self._step_is_complete(self._pending_request, payload["state"]):
                payload.update(
                    {
                        "type": "step",
                        "request_id": self._pending_request["request_id"],
                        "action": self._pending_request["action"],
                        "logic_frames": self._pending_request["logic_frames"],
                    }
                )
                self._emit(payload)
                self._pending_request = None
            else:
                # Let the original loop run another logic/render frame.  No
                # wall-clock sleep is used to decide when the step is done.
                return

        while True:
            command = self._commands.get()
            operation = command.get("op")
            if operation == "close":
                self._pygame.quit()
                raise SystemExit(0)
            if operation == "step" and command.get("action") in {
                "U",
                "D",
                "L",
                "R",
                "S",
            }:
                command["start_row"] = payload["state"]["row"]
                command["start_col"] = payload["state"]["col"]
                command["blocked_at_start"] = (
                    command["action"] in payload["state"]["blocked"]
                )
                command["logic_frames"] = 0
                self._pending_request = command
                self._next_action = command["action"]
                return
            self._emit(
                {
                    "type": "error",
                    "request_id": command.get("request_id"),
                    "error": f"unsupported command: {command!r}",
                }
            )

    def _step_is_complete(
        self, request: dict[str, Any], state: dict[str, Any]
    ) -> bool:
        if request["logic_frames"] < 1:
            return False
        if request["action"] == "S" or request["blocked_at_start"]:
            return True
        if int(state["mode"]) != 1:
            return True
        return (int(state["row"]), int(state["col"])) != (
            int(request["start_row"]),
            int(request["start_col"]),
        )

    def _capture(self, globals_dict: dict[str, Any]) -> dict[str, Any]:
        pygame = self._pygame
        game = globals_dict["thisGame"]
        player = globals_dict["player"]
        level = globals_dict["thisLevel"]
        tile_ids = globals_dict.get("tileID", {})
        surface = pygame.display.get_surface()
        frame = pygame.surfarray.array3d(surface).swapaxes(0, 1).copy()
        raw = frame.tobytes(order="C")

        pellet_id = tile_ids.get("pellet")
        power_id = tile_ids.get("pellet-power")
        normal_pellets = 0
        power_pellets = 0
        for row in range(int(level.lvlHeight)):
            for col in range(int(level.lvlWidth)):
                tile = level.GetMapTile(row, col)
                normal_pellets += int(tile == pellet_id)
                power_pellets += int(tile == power_id)

        blocked: list[str] = []
        open_actions: list[str] = []
        deltas = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
        row = int(player.nearestRow)
        col = int(player.nearestCol)
        for action, (row_delta, col_delta) in deltas.items():
            target_row = row + row_delta
            target_col = col + col_delta
            (blocked if level.IsWall(target_row, target_col) else open_actions).append(
                action
            )

        return {
            "frame": {
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "encoding": "zlib+base64",
                "data": base64.b64encode(zlib.compress(raw, level=1)).decode("ascii"),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            "state": {
                "row": row,
                "col": col,
                "facing": player.lastMoveDir if player.lastMoveDir in "UDLRS" else "S",
                "level": int(game.GetLevelNum()),
                "mode": int(game.mode),
                "score": int(game.score),
                "lives": int(game.lives),
                "width": int(level.lvlWidth),
                "height": int(level.lvlHeight),
                "normal_pellets": normal_pellets,
                "power_pellets": power_pellets,
                "collectibles_remaining": normal_pellets + power_pellets,
                "blocked": blocked,
                "open": open_actions,
            },
        }

    def _emit(self, payload: dict[str, Any]) -> None:
        self._protocol_output.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._protocol_output.flush()


def main() -> int:
    args = _parse_args()
    script = Path(args.script).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)

    protocol_output = sys.stdout
    sys.stdout = sys.stderr
    sys.path[0] = str(script.parent)
    sys.argv = [str(script), "--start-level", "1"]

    import pygame

    bridge = _PygameBridge(pygame, protocol_output)
    bridge.start()
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
