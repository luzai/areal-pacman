from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from areal_pacman.synthetic.env import OPPOSITE_ACTION, PacmanEnv, PacmanState
from areal_pacman.synthetic.maze_suite import (
    SPLIT_SIZES,
    SUITE_NAME,
    SUITE_PATH,
    layout_hash,
    topology_hash,
)


HEIGHT = 6
WIDTH = 9
PELLET_COUNT = 5
MAX_STEPS = 30


def connected(cells: set[tuple[int, int]]) -> bool:
    queue = deque([next(iter(cells))])
    reached = {queue[0]}
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + dr, col + dc)
            if nxt in cells and nxt not in reached:
                reached.add(nxt)
                queue.append(nxt)
    return reached == cells


def distances(cells: set[tuple[int, int]], start: tuple[int, int]) -> dict[tuple[int, int], int]:
    queue = deque([(start, 0)])
    result = {start: 0}
    while queue:
        (row, col), distance = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (row + dr, col + dc)
            if nxt in cells and nxt not in result:
                result[nxt] = distance + 1
                queue.append((nxt, distance + 1))
    return result


def generate_candidate(seed: int) -> tuple[str, ...]:
    rng = Random(seed)
    interior = {(row, col) for row in range(1, HEIGHT - 1) for col in range(1, WIDTH - 1)}
    open_cells = set(interior)
    candidates = list(interior)
    rng.shuffle(candidates)
    target_walls = rng.randint(3, 8)
    for cell in candidates:
        if len(interior - open_cells) >= target_walls:
            break
        proposed = open_cells - {cell}
        if len(proposed) >= PELLET_COUNT + 2 and connected(proposed):
            open_cells = proposed

    starts = list(open_cells)
    rng.shuffle(starts)
    pacman = starts[0]
    distance_map = distances(open_cells, pacman)
    ghost_candidates = [cell for cell, distance in distance_map.items() if distance >= 6]
    if not ghost_candidates:
        raise ValueError("candidate has no sufficiently distant ghost start")
    ghost = rng.choice(ghost_candidates)
    pellet_candidates = list(open_cells - {pacman, ghost})
    rng.shuffle(pellet_candidates)
    pellets = set(pellet_candidates[:PELLET_COUNT])

    rows = [["#" for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for row, col in open_cells:
        rows[row][col] = " "
    for row, col in pellets:
        rows[row][col] = "."
    rows[pacman[0]][pacman[1]] = "P"
    rows[ghost[0]][ghost[1]] = "G"
    return tuple("".join(row) for row in rows)


def oracle_choices(env: PacmanEnv, previous_action: str | None) -> list[str]:
    actions = env.legal_actions()
    if any(action != "stay" for action in actions):
        actions = [action for action in actions if action != "stay"]
    reverse = OPPOSITE_ACTION.get(previous_action or "")
    if reverse in actions and any(action != reverse for action in actions):
        actions = [action for action in actions if action != reverse]
    return actions or ["stay"]


def oracle_solution(layout: tuple[str, ...]) -> list[str] | None:
    env = PacmanEnv(layout=layout, max_steps=MAX_STEPS)
    start = (env.state.pacman, env.state.ghost, env.state.pellets, env.state.steps, None, tuple())
    queue = deque([start])
    visited = {(env.state.pacman, env.state.ghost, env.state.pellets, 0, None)}
    while queue:
        pacman, ghost, pellets, steps, previous_action, actions_so_far = queue.popleft()
        env.state = PacmanState(pacman, ghost, pellets, steps, 0, False, False)
        for action in oracle_choices(env, previous_action):
            env.state = PacmanState(pacman, ghost, pellets, steps, 0, False, False)
            state, _, done, info = env.step(action)
            next_actions = actions_so_far + (action,)
            if done and state.won:
                return list(next_actions)
            if done or info["reason"] != "running":
                continue
            key = (state.pacman, state.ghost, state.pellets, state.steps % 2, action)
            if key in visited:
                continue
            visited.add(key)
            queue.append((state.pacman, state.ghost, state.pellets, state.steps, action, next_actions))
    return None


def build_suite(generator_seed: int) -> dict[str, object]:
    records_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in SPLIT_SIZES}
    seen_hashes: set[str] = set()
    seen_topologies: set[str] = set()
    candidate_index = 0
    for split, target_size in SPLIT_SIZES.items():
        while len(records_by_split[split]) < target_size:
            seed = generator_seed + candidate_index
            candidate_index += 1
            try:
                layout = generate_candidate(seed)
            except ValueError:
                continue
            digest = layout_hash(layout)
            topology_digest = topology_hash(layout)
            if digest in seen_hashes or topology_digest in seen_topologies:
                continue
            solution = oracle_solution(layout)
            if solution is None:
                continue
            index = len(records_by_split[split])
            record = {
                "name": f"mmv1_{split}_{index:03d}",
                "split": split,
                "layout": list(layout),
                "layout_hash": digest,
                "topology_hash": topology_digest,
                "generator_seed": seed,
                "oracle_won": True,
                "oracle_steps": len(solution),
                "oracle_actions": solution,
            }
            records_by_split[split].append(record)
            seen_hashes.add(digest)
            seen_topologies.add(topology_digest)
    return {
        "name": SUITE_NAME,
        "generator_seed": generator_seed,
        "height": HEIGHT,
        "width": WIDTH,
        "pellet_count": PELLET_COUNT,
        "max_steps": MAX_STEPS,
        "candidate_count": candidate_index,
        "splits": records_by_split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic multi-maze PacMan suite.")
    parser.add_argument("--output", type=Path, default=SUITE_PATH)
    parser.add_argument("--generator-seed", type=int, default=7152026)
    args = parser.parse_args()
    payload = build_suite(args.generator_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    print(f"candidate_count={payload['candidate_count']}")
    for split, records in payload["splits"].items():
        steps = [record["oracle_steps"] for record in records]
        print(f"{split}={len(records)} oracle_steps={min(steps)}..{max(steps)}")


if __name__ == "__main__":
    main()
