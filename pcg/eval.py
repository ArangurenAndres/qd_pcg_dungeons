# qd_dungeon/pcg/eval.py

import random
from dataclasses import dataclass
from collections import deque
from typing import Dict, Tuple, Optional, List

import numpy as np

from .level import Level, START, GOAL

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def normalize(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)

def bfs_shortest_path(level: Level) -> Tuple[Optional[int], Dict[Tuple[int, int], Tuple[int, int]]]:
    s = level.find_tile(START)
    g = level.find_tile(GOAL)

    q = deque([s])
    dist = {s: 0}
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    while q:
        cur = q.popleft()
        if cur == g:
            return dist[cur], parent
        for nxt in level.neighbors4(*cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                parent[nxt] = cur
                q.append(nxt)

    return None, parent

def reconstruct_shortest_path(level: Level) -> Optional[List[Tuple[int, int]]]:
    d, parent = bfs_shortest_path(level)
    if d is None:
        return None
    s = level.find_tile(START)
    g = level.find_tile(GOAL)

    path = [g]
    while path[-1] != s:
        if path[-1] not in parent:
            return None
        path.append(parent[path[-1]])
    path.reverse()
    return path

def noisy_agent_rollouts(level: Level, rollouts: int, max_steps: int, follow_prob: float):
    path = reconstruct_shortest_path(level)
    if not path or len(path) < 2:
        return 0.0, float("inf"), 0.0

    s = level.find_tile(START)
    g = level.find_tile(GOAL)
    next_step = {path[i]: path[i+1] for i in range(len(path)-1)}

    successes = 0
    steps_success = []
    visited_fracs = []

    walkable = sum(1 for r in range(level.h) for c in range(level.w) if level.is_walkable(r, c))
    walkable = max(1, walkable)

    for _ in range(rollouts):
        pos = s
        visited = {pos}
        for step in range(max_steps):
            if pos == g:
                successes += 1
                steps_success.append(step)
                break
            moves = list(level.neighbors4(*pos))
            if not moves:
                break

            if random.random() < follow_prob and pos in next_step:
                pref = next_step[pos]
                pos = pref if pref in moves else random.choice(moves)
            else:
                pos = random.choice(moves)

            visited.add(pos)

        visited_fracs.append(len(visited) / walkable)

    sr = successes / rollouts
    avg_steps = float(np.mean(steps_success)) if steps_success else float("inf")
    vf = float(np.mean(visited_fracs)) if visited_fracs else 0.0
    return sr, avg_steps, vf

@dataclass
class EvalResult:
    solvable: bool
    path_len: Optional[int]
    success_rate: float
    visited_frac: float
    wall_density: float
    fitness: float

def evaluate(
    level: Level,
    rollouts: int,
    max_steps: int,
    follow_prob: float,
    path_min: int,
    path_max: int,
    dens_target: float,
    dens_pen_max: float,
    w_sr: float,
    w_path: float,
    w_vf: float,
    w_dens: float,
) -> EvalResult:
    d, _ = bfs_shortest_path(level)
    dens = level.wall_density()

    if d is None:
        return EvalResult(False, None, 0.0, 0.0, dens, -1e9)

    sr, _avg_steps, vf = noisy_agent_rollouts(level, rollouts=rollouts, max_steps=max_steps, follow_prob=follow_prob)

    path_n = normalize(d if sr > 0 else 0, path_min, path_max)
    dens_pen = normalize(abs(dens - dens_target), 0.0, dens_pen_max)

    fitness = w_sr * sr + w_path * path_n + w_vf * vf - w_dens * dens_pen
    return EvalResult(True, d, sr, vf, dens, float(fitness))
