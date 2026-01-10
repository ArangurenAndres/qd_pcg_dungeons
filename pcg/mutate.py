# qd_dungeon/pcg/mutate.py

import math
import random
from dataclasses import dataclass

from .level import Level, WALL, FLOOR, START, GOAL

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

@dataclass
class Individual:
    level: Level
    sigma: float

def mutate_block(lvl: Level, block_size: int, p_wall: float = 0.5):
    r0 = random.randint(1, lvl.h - 1 - block_size)
    c0 = random.randint(1, lvl.w - 1 - block_size)
    make_wall = (random.random() < p_wall)

    protected = {(1, 1), (lvl.h - 2, lvl.w - 2)}
    for r in range(r0, r0 + block_size):
        for c in range(c0, c0 + block_size):
            if (r, c) in protected:
                continue
            lvl.grid[r][c] = WALL if make_wall else FLOOR

def mutate(
    ind: Individual,
    tau: float,
    sigma_min: float,
    sigma_max: float,
    block_prob: float,
) -> Individual:
    lvl = ind.level.copy()

    sigma = clamp(ind.sigma * math.exp(tau * random.gauss(0, 1)), sigma_min, sigma_max)

    if random.random() < block_prob:
        mutate_block(lvl, block_size=random.choice([3, 4, 5]), p_wall=0.5)
    else:
        protected = {(1, 1), (lvl.h - 2, lvl.w - 2)}
        for r in range(1, lvl.h - 1):
            for c in range(1, lvl.w - 1):
                if (r, c) in protected:
                    continue
                if random.random() < sigma:
                    lvl.grid[r][c] = FLOOR if lvl.grid[r][c] == WALL else WALL

    lvl.grid[1][1] = START
    lvl.grid[lvl.h - 2][lvl.w - 2] = GOAL

    # small repair around S/G
    for (rr, cc) in [(1, 2), (2, 1), (lvl.h - 2, lvl.w - 3), (lvl.h - 3, lvl.w - 2)]:
        if lvl.grid[rr][cc] == WALL:
            lvl.grid[rr][cc] = FLOOR

    return Individual(lvl, sigma)
