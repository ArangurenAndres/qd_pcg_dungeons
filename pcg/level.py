# qd_dungeon/pcg/level.py

import random
from typing import Optional, List, Tuple

WALL = "#"
FLOOR = "."
START = "S"
GOAL = "G"

class Level:
    def __init__(self, h: int, w: int, grid: Optional[List[List[str]]] = None):
        self.h = h
        self.w = w
        self.grid = grid if grid is not None else [[FLOOR for _ in range(w)] for _ in range(h)]

    @staticmethod
    def random(h: int, w: int, wall_p: float = 0.25) -> "Level":
        lvl = Level(h, w)
        for r in range(h):
            for c in range(w):
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    lvl.grid[r][c] = WALL
                else:
                    lvl.grid[r][c] = WALL if random.random() < wall_p else FLOOR

        lvl.grid[1][1] = START
        lvl.grid[h - 2][w - 2] = GOAL

        # small repair around S/G
        for (rr, cc) in [(1, 2), (2, 1), (h - 2, w - 3), (h - 3, w - 2)]:
            if lvl.grid[rr][cc] == WALL:
                lvl.grid[rr][cc] = FLOOR

        return lvl

    def copy(self) -> "Level":
        return Level(self.h, self.w, [row[:] for row in self.grid])

    def to_ascii(self) -> str:
        return "\n".join("".join(row) for row in self.grid)

    def find_tile(self, t: str) -> Tuple[int, int]:
        for r in range(self.h):
            for c in range(self.w):
                if self.grid[r][c] == t:
                    return (r, c)
        raise ValueError(f"Tile {t} not found")

    def is_walkable(self, r: int, c: int) -> bool:
        return self.grid[r][c] != WALL

    def neighbors4(self, r: int, c: int):
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.h and 0 <= cc < self.w and self.is_walkable(rr, cc):
                yield rr, cc

    def wall_density(self) -> float:
        # density on interior only (borders always walls)
        interior = (self.h - 2) * (self.w - 2)
        walls = 0
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if self.grid[r][c] == WALL:
                    walls += 1
        return walls / max(1, interior)
