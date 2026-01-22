# core/archive.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, List
import math
import random

@dataclass
class Elite:
    item: Any
    fitness: float
    desc: Tuple[int, int]
    meta: Dict[str, Any]

class MapElites:
    """
    Simple 2D MAP-Elites.
    desc is a tuple (x_bin, y_bin) already discretized into [0..w-1], [0..h-1].
    """
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.grid: List[List[Optional[Elite]]] = [[None for _ in range(w)] for _ in range(h)]
        self.evals = 0

    def add(self, item: Any, fitness: float, desc: Tuple[int, int], meta: Dict[str, Any]) -> bool:
        x, y = desc
        if not (0 <= x < self.w and 0 <= y < self.h):
            return False

        current = self.grid[y][x]
        if current is None or fitness > current.fitness:
            self.grid[y][x] = Elite(item=item, fitness=fitness, desc=desc, meta=meta)
            return True
        return False

    def coverage(self) -> float:
        filled = sum(1 for row in self.grid for e in row if e is not None)
        return filled / (self.w * self.h)

    def qd_score(self) -> float:
        return sum(e.fitness for row in self.grid for e in row if e is not None)

    def elites(self) -> List[Elite]:
        return [e for row in self.grid for e in row if e is not None]

    def sample_elite(self) -> Elite:
        els = self.elites()
        if not els:
            raise RuntimeError("Archive is empty.")
        return random.choice(els)
