"""
QD + Self-adaptive ES for simple 2D grid-level generation (dungeon-style).

This version is tuned to INCREASE COVERAGE.

Key changes vs your previous code:
1) Behavior descriptors use:
      x = wall density bin
      y = path length bin   (instead of noisy success-rate)
   Path length varies more smoothly => many more reachable bins.

2) Mutation includes a "big jump" operator:
      - 20% chance: block mutation (fills/clears a square block)
      - 80% chance: per-tile flips with self-adaptive sigma
   This helps move across density + structure quickly.

3) Injection schedule:
      - 20% random injection for the first 20k iterations
      - 5% after that
   Prevents early stagnation and improves exploration.

4) Optional: coverage-biased parent selection to push underfilled bins.

Outputs:
    outputs/archive_summary.json
    outputs/samples/*.txt
    outputs/png_top/*.png
    outputs/png_random/*.png
    outputs/coverage_filled.png
    outputs/coverage_fitness.png
    outputs/progress.png

Requires:
    pip install numpy matplotlib
"""

import os
import json
import math
import random
from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt


# =============================
# Tiles
# =============================
WALL = "#"
FLOOR = "."
START = "S"
GOAL = "G"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def normalize(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp((x - lo) / (hi - lo), 0.0, 1.0)


# =============================
# Level representation
# =============================
class Level:
    def __init__(self, h, w, grid=None):
        self.h = h
        self.w = w
        self.grid = grid if grid is not None else [[FLOOR for _ in range(w)] for _ in range(h)]

    @staticmethod
    def random(h, w, wall_p=0.25):
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

    def copy(self):
        return Level(self.h, self.w, [row[:] for row in self.grid])

    def to_ascii(self):
        return "\n".join("".join(row) for row in self.grid)

    def find_tile(self, t):
        for r in range(self.h):
            for c in range(self.w):
                if self.grid[r][c] == t:
                    return (r, c)
        raise ValueError(f"Tile {t} not found")

    def is_walkable(self, r, c):
        return self.grid[r][c] != WALL

    def neighbors4(self, r, c):
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.h and 0 <= cc < self.w and self.is_walkable(rr, cc):
                yield rr, cc

    # density on interior only (borders always walls)
    def wall_density(self):
        interior = (self.h - 2) * (self.w - 2)
        walls = 0
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if self.grid[r][c] == WALL:
                    walls += 1
        return walls / max(1, interior)


# =============================
# BFS + Noisy agent
# =============================
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


def noisy_agent_rollouts(level: Level, rollouts=12, max_steps=200, follow_prob=0.85):
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


# =============================
# Evaluation (fitness)
# =============================
PATH_MIN, PATH_MAX = 10, 120
DENS_TARGET = 0.28
DENS_PEN_MAX = 0.72

W_SR, W_PATH, W_VF, W_DENS = 2.0, 1.0, 0.8, 1.0


@dataclass
class EvalResult:
    solvable: bool
    path_len: Optional[int]
    success_rate: float
    visited_frac: float
    wall_density: float
    fitness: float


def evaluate(level: Level) -> EvalResult:
    d, _ = bfs_shortest_path(level)
    dens = level.wall_density()

    if d is None:
        return EvalResult(False, None, 0.0, 0.0, dens, -1e9)

    sr, _avg_steps, vf = noisy_agent_rollouts(level)

    # normalized parts
    path_n = normalize(d if sr > 0 else 0, PATH_MIN, PATH_MAX)
    dens_pen = normalize(abs(dens - DENS_TARGET), 0.0, DENS_PEN_MAX)

    fitness = W_SR * sr + W_PATH * path_n + W_VF * vf - W_DENS * dens_pen
    return EvalResult(True, d, sr, vf, dens, float(fitness))


# =============================
# Evolution (self-adaptive sigma + block mutation)
# =============================
@dataclass
class Individual:
    level: Level
    sigma: float


def mutate_block(lvl: Level, block_size=4, p_wall=0.5):
    """Big jump: set a square block to either walls or floors."""
    r0 = random.randint(1, lvl.h - 1 - block_size)
    c0 = random.randint(1, lvl.w - 1 - block_size)
    make_wall = (random.random() < p_wall)

    protected = {(1, 1), (lvl.h - 2, lvl.w - 2)}
    for r in range(r0, r0 + block_size):
        for c in range(c0, c0 + block_size):
            if (r, c) in protected:
                continue
            lvl.grid[r][c] = WALL if make_wall else FLOOR


def mutate(ind: Individual, tau=0.35, sigma_min=0.002, sigma_max=0.35, block_prob=0.20) -> Individual:
    lvl = ind.level.copy()

    # log-normal self-adaptation
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

    # enforce S/G
    lvl.grid[1][1] = START
    lvl.grid[lvl.h - 2][lvl.w - 2] = GOAL

    # small repair around S/G (helps keep solvable instances reachable)
    for (rr, cc) in [(1, 2), (2, 1), (lvl.h - 2, lvl.w - 3), (lvl.h - 3, lvl.w - 2)]:
        if lvl.grid[rr][cc] == WALL:
            lvl.grid[rr][cc] = FLOOR

    return Individual(lvl, sigma)


# =============================
# MAP-Elites (descriptor = density x path-length)
# =============================
def behavior_bin(ev: EvalResult, dens_bins: int, path_bins: int):
    x = clamp(int(ev.wall_density * dens_bins), 0, dens_bins - 1)
    pl_n = normalize(ev.path_len if ev.path_len is not None else 0, PATH_MIN, PATH_MAX)
    y = clamp(int(pl_n * path_bins), 0, path_bins - 1)
    return x, y


def pick_parent_key(keys, archive, dens_bins, path_bins):
    """
    Coverage-biased selection:
    50% uniform random
    50% prefer parents whose x/y bins are currently rare.
    """
    if random.random() < 0.5:
        return random.choice(keys)

    occ_x = [0] * dens_bins
    occ_y = [0] * path_bins
    for (bx, by) in archive.keys():
        occ_x[bx] += 1
        occ_y[by] += 1

    best_k = None
    best_score = -1e9
    for k in keys:
        bx, by = k
        score = (1.0 / (1 + occ_x[bx])) + (1.0 / (1 + occ_y[by]))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k if best_k is not None else random.choice(keys)


def map_elites(
    iters=100_000,
    h=14, w=28,
    dens_bins=12, path_bins=10,     # 120 bins total
    wall_p=0.25,
    init_sigma=0.06,
    seed=42,
    log_every=5000,
):
    random.seed(seed)
    np.random.seed(seed)

    archive: Dict[Tuple[int, int], Tuple[float, Individual, EvalResult]] = {}
    history = {"iter": [], "archive_size": [], "best_fitness": [], "coverage": []}

    # seed archive
    for _ in range(500):
        lvl = Level.random(h, w, wall_p=wall_p)
        ev = evaluate(lvl)
        if ev.solvable:
            b = behavior_bin(ev, dens_bins, path_bins)
            ind = Individual(lvl, init_sigma)
            if b not in archive or ev.fitness > archive[b][0]:
                archive[b] = (ev.fitness, ind, ev)

    keys = list(archive.keys())

    for t in range(1, iters + 1):
        # injection schedule: higher early, lower later
        inj = 0.20 if t < 20000 else 0.05

        if keys and random.random() > inj:
            pk = pick_parent_key(keys, archive, dens_bins, path_bins)
            parent = archive[pk][1]
            child = mutate(parent)
        else:
            child = Individual(Level.random(h, w, wall_p=wall_p), init_sigma)

        ev = evaluate(child.level)
        if ev.solvable:
            b = behavior_bin(ev, dens_bins, path_bins)
            if b not in archive or ev.fitness > archive[b][0]:
                archive[b] = (ev.fitness, child, ev)
                keys = list(archive.keys())

        if t % 1000 == 0:
            best = max((v[0] for v in archive.values()), default=-1e9)
            cov = len(archive) / (dens_bins * path_bins)
            history["iter"].append(t)
            history["archive_size"].append(len(archive))
            history["best_fitness"].append(best)
            history["coverage"].append(cov)

        if t % log_every == 0:
            best = max((v[0] for v in archive.values()), default=-1e9)
            cov = len(archive) / (dens_bins * path_bins)
            print(f"[{t:>6}/{iters}] archive={len(archive):>4}  coverage={cov*100:5.1f}%  best_f={best:7.3f}")

    return archive, history


# =============================
# Outputs / plotting
# =============================
def save_archive_summary(archive, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for (bx, by), (fit, ind, ev) in archive.items():
        summary[f"{bx}_{by}"] = {
            "fitness": float(fit),
            "sigma": float(ind.sigma),
            "path_len": int(ev.path_len) if ev.path_len is not None else None,
            "success_rate": float(ev.success_rate),
            "visited_frac": float(ev.visited_frac),
            "wall_density": float(ev.wall_density),
        }
    with open(os.path.join(out_dir, "archive_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def save_top_elites_as_ascii(archive, out_dir="outputs", top_k=30):
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    items = list(archive.items())
    items.sort(key=lambda kv: kv[1][0], reverse=True)

    for i, ((bx, by), (fit, ind, ev)) in enumerate(items[: min(top_k, len(items))]):
        fname = f"elite_{i:03d}_bin_{bx}_{by}_fit_{fit:.3f}_pl_{ev.path_len}.txt"
        with open(os.path.join(sample_dir, fname), "w") as f:
            f.write(ind.level.to_ascii())


def render_level_png(level: Level, out_path: str, tile_px: int = 18):
    palette = {
        WALL: (0, 0, 0),
        FLOOR: (255, 255, 255),
        START: (0, 200, 0),
        GOAL: (220, 0, 0),
    }
    img = np.zeros((level.h, level.w, 3), dtype=np.uint8)
    for r in range(level.h):
        for c in range(level.w):
            img[r, c] = palette.get(level.grid[r][c], (128, 128, 128))

    img = np.repeat(np.repeat(img, tile_px, axis=0), tile_px, axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_some_elites_as_png(archive, out_dir="outputs/png_top", k=10, mode="top", seed=0, tile_px=18):
    os.makedirs(out_dir, exist_ok=True)
    items = list(archive.items())
    if not items:
        return

    if mode == "top":
        items.sort(key=lambda kv: kv[1][0], reverse=True)
        chosen = items[: min(k, len(items))]
    else:
        random.seed(seed)
        chosen = random.sample(items, k=min(k, len(items)))

    for i, ((bx, by), (fit, ind, ev)) in enumerate(chosen):
        fname = f"{mode}_{i:02d}_bin_{bx}_{by}_fit_{fit:.3f}_pl_{ev.path_len}.png"
        render_level_png(ind.level, os.path.join(out_dir, fname), tile_px=tile_px)


def plot_coverage_filled(archive, dens_bins, path_bins, out_path="outputs/coverage_filled.png"):
    grid = np.zeros((path_bins, dens_bins), dtype=float)
    for (bx, by) in archive.keys():
        grid[by, bx] = 1.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.imshow(grid, aspect="auto", origin="lower")
    plt.colorbar(label="Filled (1) / Empty (0)")
    plt.xlabel("Wall density bin (x)")
    plt.ylabel("Path length bin (y)")
    plt.title("MAP-Elites: coverage (filled bins)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_coverage_fitness(archive, dens_bins, path_bins, out_path="outputs/coverage_fitness.png"):
    grid = np.full((path_bins, dens_bins), np.nan, dtype=float)
    for (bx, by), (fit, _ind, _ev) in archive.items():
        grid[by, bx] = fit if np.isnan(grid[by, bx]) else max(grid[by, bx], fit)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.imshow(grid, aspect="auto", origin="lower")
    plt.colorbar(label="Elite fitness")
    plt.xlabel("Wall density bin (x)")
    plt.ylabel("Path length bin (y)")
    plt.title("MAP-Elites: best fitness per bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_progress(hist, out_path="outputs/progress.png"):
    it = np.array(hist["iter"], dtype=float)
    if len(it) == 0:
        return
    archive_size = np.array(hist["archive_size"], dtype=float)
    best_f = np.array(hist["best_fitness"], dtype=float)
    coverage = np.array(hist["coverage"], dtype=float)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(it, archive_size, label="Archive size")
    plt.plot(it, best_f, label="Best fitness")
    plt.plot(it, coverage * 100.0, label="Coverage (%)")
    plt.xlabel("Iteration")
    plt.title("Progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =============================
# Main
# =============================
if __name__ == "__main__":
    DENS_BINS = 12
    PATH_BINS = 10

    archive, hist = map_elites(
        iters=100_000,
        h=14, w=28,
        dens_bins=DENS_BINS,
        path_bins=PATH_BINS,
        wall_p=0.25,
        init_sigma=0.06,
        seed=42,
        log_every=5000,
    )

    out_dir = "outputs"
    save_archive_summary(archive, out_dir=out_dir)
    save_top_elites_as_ascii(archive, out_dir=out_dir, top_k=30)

    save_some_elites_as_png(archive, out_dir=os.path.join(out_dir, "png_top"), k=10, mode="top", seed=42, tile_px=18)
    save_some_elites_as_png(archive, out_dir=os.path.join(out_dir, "png_random"), k=10, mode="random", seed=42, tile_px=18)

    plot_coverage_filled(archive, dens_bins=DENS_BINS, path_bins=PATH_BINS, out_path=os.path.join(out_dir, "coverage_filled.png"))
    plot_coverage_fitness(archive, dens_bins=DENS_BINS, path_bins=PATH_BINS, out_path=os.path.join(out_dir, "coverage_fitness.png"))
    plot_progress(hist, out_path=os.path.join(out_dir, "progress.png"))

    best_fit = max((v[0] for v in archive.values()), default=-1e9)
    print("\nDone.")
    print(f"Archive size: {len(archive)} / {DENS_BINS*PATH_BINS} bins")
    print(f"Best fitness: {best_fit:.3f}")
    print(f"Outputs saved to: ./{out_dir}/")
