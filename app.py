import os
import time
import json
import math
import random
import threading
import pickle
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # server-safe backend
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_from_directory, render_template_string, abort

# ============================================================
# Core PCG / MAP-Elites code (Dungeon)
# ============================================================

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
        return "\n".join("".join(r) for r in self.grid)

    def find_tile(self, t):
        for r in range(self.h):
            for c in range(self.w):
                if self.grid[r][c] == t:
                    return (r, c)
        raise ValueError(f"Tile {t} not found")

    def is_walkable(self, r, c):
        return self.grid[r][c] != WALL

    def neighbors4(self, r, c):
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.h and 0 <= cc < self.w and self.is_walkable(rr, cc):
                yield rr, cc

    # IMPORTANT: density on INTERIOR ONLY
    def wall_density(self):
        interior = (self.h - 2) * (self.w - 2)
        walls = 0
        for r in range(1, self.h - 1):
            for c in range(1, self.w - 1):
                if self.grid[r][c] == WALL:
                    walls += 1
        return walls / max(1, interior)


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


def noisy_agent_rollouts(level: Level, rollouts=12, max_steps=200, follow_prob=0.85):
    d, parent = bfs_shortest_path(level)
    if d is None:
        return 0.0, float("inf"), 0.0

    s = level.find_tile(START)
    g = level.find_tile(GOAL)

    # reconstruct shortest path
    path = [g]
    while path[-1] != s:
        if path[-1] not in parent:
            return 0.0, float("inf"), 0.0
        path.append(parent[path[-1]])
    path.reverse()

    next_step = {path[i]: path[i + 1] for i in range(len(path) - 1)}

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


def evaluate(level: Level, rollouts=12, follow_prob=0.85) -> EvalResult:
    d, _ = bfs_shortest_path(level)
    dens = level.wall_density()
    if d is None:
        return EvalResult(False, None, 0.0, 0.0, dens, -1e9)

    sr, _avg_steps, vf = noisy_agent_rollouts(level, rollouts=rollouts, max_steps=200, follow_prob=follow_prob)
    path_n = normalize(d if sr > 0 else 0, PATH_MIN, PATH_MAX)
    dens_pen = normalize(abs(dens - DENS_TARGET), 0.0, DENS_PEN_MAX)

    fitness = W_SR * sr + W_PATH * path_n + W_VF * vf - W_DENS * dens_pen
    return EvalResult(True, d, sr, vf, dens, float(fitness))


@dataclass
class Individual:
    level: Level
    sigma: float


def mutate_block(lvl: Level, block_size=4, p_wall=0.5):
    """Big jump mutation: set a square block to either walls or floors."""
    r0 = random.randint(1, lvl.h - 1 - block_size)
    c0 = random.randint(1, lvl.w - 1 - block_size)
    make_wall = (random.random() < p_wall)

    protected = {(1, 1), (lvl.h - 2, lvl.w - 2)}
    for r in range(r0, r0 + block_size):
        for c in range(c0, c0 + block_size):
            if (r, c) in protected:
                continue
            lvl.grid[r][c] = WALL if make_wall else FLOOR


def mutate(ind: Individual, tau=0.35, sigma_min=0.002, sigma_max=0.35, block_prob=0.20):
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


def behavior_bin(ev: EvalResult, dens_bins: int, path_bins: int):
    """Descriptor: x=density bin, y=PATH LENGTH bin."""
    x = clamp(int(ev.wall_density * dens_bins), 0, dens_bins - 1)
    pl_n = normalize(ev.path_len if ev.path_len is not None else 0, PATH_MIN, PATH_MAX)
    y = clamp(int(pl_n * path_bins), 0, path_bins - 1)
    return (x, y)


def pick_parent_key(keys, archive, dens_bins, path_bins):
    """Coverage-biased parent selection to push search into underfilled regions."""
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


def save_some_elites_as_png(archive, out_dir, k=10, mode="top", seed=0, tile_px=18):
    os.makedirs(out_dir, exist_ok=True)
    items = list(archive.items())
    if not items:
        return []

    if mode == "top":
        items.sort(key=lambda kv: kv[1][0], reverse=True)
        chosen = items[: min(k, len(items))]
    else:
        random.seed(seed)
        chosen = random.sample(items, k=min(k, len(items)))

    paths = []
    for i, ((bx, by), (fit, ind, ev)) in enumerate(chosen):
        fname = f"{mode}_{i:02d}_bin_{bx}_{by}_fit_{fit:.3f}_pl_{ev.path_len}.png"
        path = os.path.join(out_dir, fname)
        render_level_png(ind.level, path, tile_px=tile_px)
        paths.append(fname)
    return paths


# ============================================================
# Narrative archive browsing (loads narrative_archive.pkl)
# - This does NOT run narrative QD in Flask.
# - It only loads a precomputed archive and renders it.
# ============================================================

NARRATIVE_ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), "narrative_archive.pkl")


def _load_narrative_archive():
    if not os.path.exists(NARRATIVE_ARCHIVE_PATH):
        raise FileNotFoundError(
            f"Missing {os.path.basename(NARRATIVE_ARCHIVE_PATH)}. "
            f"Generate it first (python run_narrative_qd.py) and keep it next to app.py."
        )
    with open(NARRATIVE_ARCHIVE_PATH, "rb") as f:
        return pickle.load(f)


def _narrative_shortest_path_text(g) -> str:
    """
    Fallback renderer that works even if you have not added domains/narrative/render.py yet.
    Expected graph object shape:
      g.scenes: dict[sid] with .stype, .location, .conflict, .text_seed
      g.edges: list with .src, .dst, .label
      g.start_id, g.end_id
    """
    adj = defaultdict(list)
    for e in getattr(g, "edges", []):
        adj[e.src].append((e.dst, getattr(e, "label", "")))

    start = getattr(g, "start_id", None)
    goal = getattr(g, "end_id", None)
    if start is None or goal is None:
        return "Invalid narrative graph: missing start_id/end_id."

    q = deque([start])
    parent = {start: None}
    parent_label = {}

    while q:
        u = q.popleft()
        if u == goal:
            break
        for v, lab in adj.get(u, []):
            if v not in parent:
                parent[v] = u
                parent_label[v] = lab
                q.append(v)

    if goal not in parent:
        return "Unsolvable story."

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    lines = []
    for i, sid in enumerate(path):
        sc = g.scenes[sid]
        prefix = "START" if sid == start else ("END" if sid == goal else f"S{i}")
        lines.append(f"{prefix}: [{sc.stype}] in {sc.location}, conflict={sc.conflict}. {sc.text_seed}")
        if i < len(path) - 1:
            nxt = path[i + 1]
            # edge label stored by child node
            lab = parent_label.get(nxt, "")
            if lab:
                lines.append(f"  Choice: {lab}")
    return "\n".join(lines)


NARRATIVE_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Narrative MAP-Elites</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; max-width: 1200px; }
    a { color: #111; }
    .muted { color:#666; font-size: 13px; }
    table { border-collapse: collapse; margin-top: 12px; }
    td { width: 18px; height: 18px; border: 1px solid #e5e7eb; padding: 0; }
    td a { display: block; width: 100%; height: 100%; text-decoration: none; }
    .topbar { display:flex; align-items:center; justify-content:space-between; gap: 12px; }
    .pill { font-size: 12px; padding: 6px 10px; border-radius: 999px; background: #f3f4f6; border: 1px solid #e5e7eb; }
  </style>
</head>
<body>
  <div class="topbar">
    <div>
      <h2 style="margin:0;">Narrative MAP-Elites</h2>
      <div class="muted">Browse a precomputed narrative archive (repertoire). Click a filled cell to view an elite story.</div>
    </div>
    <div class="pill">
      <a href="/">Back to dungeon demo</a>
    </div>
  </div>

  <div style="margin-top:10px;" class="pill">
    coverage={{ "%.1f"|format(coverage*100) }}% ,
    qd={{ "%.1f"|format(qd) }} ,
    evals={{ evals }}
    <span class="muted"> | x: branching (binned), y: tension variability (binned)</span>
  </div>

  <table>
    {% for y in range(h) %}
      <tr>
        {% for x in range(w) %}
          {% set e = grid[y][x] %}
          {% if e is none %}
            <td></td>
          {% else %}
            {% set t = 0.0 %}
            {% if fmax > fmin %}
              {% set t = (e.fitness - fmin) / (fmax - fmin) %}
            {% endif %}
            {% set shade = (255 - (t * 180))|int %}
            <td style="background: rgb({{ shade }}, {{ shade }}, 255);">
              <a href="/narrative/{{ x }}/{{ y }}" title="fit={{ '%.2f'|format(e.fitness) }}"></a>
            </td>
          {% endif %}
        {% endfor %}
      </tr>
    {% endfor %}
  </table>
</body>
</html>
"""

NARRATIVE_CELL_PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Elite {{x}},{{y}}</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; max-width: 1100px; }
    a { color:#111; }
    .muted { color:#666; font-size: 13px; }
    pre { background: #f6f6f6; padding: 12px; border-radius: 10px; border: 1px solid #e5e7eb; overflow-x: auto; }
    .pill { font-size: 12px; padding: 6px 10px; border-radius: 999px; background: #f3f4f6; border: 1px solid #e5e7eb; display:inline-block; }
    .row { display:flex; gap: 12px; flex-wrap: wrap; align-items:center; }
  </style>
</head>
<body>
  <div class="row">
    <div class="pill"><a href="/narrative">Back to archive</a></div>
    <div class="pill"><a href="/">Back to dungeon demo</a></div>
  </div>

  <h2 style="margin-top: 14px;">Elite at ({{x}}, {{y}})</h2>
  <div class="muted">fitness={{ "%.3f"|format(fitness) }}</div>

  <h3>Meta</h3>
  <pre>{{ meta }}</pre>

  <h3>Shortest path story</h3>
  <pre>{{ story }}</pre>
</body>
</html>
"""


# ============================================================
# Flask app: run dungeon job in background thread + poll progress
# ============================================================

app = Flask(__name__)

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

RUN_LOCK = threading.Lock()
CURRENT_RUN = {
    "running": False,
    "run_id": None,
    "params": None,
    "progress": {"t": 0, "iters": 0, "archive": 0, "coverage": 0.0, "best_f": -1e9},
    "done": False,
    "output_dir": None,
    "png_top": [],
    "png_random": [],
    "error": None,
}


def run_map_elites_job(params: dict, run_id: str):
    try:
        with RUN_LOCK:
            CURRENT_RUN["running"] = True
            CURRENT_RUN["done"] = False
            CURRENT_RUN["run_id"] = run_id
            CURRENT_RUN["params"] = params
            CURRENT_RUN["error"] = None
            CURRENT_RUN["png_top"] = []
            CURRENT_RUN["png_random"] = []

        out_dir = os.path.join(RUNS_DIR, run_id)
        os.makedirs(out_dir, exist_ok=True)

        seed = int(params["seed"])
        random.seed(seed)
        np.random.seed(seed)

        iters = int(params["iters"])
        h = int(params["h"])
        w = int(params["w"])
        dens_bins = int(params["dens_bins"])
        path_bins = int(params["path_bins"])
        wall_p = float(params["wall_p"])
        init_sigma = float(params["init_sigma"])
        rollouts = int(params["rollouts"])
        follow_prob = float(params["follow_prob"])
        tile_px = int(params["tile_px"])
        log_every = int(params["log_every"])
        block_prob = float(params["block_prob"])

        archive: Dict[Tuple[int, int], Tuple[float, Individual, EvalResult]] = {}

        # seed archive
        for _ in range(400):
            lvl = Level.random(h, w, wall_p=wall_p)
            ev = evaluate(lvl, rollouts=rollouts, follow_prob=follow_prob)
            if ev.solvable:
                b = behavior_bin(ev, dens_bins, path_bins)
                ind = Individual(lvl, init_sigma)
                if b not in archive or ev.fitness > archive[b][0]:
                    archive[b] = (ev.fitness, ind, ev)

        keys = list(archive.keys())

        # time-based UI pushing
        last_ui_push = time.time()

        for t in range(1, iters + 1):
            inj = 0.20 if t < 20000 else 0.05

            if keys and random.random() > inj:
                pk = pick_parent_key(keys, archive, dens_bins, path_bins)
                parent = archive[pk][1]
                child = mutate(parent, block_prob=block_prob)
            else:
                child = Individual(Level.random(h, w, wall_p=wall_p), init_sigma)

            ev = evaluate(child.level, rollouts=rollouts, follow_prob=follow_prob)
            if ev.solvable:
                b = behavior_bin(ev, dens_bins, path_bins)
                if b not in archive or ev.fitness > archive[b][0]:
                    archive[b] = (ev.fitness, child, ev)
                    keys = list(archive.keys())

            now = time.time()
            should_push = (t % log_every == 0) or (t == iters) or ((now - last_ui_push) > 0.5)
            if should_push:
                best = max((v[0] for v in archive.values()), default=-1e9)
                cov = len(archive) / max(1, dens_bins * path_bins)

                with RUN_LOCK:
                    if CURRENT_RUN["run_id"] != run_id:
                        return
                    CURRENT_RUN["progress"] = {
                        "t": int(t),
                        "iters": int(iters),
                        "archive": int(len(archive)),
                        "coverage": float(cov),
                        "best_f": float(best),
                    }

                last_ui_push = now

        # save summary json
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

        # PNG samples
        png_top_dir = os.path.join(out_dir, "png_top")
        png_rand_dir = os.path.join(out_dir, "png_random")
        png_top = save_some_elites_as_png(archive, png_top_dir, k=10, mode="top", seed=seed, tile_px=tile_px)
        png_rand = save_some_elites_as_png(archive, png_rand_dir, k=10, mode="random", seed=seed, tile_px=tile_px)

        with RUN_LOCK:
            if CURRENT_RUN["run_id"] != run_id:
                return
            CURRENT_RUN["done"] = True
            CURRENT_RUN["running"] = False
            CURRENT_RUN["output_dir"] = out_dir
            CURRENT_RUN["png_top"] = png_top
            CURRENT_RUN["png_random"] = png_rand

    except Exception as e:
        with RUN_LOCK:
            CURRENT_RUN["error"] = str(e)
            CURRENT_RUN["running"] = False
            CURRENT_RUN["done"] = True


# ============================================================
# Web UI (single-page)
# ============================================================

PAGE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>QD PCG Demo (MAP-Elites + ES)</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; max-width: 1100px; }
    h1 { margin-bottom: 6px; }
    .row { display: flex; gap: 18px; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 14px; flex: 1; min-width: 320px; }
    label { display:block; font-size: 13px; margin-top: 8px; color:#333; }
    input { width: 100%; padding: 8px; border-radius: 8px; border: 1px solid #ccc; }
    button { padding: 10px 14px; border-radius: 10px; border: 0; background: #111; color: white; cursor: pointer; }
    button:disabled { background:#888; cursor:not-allowed; }
    .bar { width:100%; height: 18px; background:#eee; border-radius: 999px; overflow:hidden; margin-top: 10px;}
    .fill { height:100%; width:0%; background:#111; transition: width 200ms linear; }
    .stat { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
    img { width: 100%; border-radius: 10px; border: 1px solid #ddd; }
    .muted { color:#666; font-size: 13px; }
    .error { color: #b00020; font-weight: 600; }
    .pill { font-size: 12px; padding: 6px 10px; border-radius: 999px; background: #f3f4f6; border: 1px solid #e5e7eb; display:inline-block; }
    a { color:#111; }
  </style>
</head>
<body>
  <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
    <div>
      <h1 style="margin:0;">MAP-Elites + Self-adaptive ES (Dungeon PCG)</h1>
      <div class="muted">Descriptor: wall-density × path-length. Live progress updates are pushed every ~0.5s.</div>
    </div>
    <div class="pill">
      <a href="/narrative">Open narrative archive</a>
    </div>
  </div>

  <div style="
    margin-top: 12px;
    margin-bottom: 18px;
    padding: 14px 16px;
    border-radius: 12px;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    font-size: 14px;
    line-height: 1.55;
  ">
    <strong>What this app does.</strong>
    This interactive demo runs a <em>Quality Diversity</em> search (MAP-Elites) with
    <em>self-adaptive Evolution Strategies</em> to generate a diverse archive of
    <strong>playable 2D dungeon levels</strong>. Instead of optimizing a single best map,
    the system discovers many distinct level styles across a behavior space
    (wall density × path length).

    <br><br>

    <strong>How to use it.</strong>
    Set the generation parameters on the left (grid size, iterations, mutation settings),
    click <em>Run</em>, and monitor live progress on the right. The archive grows over time as
    new behavioral niches are discovered.

    <br><br>

    <strong>What you get.</strong>
    Once the run finishes, the app renders:
    <ul style="margin:6px 0 0 18px;">
      <li><strong>Top elites</strong>: highest-fitness levels from different niches</li>
      <li><strong>Random elites</strong>: diverse samples across the archive</li>
    </ul>
    Each image represents a valid, solvable dungeon produced by the search.
  </div>

  <div class="row" style="margin-top: 14px;">
    <div class="card">
      <h3>Parameters</h3>

      <label>Iters</label><input id="iters" type="number" value="5000"/>

      <div class="row">
        <div style="flex:1;">
          <label>Grid height (h)</label><input id="h" type="number" value="14"/>
        </div>
        <div style="flex:1;">
          <label>Grid width (w)</label><input id="w" type="number" value="28"/>
        </div>
      </div>

      <div class="row">
        <div style="flex:1;">
          <label>Density bins (x)</label><input id="dens_bins" type="number" value="12"/>
        </div>
        <div style="flex:1;">
          <label>Path bins (y)</label><input id="path_bins" type="number" value="10"/>
        </div>
      </div>

      <div class="row">
        <div style="flex:1;">
          <label>Wall probability (init)</label><input id="wall_p" type="number" step="0.01" value="0.25"/>
        </div>
        <div style="flex:1;">
          <label>Init sigma</label><input id="init_sigma" type="number" step="0.01" value="0.06"/>
        </div>
      </div>

      <div class="row">
        <div style="flex:1;">
          <label>Block mutation prob</label><input id="block_prob" type="number" step="0.01" value="0.20"/>
        </div>
        <div style="flex:1;">
          <label>Seed</label><input id="seed" type="number" value="42"/>
        </div>
      </div>

      <div class="row">
        <div style="flex:1;">
          <label>Noisy rollouts</label><input id="rollouts" type="number" value="12"/>
        </div>
        <div style="flex:1;">
          <label>Follow prob</label><input id="follow_prob" type="number" step="0.01" value="0.85"/>
        </div>
      </div>

      <div class="row">
        <div style="flex:1;">
          <label>Progress update every (iters)</label><input id="log_every" type="number" value="200"/>
        </div>
        <div style="flex:1;">
          <label>PNG tile size (px)</label><input id="tile_px" type="number" value="18"/>
        </div>
      </div>

      <div style="margin-top: 12px;">
        <button id="runBtn" onclick="startRun()">Run</button>
      </div>
      <div id="msg" class="muted" style="margin-top:10px;"></div>
      <div id="err" class="error" style="margin-top:10px;"></div>
    </div>

    <div class="card">
      <h3>Live progress</h3>
      <div class="stat" id="stat">No run yet.</div>
      <div class="bar"><div class="fill" id="fill"></div></div>
      <div class="muted" style="margin-top:10px;">
        Coverage = archive_size / (dens_bins × path_bins)
      </div>
    </div>
  </div>

  <div class="row" style="margin-top: 18px;">
    <div class="card">
      <h3>Top samples</h3>
      <div id="topGrid" class="grid"></div>
    </div>
    <div class="card">
      <h3>Random samples</h3>
      <div id="randGrid" class="grid"></div>
    </div>
  </div>

<script>
let pollTimer = null;

function getParams() {
  const fields = ["iters","h","w","dens_bins","path_bins","wall_p","init_sigma","block_prob","seed","rollouts","follow_prob","log_every","tile_px"];
  const p = {};
  for (const f of fields) p[f] = document.getElementById(f).value;
  return p;
}

async function startRun() {
  document.getElementById("err").textContent = "";
  document.getElementById("msg").textContent = "Starting run...";
  document.getElementById("runBtn").disabled = true;

  const res = await fetch("/start", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(getParams())
  });

  const data = await res.json();
  if (!data.ok) {
    document.getElementById("err").textContent = data.error || "Failed to start.";
    document.getElementById("msg").textContent = "";
    document.getElementById("runBtn").disabled = false;
    return;
  }

  document.getElementById("msg").textContent = "Run started. Polling progress...";
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(pollProgress, 500);
}

async function pollProgress() {
  const res = await fetch("/status");
  const data = await res.json();

  if (data.error) {
    document.getElementById("err").textContent = data.error;
  }

  const p = data.progress;
  const pct = (p.iters > 0) ? (100.0 * p.t / p.iters) : 0.0;
  document.getElementById("fill").style.width = pct.toFixed(1) + "%";

  document.getElementById("stat").textContent =
    `iter=${p.t}/${p.iters} | archive=${p.archive} | coverage=${(p.coverage*100).toFixed(1)}% | best_f=${p.best_f.toFixed(3)}`;

  if (data.done) {
    clearInterval(pollTimer);
    pollTimer = null;
    document.getElementById("msg").textContent = "Done. Showing samples below.";
    document.getElementById("runBtn").disabled = false;
    renderImages("topGrid", data.run_id, "png_top", data.png_top);
    renderImages("randGrid", data.run_id, "png_random", data.png_random);
  } else if (!data.running) {
    document.getElementById("runBtn").disabled = false;
  }
}

function renderImages(containerId, runId, subdir, files) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  if (!files || files.length === 0) {
    el.innerHTML = "<div class='muted'>No images yet.</div>";
    return;
  }
  for (const f of files) {
    const img = document.createElement("img");
    img.src = `/runs/${runId}/${subdir}/${encodeURIComponent(f)}?t=${Date.now()}`;
    img.title = f;
    el.appendChild(img);
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/start", methods=["POST"])
def start():
    params = request.get_json(force=True)

    try:
        required = ["iters","h","w","dens_bins","path_bins","wall_p","init_sigma",
                    "block_prob","seed","rollouts","follow_prob","log_every","tile_px"]
        for k in required:
            if k not in params:
                return jsonify(ok=False, error=f"Missing param: {k}")

        with RUN_LOCK:
            if CURRENT_RUN["running"]:
                return jsonify(ok=False, error="A run is already in progress.")

        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_seed{params['seed']}"
        thread = threading.Thread(target=run_map_elites_job, args=(params, run_id), daemon=True)
        thread.start()
        return jsonify(ok=True, run_id=run_id)
    except Exception as e:
        return jsonify(ok=False, error=str(e))


@app.route("/status")
def status():
    with RUN_LOCK:
        payload = dict(CURRENT_RUN)
        return jsonify(
            running=payload["running"],
            done=payload["done"],
            run_id=payload["run_id"],
            progress=payload["progress"],
            png_top=payload["png_top"],
            png_random=payload["png_random"],
            error=payload["error"],
        )


@app.route("/runs/<run_id>/<path:filename>")
def runs_static(run_id, filename):
    directory = os.path.join(RUNS_DIR, run_id)
    return send_from_directory(directory, filename)


# ============================================================
# Narrative browsing routes
# ============================================================

@app.route("/narrative")
def narrative_home():
    try:
        arc = _load_narrative_archive()
    except Exception as e:
        # Render a very small error page with a direct instruction
        return render_template_string(
            "<h2>Narrative archive missing</h2>"
            "<p style='font-family: system-ui;'>"
            "I could not load <code>narrative_archive.pkl</code> next to <code>app.py</code>.<br>"
            "Generate it first, then refresh this page.<br><br>"
            f"Error: <pre>{str(e)}</pre>"
            "</p>"
            "<p><a href='/'>Back</a></p>"
        )

    grid = arc.grid
    fits = [e.fitness for row in grid for e in row if e is not None]
    fmin, fmax = (min(fits), max(fits)) if fits else (0.0, 1.0)

    return render_template_string(
        NARRATIVE_PAGE,
        w=arc.w,
        h=arc.h,
        grid=grid,
        fmin=fmin,
        fmax=fmax,
        coverage=arc.coverage(),
        qd=arc.qd_score(),
        evals=getattr(arc, "evals", 0),
    )


@app.route("/narrative/<int:x>/<int:y>")
def narrative_cell(x: int, y: int):
    arc = _load_narrative_archive()

    if not (0 <= x < arc.w and 0 <= y < arc.h):
        abort(404)

    elite = arc.grid[y][x]
    if elite is None:
        abort(404)

    g = elite.item
    story = _narrative_shortest_path_text(g)

    return render_template_string(
        NARRATIVE_CELL_PAGE,
        x=x,
        y=y,
        fitness=float(elite.fitness),
        meta=str(elite.meta),
        story=story,
    )


if __name__ == "__main__":
    # IMPORTANT: disable reloader, otherwise the background thread may run in a different process
    app.run(debug=True, use_reloader=False,port=5002)
