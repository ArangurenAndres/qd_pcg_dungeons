# qd_dungeon/pcg/viz.py

import os
import numpy as np
import matplotlib.pyplot as plt

from .level import WALL, FLOOR, START, GOAL

def render_level_png(level, out_path: str, tile_px: int = 18):
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

def save_some_elites_as_png(archive, out_dir: str, k: int, mode: str, seed: int, tile_px: int):
    import random
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

def plot_coverage_filled(archive, dens_bins, path_bins, out_path: str):
    import numpy as np
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

def plot_coverage_fitness(archive, dens_bins, path_bins, out_path: str):
    import numpy as np
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

def plot_progress(hist, out_path: str):
    import numpy as np
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
