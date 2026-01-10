# qd_dungeon/run.py

import os

from config import Config
from pcg.search import map_elites
from pcg.io_utils import save_archive_summary, save_top_elites_as_ascii
from pcg.viz import (
    save_some_elites_as_png,
    plot_coverage_filled,
    plot_coverage_fitness,
    plot_progress,
)

def main():
    cfg = Config()

    archive, hist = map_elites(cfg)

    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    save_archive_summary(archive, out_dir=out_dir)
    save_top_elites_as_ascii(archive, out_dir=out_dir, top_k=cfg.top_k_ascii)

    save_some_elites_as_png(archive, out_dir=os.path.join(out_dir, "png_top"),
                            k=cfg.top_k_png, mode="top", seed=cfg.seed, tile_px=cfg.tile_px)
    save_some_elites_as_png(archive, out_dir=os.path.join(out_dir, "png_random"),
                            k=cfg.top_k_png, mode="random", seed=cfg.seed, tile_px=cfg.tile_px)

    plot_coverage_filled(archive, cfg.dens_bins, cfg.path_bins, out_path=os.path.join(out_dir, "coverage_filled.png"))
    plot_coverage_fitness(archive, cfg.dens_bins, cfg.path_bins, out_path=os.path.join(out_dir, "coverage_fitness.png"))
    plot_progress(hist, out_path=os.path.join(out_dir, "progress.png"))

    best_fit = max((v[0] for v in archive.values()), default=-1e9)
    print("\nDone.")
    print(f"Archive size: {len(archive)} / {cfg.dens_bins * cfg.path_bins} bins")
    print(f"Best fitness: {best_fit:.3f}")
    print(f"Outputs saved to: ./{out_dir}/")

if __name__ == "__main__":
    main()
