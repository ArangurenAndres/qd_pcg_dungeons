# qd_dungeon/config.py

from dataclasses import dataclass

@dataclass
class Config:
    # Level
    h: int = 14
    w: int = 28
    wall_p: float = 0.25

    # MAP-Elites bins (descriptor = density x path_length)
    dens_bins: int = 12
    path_bins: int = 10

    # Search
    iters: int = 100_000
    seed: int = 42
    init_sigma: float = 0.06
    log_every: int = 5000

    # Evaluation
    rollouts: int = 12
    max_steps: int = 200
    follow_prob: float = 0.85

    # Fitness normalization ranges
    path_min: int = 10
    path_max: int = 120
    dens_target: float = 0.28
    dens_pen_max: float = 0.72

    # Fitness weights
    w_sr: float = 2.0
    w_path: float = 1.0
    w_vf: float = 0.8
    w_dens: float = 1.0

    # Mutation
    tau: float = 0.35
    sigma_min: float = 0.002
    sigma_max: float = 0.35
    block_prob: float = 0.20

    # Injection schedule
    inj_early: float = 0.20
    inj_late: float = 0.05
    inj_switch_iter: int = 20_000

    # Output
    out_dir: str = "outputs"
    top_k_ascii: int = 30
    top_k_png: int = 10
    tile_px: int = 18
