# qd_dungeon/pcg/search.py

import random
from typing import Dict, Tuple

import numpy as np

from .level import Level
from .mutate import Individual, mutate
from .eval import evaluate, EvalResult, normalize

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def behavior_bin(ev: EvalResult, dens_bins: int, path_bins: int, path_min: int, path_max: int):
    x = clamp(int(ev.wall_density * dens_bins), 0, dens_bins - 1)
    pl_n = normalize(ev.path_len if ev.path_len is not None else 0, path_min, path_max)
    y = clamp(int(pl_n * path_bins), 0, path_bins - 1)
    return x, y

def pick_parent_key(keys, archive, dens_bins, path_bins):
    # 50% uniform, 50% biased toward rare x/y usage
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

def map_elites(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    archive: Dict[Tuple[int, int], Tuple[float, Individual, EvalResult]] = {}
    history = {"iter": [], "archive_size": [], "best_fitness": [], "coverage": []}

    # seed archive
    for _ in range(500):
        lvl = Level.random(cfg.h, cfg.w, wall_p=cfg.wall_p)
        ev = evaluate(
            lvl,
            rollouts=cfg.rollouts,
            max_steps=cfg.max_steps,
            follow_prob=cfg.follow_prob,
            path_min=cfg.path_min,
            path_max=cfg.path_max,
            dens_target=cfg.dens_target,
            dens_pen_max=cfg.dens_pen_max,
            w_sr=cfg.w_sr,
            w_path=cfg.w_path,
            w_vf=cfg.w_vf,
            w_dens=cfg.w_dens,
        )
        if ev.solvable:
            b = behavior_bin(ev, cfg.dens_bins, cfg.path_bins, cfg.path_min, cfg.path_max)
            ind = Individual(lvl, cfg.init_sigma)
            if b not in archive or ev.fitness > archive[b][0]:
                archive[b] = (ev.fitness, ind, ev)

    keys = list(archive.keys())

    for t in range(1, cfg.iters + 1):
        inj = cfg.inj_early if t < cfg.inj_switch_iter else cfg.inj_late

        if keys and random.random() > inj:
            pk = pick_parent_key(keys, archive, cfg.dens_bins, cfg.path_bins)
            parent = archive[pk][1]
            child = mutate(
                parent,
                tau=cfg.tau,
                sigma_min=cfg.sigma_min,
                sigma_max=cfg.sigma_max,
                block_prob=cfg.block_prob,
            )
        else:
            child = Individual(Level.random(cfg.h, cfg.w, wall_p=cfg.wall_p), cfg.init_sigma)

        ev = evaluate(
            child.level,
            rollouts=cfg.rollouts,
            max_steps=cfg.max_steps,
            follow_prob=cfg.follow_prob,
            path_min=cfg.path_min,
            path_max=cfg.path_max,
            dens_target=cfg.dens_target,
            dens_pen_max=cfg.dens_pen_max,
            w_sr=cfg.w_sr,
            w_path=cfg.w_path,
            w_vf=cfg.w_vf,
            w_dens=cfg.w_dens,
        )

        if ev.solvable:
            b = behavior_bin(ev, cfg.dens_bins, cfg.path_bins, cfg.path_min, cfg.path_max)
            if b not in archive or ev.fitness > archive[b][0]:
                archive[b] = (ev.fitness, child, ev)
                keys = list(archive.keys())

        if t % 1000 == 0:
            best = max((v[0] for v in archive.values()), default=-1e9)
            cov = len(archive) / (cfg.dens_bins * cfg.path_bins)
            history["iter"].append(t)
            history["archive_size"].append(len(archive))
            history["best_fitness"].append(best)
            history["coverage"].append(cov)

        if t % cfg.log_every == 0:
            best = max((v[0] for v in archive.values()), default=-1e9)
            cov = len(archive) / (cfg.dens_bins * cfg.path_bins)
            print(f"[{t:>6}/{cfg.iters}] archive={len(archive):>4}  coverage={cov*100:5.1f}%  best_f={best:7.3f}")

    return archive, history
