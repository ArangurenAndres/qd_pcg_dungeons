# qd_dungeon/pcg/io_utils.py

import os
import json

def save_archive_summary(archive, out_dir: str):
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

def save_top_elites_as_ascii(archive, out_dir: str, top_k: int):
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    items = list(archive.items())
    items.sort(key=lambda kv: kv[1][0], reverse=True)

    for i, ((bx, by), (fit, ind, ev)) in enumerate(items[: min(top_k, len(items))]):
        fname = f"elite_{i:03d}_bin_{bx}_{by}_fit_{fit:.3f}_pl_{ev.path_len}.txt"
        with open(os.path.join(sample_dir, fname), "w") as f:
            f.write(ind.level.to_ascii())
