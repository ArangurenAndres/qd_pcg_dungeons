import os
import pickle
import random

from core.archive import MapElites
from domains.narrative.genotype import make_initial_graph
from domains.narrative.mutate import mutate_graph
from domains.narrative.evaluate import evaluate

def run(
    seed: int = 0,
    iters: int = 60000,
    w: int = 20,
    h: int = 20,
    out_path: str | None = None,
):
    rng = random.Random(seed)
    archive = MapElites(w=w, h=h)

    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "narrative_archive.pkl")

    # Bootstrap
    for _ in range(400):
        g = make_initial_graph(rng, n_scenes=rng.randint(10, 18))
        fit, desc, meta = evaluate(g, w=w, h=h)
        archive.add(g, fit, desc, meta)
        archive.evals += 1

    if len(archive.elites()) == 0:
        raise RuntimeError("Archive is empty after bootstrap. Check evaluate() / initial graph generation.")

    for t in range(iters):
        parent = archive.sample_elite().item
        child = mutate_graph(rng, parent)
        fit, desc, meta = evaluate(child, w=w, h=h)
        archive.add(child, fit, desc, meta)
        archive.evals += 1

        if (t + 1) % 5000 == 0:
            print(
                f"[{t+1}/{iters}] "
                f"coverage={archive.coverage():.3f} qd={archive.qd_score():.1f} elites={len(archive.elites())}"
            )

    with open(out_path, "wb") as f:
        pickle.dump(archive, f)

    print("Saved:", out_path)

if __name__ == "__main__":
    run(seed=7, iters=60000, w=30, h=30)
