# ui/app.py
import pickle
from flask import Flask, render_template, abort

from domains.narrative.render import story_text, adjacency_text

app = Flask(__name__)
ARCHIVE_PATH = "narrative_archive.pkl"

def load_archive():
    with open(ARCHIVE_PATH, "rb") as f:
        return pickle.load(f)

@app.route("/narrative")
def narrative_home():
    arc = load_archive()
    grid = arc.grid

    # For coloring, find min/max fitness among filled cells
    fits = [e.fitness for row in grid for e in row if e is not None]
    if fits:
        fmin, fmax = min(fits), max(fits)
    else:
        fmin, fmax = 0.0, 1.0

    return render_template(
        "narrative.html",
        w=arc.w,
        h=arc.h,
        grid=grid,
        fmin=fmin,
        fmax=fmax,
        coverage=arc.coverage(),
        qd=arc.qd_score(),
        evals=arc.evals,
    )

@app.route("/narrative/<int:x>/<int:y>")
def narrative_cell(x: int, y: int):
    arc = load_archive()
    if not (0 <= x < arc.w and 0 <= y < arc.h):
        abort(404)
    elite = arc.grid[y][x]
    if elite is None:
        abort(404)

    g = elite.item
    return render_template(
        "narrative_cell.html",
        x=x,
        y=y,
        fitness=elite.fitness,
        meta=elite.meta,
        story=story_text(g),
        adjacency=adjacency_text(g),
    )

if __name__ == "__main__":
    app.run(debug=True)
