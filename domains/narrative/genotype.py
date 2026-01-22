# domains/narrative/genotype.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random

SCENE_TYPES = ["intro", "clue", "conflict", "twist", "reveal", "resolution"]
CONFLICTS = ["pursuit", "betrayal", "mystery", "deadline", "rivalry", "survival"]
LOCATIONS = ["city", "forest", "lab", "spaceport", "hotel", "subway", "museum"]

@dataclass
class Scene:
    sid: int
    stype: str
    location: str
    conflict: str
    text_seed: str  # small string used by renderer

@dataclass
class Edge:
    src: int
    dst: int
    label: str

@dataclass
class NarrativeGraph:
    scenes: Dict[int, Scene] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    start_id: int = 0
    end_id: int = 1

    def clone(self) -> "NarrativeGraph":
        return NarrativeGraph(
            scenes={k: Scene(**vars(v)) for k, v in self.scenes.items()},
            edges=[Edge(**vars(e)) for e in self.edges],
            start_id=self.start_id,
            end_id=self.end_id,
        )

def make_initial_graph(rng: random.Random, n_scenes: int = 6) -> NarrativeGraph:
    """
    Start with a simple chain start -> ... -> end, plus a couple optional branches.
    """
    g = NarrativeGraph()
    g.start_id = 0
    g.end_id = 1

    # Start and end fixed
    g.scenes[0] = Scene(0, "intro", rng.choice(LOCATIONS), rng.choice(CONFLICTS), "You arrive.")
    g.scenes[1] = Scene(1, "resolution", rng.choice(LOCATIONS), rng.choice(CONFLICTS), "It ends.")

    next_id = 2
    mids = []
    for _ in range(max(0, n_scenes - 2)):
        sid = next_id
        next_id += 1
        stype = rng.choice([t for t in SCENE_TYPES if t not in ["intro", "resolution"]])
        g.scenes[sid] = Scene(sid, stype, rng.choice(LOCATIONS), rng.choice(CONFLICTS), "Something happens.")
        mids.append(sid)

    # Chain: start -> mids -> end
    chain = [g.start_id] + mids + [g.end_id]
    for a, b in zip(chain[:-1], chain[1:]):
        g.edges.append(Edge(a, b, label=rng.choice(["continue", "investigate", "confront", "hide"])))

    # Add 1 to 2 extra branches
    for _ in range(rng.randint(1, 2)):
        if len(chain) < 4:
            break
        src = rng.choice(chain[:-2])
        dst = rng.choice(chain[2:])
        if src != dst:
            g.edges.append(Edge(src, dst, label=rng.choice(["take shortcut", "take risk", "ask help"])))

    return g
