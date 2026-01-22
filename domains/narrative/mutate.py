# domains/narrative/mutate.py
from __future__ import annotations
from typing import List
import random

from .genotype import NarrativeGraph, Scene, Edge, SCENE_TYPES, CONFLICTS, LOCATIONS

EDGE_LABELS = ["continue", "investigate", "confront", "hide", "negotiate", "explore", "escape"]

def _new_scene_id(g: NarrativeGraph) -> int:
    return (max(g.scenes.keys()) + 1) if g.scenes else 0

def mutate_graph(rng: random.Random, g: NarrativeGraph) -> NarrativeGraph:
    """
    Pick one mutation. Always keep start and end scenes.
    """
    child = g.clone()
    ops = [
        _mut_add_scene,
        _mut_remove_scene,
        _mut_rewire_edge,
        _mut_add_edge,
        _mut_remove_edge,
        _mut_edit_scene,
    ]
    op = rng.choice(ops)
    op(rng, child)
    _cleanup(rng, child)
    return child

def _mut_add_scene(rng: random.Random, g: NarrativeGraph) -> None:
    sid = _new_scene_id(g)
    stype = rng.choice([t for t in SCENE_TYPES if t not in ["intro", "resolution"]])
    g.scenes[sid] = Scene(sid, stype, rng.choice(LOCATIONS), rng.choice(CONFLICTS), "A new event.")
    # Insert into a random edge: src -> dst becomes src -> new -> dst
    if not g.edges:
        g.edges.append(Edge(g.start_id, sid, rng.choice(EDGE_LABELS)))
        g.edges.append(Edge(sid, g.end_id, rng.choice(EDGE_LABELS)))
        return
    e = rng.choice(g.edges)
    g.edges.remove(e)
    g.edges.append(Edge(e.src, sid, e.label))
    g.edges.append(Edge(sid, e.dst, rng.choice(EDGE_LABELS)))

def _mut_remove_scene(rng: random.Random, g: NarrativeGraph) -> None:
    candidates = [sid for sid in g.scenes.keys() if sid not in [g.start_id, g.end_id]]
    if not candidates:
        return
    sid = rng.choice(candidates)
    # Remove scene and reconnect incoming to outgoing via simple shortcut edges
    incoming = [e for e in g.edges if e.dst == sid]
    outgoing = [e for e in g.edges if e.src == sid]
    g.edges = [e for e in g.edges if e.src != sid and e.dst != sid]
    del g.scenes[sid]
    for inc in incoming:
        for out in outgoing:
            if inc.src != out.dst:
                g.edges.append(Edge(inc.src, out.dst, rng.choice(EDGE_LABELS)))

def _mut_rewire_edge(rng: random.Random, g: NarrativeGraph) -> None:
    if not g.edges:
        return
    e = rng.choice(g.edges)
    nodes = list(g.scenes.keys())
    src = rng.choice(nodes)
    dst = rng.choice(nodes)
    if src == dst:
        return
    e.src = src
    e.dst = dst
    if rng.random() < 0.3:
        e.label = rng.choice(EDGE_LABELS)

def _mut_add_edge(rng: random.Random, g: NarrativeGraph) -> None:
    nodes = list(g.scenes.keys())
    if len(nodes) < 2:
        return
    src = rng.choice(nodes)
    dst = rng.choice(nodes)
    if src == dst:
        return
    g.edges.append(Edge(src, dst, rng.choice(EDGE_LABELS)))

def _mut_remove_edge(rng: random.Random, g: NarrativeGraph) -> None:
    if not g.edges:
        return
    g.edges.pop(rng.randrange(len(g.edges)))

def _mut_edit_scene(rng: random.Random, g: NarrativeGraph) -> None:
    sid = rng.choice(list(g.scenes.keys()))
    sc = g.scenes[sid]
    if sid == g.start_id:
        sc.stype = "intro"
    elif sid == g.end_id:
        sc.stype = "resolution"
    else:
        if rng.random() < 0.5:
            sc.stype = rng.choice([t for t in SCENE_TYPES if t not in ["intro", "resolution"]])
        if rng.random() < 0.5:
            sc.conflict = rng.choice(CONFLICTS)
        if rng.random() < 0.5:
            sc.location = rng.choice(LOCATIONS)
    if rng.random() < 0.6:
        sc.text_seed = rng.choice(["A clue appears.", "A door opens.", "A secret is revealed.", "A threat emerges."])

def _cleanup(rng: random.Random, g: NarrativeGraph) -> None:
    # Remove self-loops duplicates lightly
    new_edges = []
    seen = set()
    for e in g.edges:
        if e.src == e.dst:
            continue
        key = (e.src, e.dst, e.label)
        if key in seen:
            continue
        if e.src not in g.scenes or e.dst not in g.scenes:
            continue
        seen.add(key)
        new_edges.append(e)
    g.edges = new_edges
