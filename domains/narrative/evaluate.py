# domains/narrative/evaluate.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict, deque
import math

from .genotype import NarrativeGraph

def evaluate(g: NarrativeGraph, w: int = 20, h: int = 20) -> Tuple[float, Tuple[int, int], Dict[str, Any]]:
    """
    Returns (fitness, (x_bin, y_bin), meta)
    x: branching complexity
    y: tension variability
    """
    meta: Dict[str, Any] = {}
    adjacency = defaultdict(list)
    outdeg = defaultdict(int)
    indeg = defaultdict(int)

    for e in g.edges:
        adjacency[e.src].append(e.dst)
        outdeg[e.src] += 1
        indeg[e.dst] += 1

    # Reachability and solvability
    reachable = _reachable_nodes(adjacency, g.start_id)
    solvable, shortest_path = _has_path(adjacency, g.start_id, g.end_id)

    meta["n_scenes"] = len(g.scenes)
    meta["n_edges"] = len(g.edges)
    meta["reachable"] = len(reachable)
    meta["solvable"] = solvable
    meta["path_len"] = len(shortest_path) if shortest_path else None

    # Quality components
    if not solvable:
        fitness = -50.0
        desc = (0, 0)
        meta["reason"] = "unsolvable"
        return fitness, desc, meta

    # Dead ends (excluding end node)
    dead_ends = sum(1 for sid in g.scenes if sid != g.end_id and outdeg[sid] == 0)
    # Unreachable scenes penalty
    unreachable = len(g.scenes) - len(reachable)
    # Excessive loops penalty: count back edges in shortest path graph lightly
    loop_edges = sum(1 for e in g.edges if e.dst == g.start_id)

    # Coherence proxy: conflict consistency along shortest path
    conflict_changes = 0
    if shortest_path and len(shortest_path) >= 2:
        last = g.scenes[shortest_path[0]].conflict
        for sid in shortest_path[1:]:
            c = g.scenes[sid].conflict
            if c != last:
                conflict_changes += 1
            last = c

    # Fitness: encourage moderate length, low dead ends, low unreachable, moderate conflict changes
    path_len = len(shortest_path)
    length_score = -abs(path_len - 7) * 1.5  # best around 7 scenes
    fitness = 100.0 + length_score
    fitness -= dead_ends * 8.0
    fitness -= unreachable * 6.0
    fitness -= loop_edges * 5.0
    fitness -= max(0, conflict_changes - 3) * 2.0  # too many changes hurts

    meta["dead_ends"] = dead_ends
    meta["unreachable"] = unreachable
    meta["conflict_changes"] = conflict_changes
    meta["fitness"] = fitness

    # Descriptors
    branching = _avg_branching(outdeg, reachable)
    tension = _tension_variability(g, shortest_path)

    x_bin = _to_bin(branching, 0.0, 3.0, w)      # 0..3 average branching
    y_bin = _to_bin(tension, 0.0, 1.0, h)        # 0..1 variability

    meta["branching"] = branching
    meta["tension"] = tension

    return fitness, (x_bin, y_bin), meta

def _reachable_nodes(adj: Dict[int, List[int]], start: int) -> Set[int]:
    q = deque([start])
    seen = {start}
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen

def _has_path(adj: Dict[int, List[int]], start: int, goal: int) -> Tuple[bool, List[int]]:
    q = deque([start])
    parent = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v in adj.get(u, []):
            if v not in parent:
                parent[v] = u
                q.append(v)
    if goal not in parent:
        return False, []
    # Reconstruct shortest path
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return True, path

def _avg_branching(outdeg: Dict[int, int], reachable: Set[int]) -> float:
    if not reachable:
        return 0.0
    vals = [outdeg[sid] for sid in reachable]
    return sum(vals) / max(1, len(vals))

def _tension_variability(g: NarrativeGraph, path: List[int]) -> float:
    """
    Simple proxy: map scene types to tension levels and compute normalized variance.
    """
    if not path:
        return 0.0
    level = {
        "intro": 0.1,
        "clue": 0.3,
        "conflict": 0.7,
        "twist": 0.9,
        "reveal": 0.8,
        "resolution": 0.2,
    }
    xs = [level.get(g.scenes[sid].stype, 0.5) for sid in path]
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    # Normalize: variance max for values in [0,1] is 0.25
    return max(0.0, min(1.0, var / 0.25))

def _to_bin(x: float, lo: float, hi: float, n: int) -> int:
    if hi <= lo:
        return 0
    t = (x - lo) / (hi - lo)
    t = max(0.0, min(0.999999, t))
    return int(t * n)
