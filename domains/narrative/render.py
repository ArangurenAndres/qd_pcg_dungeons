# domains/narrative/render.py
from __future__ import annotations
from typing import Dict, List
from collections import defaultdict, deque

from .genotype import NarrativeGraph

def story_text(g: NarrativeGraph) -> str:
    # Build adjacency for a simple BFS path
    adj = defaultdict(list)
    for e in g.edges:
        adj[e.src].append((e.dst, e.label))

    path = _shortest_path(adj, g.start_id, g.end_id)
    if not path:
        return "Unsolvable story."

    lines = []
    for i, sid in enumerate(path):
        sc = g.scenes[sid]
        prefix = "START" if sid == g.start_id else ("END" if sid == g.end_id else f"S{i}")
        lines.append(f"{prefix}: [{sc.stype}] in {sc.location}, conflict={sc.conflict}. {sc.text_seed}")
        if i < len(path) - 1:
            nxt = path[i + 1]
            label = _edge_label(adj, sid, nxt)
            if label:
                lines.append(f"  Choice: {label}")
    return "\n".join(lines)

def adjacency_text(g: NarrativeGraph) -> str:
    out = defaultdict(list)
    for e in g.edges:
        out[e.src].append((e.dst, e.label))
    lines = []
    for sid in sorted(g.scenes.keys()):
        sc = g.scenes[sid]
        outs = ", ".join([f"{dst}({lab})" for dst, lab in out.get(sid, [])])
        lines.append(f"{sid} [{sc.stype}] -> {outs}")
    return "\n".join(lines)

def _shortest_path(adj, start, goal) -> List[int]:
    q = deque([start])
    parent = {start: None}
    while q:
        u = q.popleft()
        if u == goal:
            break
        for v, _lab in adj.get(u, []):
            if v not in parent:
                parent[v] = u
                q.append(v)
    if goal not in parent:
        return []
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def _edge_label(adj, src, dst) -> str:
    for v, lab in adj.get(src, []):
        if v == dst:
            return lab
    return ""
