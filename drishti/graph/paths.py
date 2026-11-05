"""Explainable connection discovery between two entities.

Paths are enumerated in order of evidence cost with Yen's k-shortest simple
paths (networkx.shortest_simple_paths), where
    cost(edge) = -log(belief) + λ_hop  (+ penalty if the edge is weak)
so the top path maximises the product of edge beliefs with a mild preference
for fewer hops. Each path is scored and returned with every hop's claims,
evidence passages, direction, independence count and an honest label:
    direct    — a single typed relation between the two entities
    indirect  — a chain; NOT evidence of a direct relationship
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import networkx as nx

from drishti.graph.projection import WEAK, GraphView


def _hop_graph(view: GraphView, include_weak: bool, min_belief: float, exclude_types: set[str]) -> nx.Graph:
    H = nx.Graph()
    for (s, p, o), e in view.edges.items():
        if s == o or (p in WEAK and not include_weak) or e["belief"] < min_belief:
            continue
        if view.G.nodes.get(s, {}).get("type") in exclude_types or view.G.nodes.get(o, {}).get("type") in exclude_types:
            continue
        c = -math.log(max(1e-6, min(0.999999, e["belief"]))) + 0.15 + (1.5 if p in WEAK else 0.0)
        if H.has_edge(s, o):
            d = H[s][o]
            d["rels"].append((s, p, o))
            d["cost"] = min(d["cost"], c)
        else:
            H.add_edge(s, o, cost=c, rels=[(s, p, o)])
    return H


def find_paths(view: GraphView, source: str, target: str, k: int = 5, max_hops: int = 4, include_weak: bool = False,
               min_belief: float = 0.05, exclude_types: set[str] | None = None, claim_lookup=None) -> list[dict[str, Any]]:
    H = _hop_graph(view, include_weak, min_belief, exclude_types or set())
    if source not in H or target not in H:
        return []
    out = []
    try:
        gen = nx.shortest_simple_paths(H, source, target, weight="cost")
        for path in itertools.islice(gen, k * 4):
            if len(path) - 1 > max_hops:
                continue
            hops, belief, indep = [], 1.0, []
            for u, v in zip(path, path[1:]):
                rels = H[u][v]["rels"]
                best = max(rels, key=lambda r: view.edges[r]["belief"])
                e = view.edges[best]
                belief *= e["belief"]
                indep.append(e["independent_sources"])
                hops.append({
                    "from": u, "to": v, "from_name": view.G.nodes[u].get("name"), "to_name": view.G.nodes[v].get("name"),
                    "predicate": best[1], "direction": "forward" if best[0] == u else "reverse", "belief": e["belief"],
                    "independent_sources": e["independent_sources"], "weak": e["weak"], "claims": e["claims"],
                    "alternatives": [r[1] for r in rels if r != best], "valid_from": e["valid_from"], "valid_to": e["valid_to"],
                    "evidence": [claim_lookup(cid) for cid in e["claims"][:3]] if claim_lookup else [],
                })
            out.append({"nodes": path, "hops": len(path) - 1, "path_belief": round(belief, 4),
                        "min_independent_sources": min(indep), "kind": "direct" if len(path) == 2 else "indirect",
                        "contains_weak_edge": any(h["weak"] for h in hops), "edges": hops,
                        "score": round(belief * (1 + 0.1 * min(indep)) / (1 + 0.15 * (len(path) - 2)), 4)})
            if len(out) >= k:
                break
    except nx.NetworkXNoPath:
        return []
    return sorted(out, key=lambda p: -p["score"])


def neighbourhood(view: GraphView, entity: str, hops: int = 2, include_weak: bool = False, limit: int = 200) -> dict[str, Any]:
    H = _hop_graph(view, include_weak, 0.0, set())
    if entity not in H:
        return {"nodes": [entity], "edges": []}
    dist = nx.single_source_shortest_path_length(H, entity, cutoff=hops)
    nodes = sorted(dist, key=lambda v: dist[v])[:limit]
    ns = set(nodes)
    edges = [{"s": s, "p": p, "o": o, **{k: e[k] for k in ("belief", "independent_sources", "weak", "valid_from", "valid_to")}}
             for (s, p, o), e in view.edges.items() if s in ns and o in ns and (include_weak or not e["weak"])]
    return {"nodes": nodes, "distance": {v: dist[v] for v in nodes}, "edges": edges}
