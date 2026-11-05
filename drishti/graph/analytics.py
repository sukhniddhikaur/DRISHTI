"""Structural graph intelligence: centrality, communities, brokers, resilience.

All metrics are computed on the *evidence-weighted* undirected projection
(edge weight = aggregated belief), excluding weak co-occurrence edges by
default. Distances for betweenness use cost = -log(belief) so that a path
through well-evidenced edges is "shorter" than one through doubtful edges.
"""
from __future__ import annotations

import math
import random
from typing import Any

import networkx as nx


def _cost_graph(U: nx.Graph) -> nx.Graph:
    H = U.copy()
    for _, _, d in H.edges(data=True):
        d["cost"] = -math.log(max(1e-6, min(0.999999, d.get("weight", 0.5)))) + 0.05
    return H


def centrality(U: nx.Graph, k_sample: int | None = None) -> dict[str, dict[str, float]]:
    if U.number_of_nodes() == 0:
        return {}
    H = _cost_graph(U)
    n = U.number_of_nodes()
    k = min(n, k_sample) if k_sample and n > k_sample else None
    btw = nx.betweenness_centrality(H, weight="cost", k=k, seed=7, normalized=True)
    pr = nx.pagerank(U, weight="weight") if U.number_of_edges() else {v: 1 / n for v in U}
    deg = nx.degree_centrality(U)
    wdeg = dict(U.degree(weight="weight"))
    try:
        eig = nx.eigenvector_centrality_numpy(U, weight="weight") if U.number_of_edges() else {}
    except Exception:
        eig = {}
    core = nx.core_number(U) if U.number_of_edges() else {v: 0 for v in U}
    art = set(nx.articulation_points(U)) if U.number_of_edges() else set()
    return {v: {"degree": round(deg[v], 4), "weighted_degree": round(wdeg.get(v, 0), 4), "betweenness": round(btw[v], 4),
                "pagerank": round(pr[v], 5), "eigenvector": round(float(eig.get(v, 0)), 4), "k_core": core.get(v, 0),
                "articulation_point": v in art} for v in U}


def communities(U: nx.Graph, resolution: float = 1.0, seed: int = 7) -> dict[str, Any]:
    if U.number_of_edges() == 0:
        return {"communities": [[v] for v in U], "modularity": 0.0, "method": "none"}
    comms = nx.community.louvain_communities(U, weight="weight", resolution=resolution, seed=seed)
    comms = sorted((sorted(c) for c in comms), key=len, reverse=True)
    q = nx.community.modularity(U, comms, weight="weight")
    lp = list(nx.community.asyn_lpa_communities(U, weight="weight", seed=seed))
    q_lp = nx.community.modularity(U, lp, weight="weight")
    # per-community summary: internal density and the structurally central member
    summary = []
    for i, c in enumerate(comms):
        sub = U.subgraph(c)
        pr = nx.pagerank(sub, weight="weight") if sub.number_of_edges() else {v: 1 for v in c}
        cut = sum(1 for u in c for w in U[u] if w not in set(c))
        summary.append({"id": i, "size": len(c), "members": c, "density": round(nx.density(sub), 4),
                        "boundary_edges": cut, "most_central": max(pr, key=pr.get)})
    return {"method": "louvain", "modularity": round(q, 4), "label_propagation_modularity": round(q_lp, 4),
            "communities": comms, "summary": summary}


def global_efficiency_w(U: nx.Graph) -> float:
    return nx.global_efficiency(U) if U.number_of_nodes() > 1 else 0.0


def resilience(U: nx.Graph, steps: int = 10, strategy: str = "betweenness", seed: int = 7) -> list[dict[str, float]]:
    """Disruption simulation: remove nodes one at a time, recomputing the target each step
    (adaptive attack), and record largest-component fraction + global efficiency."""
    H = U.copy()
    n0 = max(1, H.number_of_nodes())
    rnd = random.Random(seed)
    base_eff = global_efficiency_w(H) or 1e-9
    out = [{"removed": 0, "node": None, "lcc_fraction": round(max((len(c) for c in nx.connected_components(H)), default=0) / n0, 4),
            "efficiency_ratio": 1.0}]
    for step in range(1, min(steps, H.number_of_nodes()) + 1):
        if strategy == "random":
            v = rnd.choice(list(H.nodes))
        elif strategy == "degree":
            v = max(H.degree, key=lambda x: x[1])[0]
        else:
            b = nx.betweenness_centrality(_cost_graph(H), weight="cost")
            v = max(b, key=b.get)
        H.remove_node(v)
        lcc = max((len(c) for c in nx.connected_components(H)), default=0)
        out.append({"removed": step, "node": v, "lcc_fraction": round(lcc / n0, 4),
                    "efficiency_ratio": round(global_efficiency_w(H) / base_eff, 4)})
    return out


def components(U: nx.Graph) -> list[list[str]]:
    return sorted((sorted(c) for c in nx.connected_components(U)), key=len, reverse=True)
