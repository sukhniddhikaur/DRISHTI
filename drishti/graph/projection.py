"""Bitemporal graph projection from the claim store.

The graph is a *view*: rebuildable at any time from claims, with two clocks:
  as_of    (valid time)       — "what relationships held on 2023-06-30?"
  known_at (transaction time) — "what had DRISHTI / the sources said by then?"
Edges aggregate claims per (subject, predicate, object) with belief computed
by the evidence engine; weak co-occurrence edges are kept separate and never
shown as relationships unless explicitly requested.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx

from drishti.evidence.support import aggregate
from drishti.store.db import Store

WEAK = {"MENTIONED_WITH"}


@dataclass
class GraphView:
    G: nx.MultiDiGraph
    edges: dict[tuple[str, str, str], dict[str, Any]]

    def undirected(self, include_weak: bool = False, min_belief: float = 0.0,
                   exclude_types: frozenset[str] = frozenset({"LOCATION"})) -> nx.Graph:
        """Evidence-weighted simple graph. LOCATION nodes are excluded by default: a city
        shared by hundreds of entities is an attribute hub, not a social tie."""
        U = nx.Graph()
        U.add_nodes_from((n, d) for n, d in self.G.nodes(data=True) if d.get("type") not in exclude_types)
        for (s, p, o), e in self.edges.items():
            if (p in WEAK and not include_weak) or e["belief"] < min_belief or s == o or s not in U or o not in U:
                continue
            if U.has_edge(s, o):
                d = U[s][o]
                d["weight"] = 1 - (1 - d["weight"]) * (1 - e["belief"])
                d["predicates"].append(p)
            else:
                U.add_edge(s, o, weight=e["belief"], predicates=[p])
        return U


def _valid_at(c: dict[str, Any], t: str, strict: bool) -> bool:
    vf, vt = c["valid_from"], c["valid_to"]
    if vf is None and vt is None:
        return not strict
    return (vf or "0000") <= t <= (vt or "9999")


def build(store: Store, case_id: str, as_of: str | None = None, known_at: str | None = None, strict_time: bool = False,
          include_weak: bool = True, min_belief: float = 0.0, predicates: set[str] | None = None) -> GraphView:
    docs = {d["document_id"]: d for d in store.documents(case_id)}
    claims = [c for c in store.claims(case_id) if c["subject_entity_id"] and c["object_entity_id"]]
    if predicates:
        claims = [c for c in claims if c["predicate"] in predicates]
    if as_of:
        claims = [c for c in claims if _valid_at(c, as_of, strict_time)]
    if known_at:
        claims = [c for c in claims if (c["reported_at"] or c["ingested_at"][:10]) <= known_at]
    if not include_weak:
        claims = [c for c in claims if c["predicate"] not in WEAK]
    G = nx.MultiDiGraph()
    for e in store.entities(case_id):
        G.add_node(e["entity_id"], type=e["type"], name=e["canonical_name"], identifiers=e["identifiers"])
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for c in claims:
        grouped[(c["subject_entity_id"], c["predicate"], c["object_entity_id"])].append(c)
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (s, p, o), cs in grouped.items():
        agg = aggregate(cs, docs)
        if agg["belief"] < min_belief and p not in WEAK:
            continue
        vfs = [c["valid_from"] for c in cs if c["valid_from"]]
        vts = [c["valid_to"] for c in cs if c["valid_to"]]
        attrs = {
            **agg, "predicate": p, "weak": p in WEAK,
            "valid_from": min(vfs) if vfs else None, "valid_to": max(vts) if vts else None,
            "first_reported": min((c["reported_at"] for c in cs if c["reported_at"]), default=None),
            "epistemic": sorted({c["epistemic"] for c in cs}),
            "amount_inr": sum((c["attributes"] or {}).get("amount_inr") or 0 for c in cs) or None,
            "count": len(cs),
        }
        edges[(s, p, o)] = attrs
        if s in G and o in G:
            G.add_edge(s, o, key=p, **attrs)
    return GraphView(G, edges)
