"""Link prediction as *hypothesis generation* (epistemic = inferred).

Topological scores over the evidence-weighted graph:
  Adamic–Adar, Resource Allocation, Jaccard, Preferential Attachment,
  weighted common-neighbour belief, plus shared hard identifiers.
Scores are combined by a logistic model trained with edge hold-out on the
case's own graph (self-supervised), and evaluated by ROC-AUC on a disjoint
hold-out. Every hypothesis lists the shared neighbours and the claims that
connect them — it is a lead, never a stored fact.
"""
from __future__ import annotations

import math
import random
from typing import Any

import networkx as nx
import numpy as np

FEATS = ["aa", "ra", "jac", "pa_log", "wcn", "cn"]


def pair_features(U: nx.Graph, u: str, v: str) -> dict[str, float]:
    nu, nv = set(U[u]), set(U[v])
    cn = nu & nv
    aa = sum(1 / math.log(max(2, U.degree(w))) for w in cn)
    ra = sum(1 / max(1, U.degree(w)) for w in cn)
    jac = len(cn) / len(nu | nv) if nu | nv else 0.0
    pa = math.log1p(len(nu) * len(nv))
    wcn = sum(U[u][w].get("weight", 0.5) * U[v][w].get("weight", 0.5) for w in cn)
    return {"aa": aa, "ra": ra, "jac": jac, "pa_log": pa, "wcn": wcn, "cn": float(len(cn))}


def _candidates(U: nx.Graph, types: dict[str, str] | None, allowed: set[tuple[str, str]] | None) -> list[tuple[str, str]]:
    seen = set()
    for w in U:
        nb = list(U[w])
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                a, b = sorted((nb[i], nb[j]))
                if U.has_edge(a, b) or (a, b) in seen:
                    continue
                if allowed and types and (types.get(a), types.get(b)) not in allowed and (types.get(b), types.get(a)) not in allowed:
                    continue
                seen.add((a, b))
    return sorted(seen)


def train_and_evaluate(U: nx.Graph, holdout: float = 0.2, seed: int = 7) -> dict[str, Any]:
    """Edge hold-out: hide a fraction of edges, train on the rest, report AUC per score."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rnd = random.Random(seed)
    edges = [e for e in U.edges() if U.degree(e[0]) > 1 and U.degree(e[1]) > 1]
    rnd.shuffle(edges)
    k = max(1, int(len(edges) * holdout))
    test_pos = edges[:k]
    train_pos = edges[k:2 * k] if len(edges) > 2 * k else edges[k:]
    R = U.copy()
    R.remove_edges_from(test_pos + train_pos)
    nodes = list(U)
    neg = set()
    tries = 0
    while len(neg) < 2 * k and tries < 50 * k:
        a, b = rnd.sample(nodes, 2)
        tries += 1
        if not U.has_edge(a, b):
            neg.add(tuple(sorted((a, b))))
    neg = list(neg)
    train_neg, test_neg = neg[:len(neg) // 2], neg[len(neg) // 2:]
    if not train_pos or not test_pos or not train_neg or not test_neg:
        return {"auc": None, "reason": "graph too small"}

    def X(pairs):
        return np.array([[pair_features(R, a, b)[f] for f in FEATS] for a, b in pairs])

    Xtr = np.vstack([X(train_pos), X(train_neg)])
    ytr = np.array([1] * len(train_pos) + [0] * len(train_neg))
    Xte = np.vstack([X(test_pos), X(test_neg)])
    yte = np.array([1] * len(test_pos) + [0] * len(test_neg))
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    res = {"auc_combined": round(float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])), 4),
           "n_test_pos": len(test_pos), "n_test_neg": len(test_neg), "model": clf}
    for i, f in enumerate(FEATS):
        try:
            res[f"auc_{f}"] = round(float(roc_auc_score(yte, Xte[:, i])), 4)
        except ValueError:
            pass
    return res


def predict(U: nx.Graph, top: int = 20, types: dict[str, str] | None = None,
            allowed: set[tuple[str, str]] | None = None, model=None, view=None) -> list[dict[str, Any]]:
    cands = _candidates(U, types, allowed)
    out = []
    for a, b in cands:
        f = pair_features(U, a, b)
        if model is not None:
            s = float(model.predict_proba(np.array([[f[k] for k in FEATS]]))[0, 1])
        else:
            s = 1 - math.exp(-(f["ra"] + 0.5 * f["wcn"]))
        shared = sorted(set(U[a]) & set(U[b]))
        out.append({"source": a, "target": b, "score": round(s, 4), "features": {k: round(v, 4) for k, v in f.items()},
                    "shared_neighbours": shared, "epistemic": "inferred",
                    "explanation": f"{len(shared)} shared neighbour(s); hypothesis only — no source states this link.",
                    "supporting_edges": [{"via": w, "a_predicates": U[a][w].get("predicates", []),
                                          "b_predicates": U[b][w].get("predicates", [])} for w in shared[:5]]})
    return sorted(out, key=lambda x: -x["score"])[:top]
