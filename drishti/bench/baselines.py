"""Baselines reproducing the prototype's approach (see "Prototype vs Production").

* regex_ner          capitalised-sequence + keyword typing (no grammar, no validation)
* cooccurrence_edges every pair of entities in the same document is an edge
* exact_er           identical normalised strings are the same entity
* naive_fuzzy_er     all-pairs Jaro-Winkler >= t, then transitive closure (O(n²))
* bm25 / lsa only    single-signal retrieval (see retrieval module)
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from drishti.resolve.similarity import jaro_winkler

ORG_KW = ("Ltd", "Limited", "Pvt", "LLP", "Traders", "Enterprises", "Industries", "Bank", "Builders", "Logistics", "Exports",
          "Solutions", "Developers", "Technologies", "Constructions", "Infrastructure")
GOV_KW = ("Department", "Ministry", "Authority", "Corporation", "Police", "Board", "Court", "Commission")
CAPSEQ = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|of|&)){1,5}\b")


def regex_ner(text: str) -> list[dict[str, Any]]:
    out = []
    for m in CAPSEQ.finditer(text):
        s = m.group(0).strip()
        lab = "ORGANISATION" if any(k in s for k in ORG_KW) else "GOVT_BODY" if any(k in s for k in GOV_KW) else "PERSON"
        out.append({"label": lab, "start": m.start(), "end": m.start() + len(s), "text": s})
    for m in re.finditer(r"\b[6-9]\d{9}\b", text):
        out.append({"label": "PHONE", "start": m.start(), "end": m.end(), "text": m.group(0)})
    for m in re.finditer(r"\b\d{12}\b", text):
        out.append({"label": "ACCOUNT", "start": m.start(), "end": m.end(), "text": m.group(0)})
    return out


def cooccurrence_edges(mentions_by_doc: dict[str, list[str]]) -> set[tuple[str, str, str]]:
    edges = set()
    for doc, ents in mentions_by_doc.items():
        u = sorted(set(ents))
        for i in range(len(u)):
            for j in range(i + 1, len(u)):
                edges.add((doc, u[i], u[j]))
    return edges


def exact_er(mentions: list[dict[str, Any]]) -> dict[str, str]:
    return {m["mention_id"]: f"{m['label']}|{m['normalized']}" for m in mentions}


def naive_fuzzy_er(mentions: list[dict[str, Any]], t: float = 0.88) -> dict[str, str]:
    parent = {m["mention_id"]: m["mention_id"] for m in mentions}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_label = defaultdict(list)
    for m in mentions:
        by_label[m["label"]].append(m)
    for ms in by_label.values():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                if jaro_winkler(ms[i]["normalized"], ms[j]["normalized"]) >= t:
                    parent[find(ms[j]["mention_id"])] = find(ms[i]["mention_id"])
    return {m: find(m) for m in parent}
