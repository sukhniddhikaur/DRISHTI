"""Temporal graph intelligence: snapshots, diffs, entity timelines."""
from __future__ import annotations

from typing import Any

from drishti.graph.projection import build
from drishti.store.db import Store


def diff(store: Store, case_id: str, t1: str, t2: str) -> dict[str, Any]:
    """Edges valid at t2 but not t1 (new) and vice versa (ended). Uses strict valid time."""
    a = {k for k, e in build(store, case_id, as_of=t1, strict_time=True, include_weak=False).edges.items()}
    b = {k for k, e in build(store, case_id, as_of=t2, strict_time=True, include_weak=False).edges.items()}
    fmt = lambda ks: [{"subject": s, "predicate": p, "object": o} for s, p, o in sorted(ks)]
    return {"from": t1, "to": t2, "new": fmt(b - a), "ended": fmt(a - b), "persisting": len(a & b)}


def timeline(store: Store, case_id: str, entity_id: str | None = None) -> list[dict[str, Any]]:
    ents = {e["entity_id"]: e["canonical_name"] for e in store.entities(case_id)}
    m2e = {r["mention_id"]: r["entity_id"] for r in store.q(
        "SELECT me.mention_id, me.entity_id FROM mention_entity me JOIN entities e USING(entity_id) WHERE e.case_id=? AND me.active=1",
        (case_id,))}
    items = []
    for ev in store.events(case_id):
        roles = {r: m2e.get(mid) for r, mid in (ev["roles"] or {}).items()}
        if entity_id and entity_id not in roles.values():
            continue
        items.append({"date": ev["date"], "kind": "event", "type": ev["event_type"], "roles": roles,
                      "role_names": {r: ents.get(e) for r, e in roles.items()}, "evidence": ev["evidence_text"],
                      "document_id": ev["document_id"], "claim_id": (ev["attributes"] or {}).get("claim_id"),
                      "attributes": ev["attributes"]})
    if entity_id:
        for c in store.claims(case_id):
            if entity_id in (c["subject_entity_id"], c["object_entity_id"]) and c["reported_at"] and c["predicate"] != "MENTIONED_WITH":
                items.append({"date": c["reported_at"], "kind": "reported", "type": c["predicate"], "claim_id": c["claim_id"],
                              "evidence": c["evidence_text"], "document_id": c["document_id"],
                              "role_names": {"subject": ents.get(c["subject_entity_id"]), "object": ents.get(c["object_entity_id"])}})
    return sorted([i for i in items if i["date"]], key=lambda i: (i["date"], i["kind"]))
