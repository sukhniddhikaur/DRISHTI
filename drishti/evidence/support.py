"""Source independence, support aggregation and contradiction detection.

Two documents repeating one filing are *one* origin, not two. We group the
claims behind an edge into origin groups using
  (1) same document / same version chain,
  (2) near-duplicate evidence text (MinHash over word 4-shingles, J >= 0.5),
  (3) explicit attribution ("according to the filing ...") to an origin that
      is itself present for the same triple.
Belief is aggregated with noisy-OR **across origin groups only**:
    support = 1 - prod_g (1 - w(modality) * max_{c in g} p_c)
and the same for negating claims; belief = support * (1 - contra).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from drishti.core.domain import DomainPack
from drishti.resolve.similarity import MinHashLSH

MODALITY_WEIGHT = {"asserted": 1.0, "reported": 0.8, "hedged": 0.5, "negated": 1.0}
ATTRIBUTION_TARGETS = {"filing": "official_record", "judgment": "official_record", "order": "official_record",
                       "annual report": "corporate_disclosure", "fir": "official_record", "chargesheet": "official_record",
                       "tender": "official_record", "statement": "structured_record"}

_lsh = MinHashLSH(num_perm=30, bands=10, seed=11)


def _shingles(text: str, k: int = 4) -> set[str]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


_SIG_CACHE: dict[str, Any] = {}


def _sig(text: str):
    s = _SIG_CACHE.get(text)
    if s is None:
        if len(_SIG_CACHE) > 50_000:
            _SIG_CACHE.clear()
        s = _SIG_CACHE[text] = _lsh.signature(_shingles(text))
    return s


def jaccard_est(a: str, b: str) -> float:
    return float((_sig(a) == _sig(b)).mean())


@dataclass
class OriginGroup:
    key: str
    claim_ids: list[str] = field(default_factory=list)
    documents: set[str] = field(default_factory=set)
    reliability: set[str] = field(default_factory=set)
    reason: str = "distinct document"


def origin_groups(claims: list[dict[str, Any]], docs: dict[str, dict[str, Any]], dup_threshold: float = 0.5) -> list[OriginGroup]:
    # union-find over claims
    parent = {c["claim_id"]: c["claim_id"] for c in claims}
    reason: dict[str, str] = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, why):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            reason[ra] = why

    def chain_root(did: str) -> str:
        d = docs.get(did) or {}
        seen = set()
        while d and d.get("parent_document_id") and d["document_id"] not in seen:
            seen.add(d["document_id"])
            d = docs.get(d["parent_document_id"]) or {}
        return d.get("document_id", did)

    roots = {c["claim_id"]: chain_root(c["document_id"]) for c in claims}
    for i, a in enumerate(claims):
        for b in claims[i + 1:]:
            if find(a["claim_id"]) == find(b["claim_id"]):
                continue
            if roots[a["claim_id"]] == roots[b["claim_id"]]:
                union(a["claim_id"], b["claim_id"], "same document / version chain")
                continue
            if jaccard_est(a["evidence_text"], b["evidence_text"]) >= dup_threshold:
                union(a["claim_id"], b["claim_id"], "near-duplicate text (syndicated or copied)")
    # attribution: a reported claim that cites a record type joins a group containing that type
    for c in claims:
        if c["modality"] != "reported":
            continue
        cue = " ".join((c.get("attributes") or {}).get("cues", [])) + " " + c["evidence_text"].lower()
        for word, rel in ATTRIBUTION_TARGETS.items():
            if word in cue:
                for o in claims:
                    if o is not c and (docs.get(o["document_id"]) or {}).get("reliability") == rel:
                        union(o["claim_id"], c["claim_id"], f"attributed to {word}")
                        break
    groups: dict[str, OriginGroup] = {}
    for c in claims:
        r = find(c["claim_id"])
        g = groups.setdefault(r, OriginGroup(key=r, reason=reason.get(r, "distinct document")))
        g.claim_ids.append(c["claim_id"])
        g.documents.add(c["document_id"])
        g.reliability.add((docs.get(c["document_id"]) or {}).get("reliability", "unknown"))
    return list(groups.values())


def aggregate(claims: list[dict[str, Any]], docs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    live = [c for c in claims if c["status"] != "rejected" and c["epistemic"] != "rejected"]
    if any(c["epistemic"] == "confirmed" or c["status"] == "confirmed" for c in live if c["modality"] != "negated"):
        conf_override = 0.99
    else:
        conf_override = None
    pos = [c for c in live if c["modality"] != "negated"]
    neg = [c for c in live if c["modality"] == "negated"]
    by_id = {c["claim_id"]: c for c in live}

    def noisy_or(cs):
        if not cs:
            return 0.0, []
        gs = origin_groups(cs, docs)
        prod = 1.0
        for g in gs:
            p = max(MODALITY_WEIGHT.get(by_id[cid]["modality"], 1.0) * by_id[cid]["confidence"] for cid in g.claim_ids)
            prod *= 1 - p
        return 1 - prod, gs

    support, gpos = noisy_or(pos)
    contra, gneg = noisy_or(neg)
    belief = conf_override if conf_override is not None else support * (1 - contra)
    return {"support": round(support, 4), "contra": round(contra, 4), "belief": round(belief, 4),
            "independent_sources": len(gpos), "supporting_documents": len({c["document_id"] for c in pos}),
            "origin_groups": [{"claims": g.claim_ids, "documents": sorted(g.documents), "reason": g.reason,
                               "reliability": sorted(g.reliability)} for g in gpos],
            "contra_groups": len(gneg), "claims": [c["claim_id"] for c in live],
            "modalities": sorted({c["modality"] for c in live})}


# ============================================================================= contradictions
def _overlap(a_from, a_to, b_from, b_to) -> bool:
    lo = max(a_from or "0000", b_from or "0000")
    hi = min(a_to or "9999", b_to or "9999")
    return lo <= hi


def contradictions(claims: list[dict[str, Any]], pack: DomainPack, events: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    live = [c for c in claims if c["status"] != "rejected" and c.get("subject_entity_id") and c.get("object_entity_id")
            and c["predicate"] != "MENTIONED_WITH"]
    by_triple: dict[tuple, list[dict]] = defaultdict(list)
    for c in live:
        by_triple[(c["subject_entity_id"], c["predicate"], c["object_entity_id"])].append(c)

    # 1. polarity
    for (s, p, o), cs in by_triple.items():
        pos = [c for c in cs if c["modality"] != "negated"]
        neg = [c for c in cs if c["modality"] == "negated"]
        for a in pos:
            for b in neg:
                if _overlap(a["valid_from"], a["valid_to"], b["valid_from"], b["valid_to"]) and a["document_id"] != b["document_id"]:
                    out.append({"type": "polarity", "claims": [a["claim_id"], b["claim_id"]], "subject": s, "object": o, "predicate": p,
                                "explanation": f"One source asserts {p}; another explicitly denies it for an overlapping period."})
    # 2. functional predicates
    by_sp: dict[tuple, list[dict]] = defaultdict(list)
    for c in live:
        if c["predicate"] in pack.relations and pack.relations[c["predicate"]].functional and c["modality"] != "negated":
            by_sp[(c["subject_entity_id"], c["predicate"])].append(c)
    for (s, p), cs in by_sp.items():
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                if a["object_entity_id"] != b["object_entity_id"] and _overlap(a["valid_from"], a["valid_to"], b["valid_from"], b["valid_to"]):
                    out.append({"type": "functional", "claims": [a["claim_id"], b["claim_id"]], "subject": s, "predicate": p,
                                "explanation": f"{p} admits one object at a time, but two different objects are claimed for overlapping periods."})
    # 3. temporal staleness: relation ended at T, later source reports it as current
    for (s, p, o), cs in by_triple.items():
        ended = [c for c in cs if c["valid_to"] and c["modality"] != "negated"]
        current = [c for c in cs if not c["valid_to"] and c["modality"] in ("asserted", "reported") and c["reported_at"]]
        for e in ended:
            for c in current:
                if c["reported_at"] > e["valid_to"] and c["document_id"] != e["document_id"]:
                    out.append({"type": "temporal", "claims": [e["claim_id"], c["claim_id"]], "subject": s, "object": o, "predicate": p,
                                "explanation": f"Relation reported as ended on {e['valid_to']}, but a source published {c['reported_at']} still states it."})
    # 4. event attribute conflicts: same contract reference, different amount/date
    by_ref: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        ref = (e.get("attributes") or {}).get("contract_ref")
        if ref:
            by_ref[ref].append(e)
    for ref, es in by_ref.items():
        amts = {(e["attributes"] or {}).get("amount_inr") for e in es} - {None}
        dates = {e["date"] for e in es} - {None}
        if len(amts) > 1 or len(dates) > 1:
            out.append({"type": "attribute", "claims": [(e["attributes"] or {}).get("claim_id") for e in es], "contract_ref": ref,
                        "explanation": f"Contract {ref} is reported with differing "
                                       + " and ".join(x for x, v in (("amounts", len(amts) > 1), ("dates", len(dates) > 1)) if v)
                                       + f": amounts={sorted(amts)}, dates={sorted(dates)}."})
    # 5. same transaction-like triple reported with materially different amounts
    for (s, p, o), cs in by_triple.items():
        if p not in ("AWARDED_CONTRACT_TO", "ACQUIRED"):
            continue
        amts = {}
        for c in cs:
            a = (c.get("attributes") or {}).get("amount_inr")
            if a and c["modality"] != "negated":
                amts.setdefault(a, c)
        vals = sorted(amts)
        if len(vals) > 1 and (vals[-1] - vals[0]) / vals[-1] > 0.01:
            out.append({"type": "attribute", "claims": [amts[v]["claim_id"] for v in vals], "subject": s, "object": o, "predicate": p,
                        "explanation": f"{p} between the same parties is reported with different amounts: "
                                       + ", ".join(f"₹{v:,}" for v in vals) + "."})
    seen, uniq = set(), []
    for c in out:
        k = (c["type"], tuple(sorted(x for x in c["claims"] if x)))
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq
