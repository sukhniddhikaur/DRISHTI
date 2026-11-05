"""Deterministic semantic parser: question -> IQR.

1. Entity linking: longest-match over every canonical name and alias of the
   case's resolved entities (normalised), then fuzzy n-gram fallback
   (Jaro-Winkler ≥ 0.9) so "ABC Infra" or "R. K. Sharma" still link.
2. Intent: ordered pattern rules (most specific first).
3. Slots: entity-type words ("companies", "people", "phones"), predicate words
   ("directors", "contracts", "accused"), time ("as of 30 June 2023"), hops.
"""
from __future__ import annotations

import re
from typing import Any

from drishti.assistant.iqr import IQR
from drishti.extract.temporal import find_dates
from drishti.resolve.normalize import normalize, org_parts, person_core
from drishti.resolve.similarity import jaro_winkler

TYPE_WORDS = {
    "ORGANISATION": ("company", "companies", "firm", "firms", "organisation", "organisations", "organization", "organizations", "supplier", "suppliers", "contractor", "contractors", "business"),
    "PERSON": ("people", "person", "persons", "individual", "individuals", "who", "directors", "director", "suspects", "accused", "associates"),
    "PHONE": ("phone", "phones", "mobile", "mobiles", "number", "numbers"),
    "ACCOUNT": ("account", "accounts", "bank"),
    "GOVT_BODY": ("department", "departments", "authority", "buyer", "buyers", "government"),
    "VEHICLE": ("vehicle", "vehicles", "car", "cars"),
    "CASE": ("fir", "firs", "case", "cases"),
}
PRED_WORDS = [
    (r"\bdirector|board\b", "DIRECTOR_OF"), (r"\bsubsidiar|parent|holding compan", "SUBSIDIARY_OF"),
    (r"\bcontract|tender|work order|award", "AWARDED_CONTRACT_TO"), (r"\bsued|petition|litigation|case against", "FILED_CASE_AGAINST"),
    (r"\bacquir|takeover|took over", "ACQUIRED"), (r"\bsharehold|stake|promoter", "SHAREHOLDER_OF"),
    (r"\baccused|arrested|named in", "ACCUSED_IN"), (r"\bcomplainant", "COMPLAINANT_IN"), (r"\bassociate", "ASSOCIATE_OF"),
    (r"\bphone|mobile", "USES_PHONE"), (r"\baccount\b", "HOLDS_ACCOUNT"), (r"\bvehicle|car\b", "OWNS_VEHICLE"),
    (r"\bcall(?:s|ed)?\b", "CALLED"), (r"\btransfer|paid|payment|money|funds", "TRANSFERRED_TO"), (r"\blocated|based|office", "LOCATED_IN"),
]


class EntityLinker:
    def __init__(self, entities: list[dict[str, Any]]):
        self.surface: dict[str, str] = {}
        self.ents = {e["entity_id"]: e for e in entities}
        for e in entities:
            for n in {e["canonical_name"], *(e.get("aliases") or [])}:
                for key in self._keys(e["type"], n):
                    if len(key) >= 3:
                        self.surface.setdefault(key, e["entity_id"])

    @staticmethod
    def _keys(t: str, n: str) -> set[str]:
        k = {n.lower().strip()}
        if t == "ORGANISATION":
            k.add(org_parts(n)[0])
        elif t == "PERSON":
            k.add(person_core(n))
        elif t == "PHONE":
            k.add(re.sub(r"\D", "", n)[-10:])
        return {x for x in k if x}

    def link(self, q: str) -> list[tuple[str, int, int]]:
        low = q.lower()
        digits_q = re.sub(r"(?<=\d)[\s-](?=\d)", "", low)
        found: list[tuple[str, int, int]] = []
        taken = [False] * len(low)
        for key in sorted(self.surface, key=len, reverse=True):
            hay = digits_q if key.isdigit() else low
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", hay):
                if key.isdigit() or not any(taken[m.start():m.end()]):
                    found.append((self.surface[key], m.start(), m.end()))
                    if not key.isdigit():
                        for i in range(m.start(), m.end()):
                            taken[i] = True
        if len({f[0] for f in found}) < 2:
            # fuzzy fallback over capitalised n-grams of the original question
            words = re.findall(r"[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,4}", q)
            for w in words:
                wl = w.lower()
                best, bs = None, 0.0
                for key, eid in self.surface.items():
                    s = jaro_winkler(wl, key)
                    if s > bs:
                        best, bs = eid, s
                if best and bs >= 0.9 and best not in {f[0] for f in found}:
                    i = q.find(w)
                    found.append((best, i, i + len(w)))
        uniq, seen = [], set()
        for eid, a, b in sorted(found, key=lambda x: x[1]):
            if eid not in seen:
                seen.add(eid)
                uniq.append((eid, a, b))
        return uniq


def parse_question(q: str, case_id: str, linker: EntityLinker) -> IQR:
    low = q.lower().strip()
    links = linker.link(q)
    ids = [e for e, _, _ in links]
    # strip linked spans before reading type words so "Sunrise Traders" doesn't imply PERSON etc.
    masked = list(low)
    for _, a, b in links:
        masked[a:b] = " " * (b - a)
    rest = "".join(masked)
    types = [t for t, ws in TYPE_WORDS.items() if any(re.search(rf"\b{re.escape(w)}\b", rest) for w in ws)]
    preds = list(dict.fromkeys(p for rx, p in PRED_WORDS if re.search(rx, rest)))
    ds = find_dates(q)
    as_of = ds[0].iso if ds and re.search(r"\b(as of|as on|on|at|in|during|before|after)\b", low) else None
    hops = 2
    m = re.search(r"(\d)\s*(?:hops?|steps?|degrees?)", low)
    if m:
        hops = max(1, min(4, int(m.group(1))))
    weak = bool(re.search(r"\b(include|with) (weak|co-?occurrence|mentions?)\b", low))

    def mk(intent: str, **kw) -> IQR:
        base = dict(intent=intent, case_id=case_id, entity_ids=ids[:2], entity_types=types, predicates=preds, as_of=as_of,
                    max_hops=hops, include_weak=weak, text=q)
        base.update(kw)
        return IQR(**base)

    if re.search(r"\b(contradict|conflict|inconsisten|disput)", low):
        return mk("contradictions", entity_types=[], predicates=[])
    if re.search(r"\b(suspicious|pattern|cycle|round.?trip|layering|structur|mule|pass.?through|interlock|burst|shared (phone|number|account))", low):
        return mk("patterns", entity_types=[], predicates=[])
    if re.search(r"\b(hidden|possible|potential|predict|likely|undiscovered) (link|connection|relationship)s?", low):
        return mk("hypotheses", entity_types=[], predicates=[])
    if re.search(r"\b(communit|cluster|groups?\b|gangs?|cells?\b|rings?\b)", low) and len(ids) < 1:
        return mk("communities", entity_types=[], predicates=[])
    if re.search(r"\b(central|influential|key (player|people|entit)|important|kingpin|broker|most connected|hub)", low):
        return mk("centrality", predicates=[])
    if len(ids) >= 2 and re.search(r"\b(connect|link|path|relat|between|tie|associated)", low):
        return mk("find_paths", entity_types=[], predicates=[], max_hops=max(hops, 3) if not m else hops)
    if ids and re.search(r"\b(timeline|history|chronolog|what happened|events?)\b", low):
        return mk("timeline", entity_types=[], predicates=[])
    if ids and re.search(r"^(who|what) is\b|\b(profile|tell me about|summar|everything about|details of|overview)", low) and not preds:
        return mk("entity_profile", entity_types=[], predicates=[])
    if ids and preds:
        direction = "in" if re.search(r"\b(of|for|in|by)\s*$|\b(directors of|subsidiaries of|accused in|named in|awarded by|from)\b", low) else "any"
        return mk("relation_lookup", direction=direction, max_hops=1)
    if ids and re.search(r"\b(connect|link|relat|associated|network|around|neighbou?r)", low):
        return mk("find_related", predicates=[])
    if ids:
        return mk("entity_profile", entity_types=[], predicates=[])
    return mk("evidence_search", entity_types=[], predicates=[])
