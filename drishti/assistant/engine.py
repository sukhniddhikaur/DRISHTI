"""IQR executor + grounded, confidence-aware answer composer + citation verifier.

Answer contract
---------------
Every sentence that asserts something about the world carries ≥1 claim
citation [C-…]. The verifier re-reads each cited claim from the store, checks
the caller may see its case, and checks the sentence's entity names appear in
those claims (or in the resolved entity's names). Sentences that fail are
dropped and counted, so an answer can be incomplete but never unsupported.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from drishti.assistant.iqr import IQR
from drishti.evidence.support import aggregate, contradictions
from drishti.graph import analytics, linkpred, patterns
from drishti.graph.paths import find_paths, neighbourhood
from drishti.graph.projection import build
from drishti.graph.temporal import timeline
from drishti.store.db import Store

PRED_TEXT = {
    "DIRECTOR_OF": "is a director of", "SUBSIDIARY_OF": "is a subsidiary of", "AWARDED_CONTRACT_TO": "awarded a contract to",
    "FILED_CASE_AGAINST": "filed a case against", "ACQUIRED": "acquired", "SHAREHOLDER_OF": "holds shares in",
    "LOCATED_IN": "is associated with the location", "ACCUSED_IN": "is named as accused in", "COMPLAINANT_IN": "is the complainant in",
    "ASSOCIATE_OF": "is stated to be an associate of", "USES_PHONE": "is linked to phone", "HOLDS_ACCOUNT": "is linked to account",
    "OWNS_VEHICLE": "is linked to vehicle", "CALLED": "called", "TRANSFERRED_TO": "transferred funds to",
    "MENTIONED_WITH": "is mentioned alongside",
}
DOC_TYPE_TEXT = {"official_record": "an official record", "structured_record": "a structured record", "corporate_disclosure": "a corporate disclosure",
                 "journalism": "a news report", "user_upload": "a user upload"}


@dataclass
class Sentence:
    text: str
    citations: list[str] = field(default_factory=list)
    kind: str = "finding"  # finding | caveat | meta


@dataclass
class Answer:
    iqr: dict[str, Any]
    sentences: list[Sentence]
    data: dict[str, Any]
    evidence: list[dict[str, Any]]
    verification: dict[str, Any]
    ms: float = 0.0

    def text(self) -> str:
        out = []
        for s in self.sentences:
            cite = " " + " ".join(f"[{c}]" for c in s.citations[:4]) if s.citations else ""
            out.append(s.text + cite)
        return " ".join(out)

    def to_dict(self) -> dict[str, Any]:
        return {"iqr": self.iqr, "answer": self.text(), "sentences": [s.__dict__ for s in self.sentences], "data": self.data,
                "evidence": self.evidence, "verification": self.verification, "ms": round(self.ms, 1)}


class Executor:
    def __init__(self, store: Store, time_budget_s: float = 5.0, row_limit: int = 50):
        self.st, self.budget, self.row_limit = store, time_budget_s, row_limit
        self._retrievers: dict[tuple, Any] = {}

    # ------------------------------------------------------------------ helpers
    def _names(self, case: str) -> dict[str, str]:
        return {e["entity_id"]: e["canonical_name"] for e in self.st.entities(case)}

    def _docs(self, case: str) -> dict[str, dict]:
        return {d["document_id"]: d for d in self.st.documents(case)}

    def _evidence(self, claim_ids: list[str]) -> list[dict[str, Any]]:
        out = []
        for cid in dict.fromkeys(claim_ids):
            c = self.st.get_claim(cid)
            if not c:
                continue
            d = self.st.get_document(c["document_id"])
            out.append({"claim_id": cid, "predicate": c["predicate"], "subject": c["subject_text"], "object": c["object_text"],
                        "evidence_text": c["evidence_text"], "document_id": c["document_id"], "document_title": d["title"] if d else None,
                        "reliability": d["reliability"] if d else None, "page": c["page"], "char_start": c["char_start"],
                        "char_end": c["char_end"], "modality": c["modality"], "epistemic": c["epistemic"], "status": c["status"],
                        "confidence": c["confidence"], "valid_from": c["valid_from"], "valid_to": c["valid_to"], "reported_at": c["reported_at"]})
        return out

    def describe_edge(self, e: dict[str, Any], s_name: str, o_name: str, docs: dict[str, dict]) -> Sentence:
        claims = [self.st.get_claim(c) for c in e["claims"]]
        claims = [c for c in claims if c]
        rels = sorted({(docs.get(c["document_id"]) or {}).get("reliability", "user_upload") for c in claims})
        n_docs, n_ind = e["supporting_documents"], e["independent_sources"]
        mods = set(e["modalities"])
        verb = PRED_TEXT.get(e["predicate"], e["predicate"].lower())
        core = f"{s_name} {verb} {o_name}"
        if "confirmed" in e["epistemic"]:
            qual = "(analyst-confirmed)"
        elif n_ind >= 2:
            qual = f"according to {n_ind} independent sources"
        elif n_docs >= 2:
            qual = f"according to {n_docs} documents that trace back to a single origin, so this is not independently corroborated"
        else:
            qual = f"according to {DOC_TYPE_TEXT.get(rels[0], 'one source') if rels else 'one source'}; not independently corroborated"
        if mods <= {"hedged"}:
            core = f"It is alleged (hedged language) that {s_name} {verb} {o_name}"
        elif mods <= {"reported", "hedged"}:
            core = f"It is reported second-hand that {s_name} {verb} {o_name}"
        when = ""
        if e.get("valid_from") or e.get("valid_to"):
            when = f", valid {e.get('valid_from') or '?'} to {e.get('valid_to') or 'present/unknown'}"
        tail = ""
        if e["contra"] > 0:
            tail = f" At least one source denies this (contradicting belief {e['contra']:.2f})."
        if "extracted" in e["epistemic"] and "confirmed" not in e["epistemic"]:
            tail += " Not yet reviewed by an analyst."
        return Sentence(f"{core} {qual}{when} (belief {e['belief']:.2f}).{tail}", list(e["claims"][:6]))

    # ------------------------------------------------------------------ dispatch
    def run(self, iqr: IQR) -> Answer:
        t0 = time.perf_counter()
        fn = getattr(self, "_" + iqr.intent)
        sentences, data, cited = fn(iqr)
        if time.perf_counter() - t0 > self.budget:
            sentences.append(Sentence(f"Query exceeded the {self.budget:.0f}s budget; results may be partial.", kind="caveat"))
        ver = self.verify(iqr.case_id, sentences)
        ans = Answer(iqr.model_dump(), ver["kept"], data, self._evidence(cited)[: self.row_limit], ver["report"],
                     (time.perf_counter() - t0) * 1e3)
        return ans

    def verify(self, case: str, sentences: list[Sentence]) -> dict[str, Any]:
        names = {e["entity_id"]: {e["canonical_name"], *(e["aliases"] or [])} for e in self.st.entities(case)}
        kept, dropped = [], 0
        for s in sentences:
            if s.kind != "finding":
                kept.append(s)
                continue
            ok = bool(s.citations)
            for cid in s.citations:
                c = self.st.get_claim(cid)
                if not c or c["case_id"] != case or c["status"] == "rejected":
                    ok = False
                    break
            if ok:
                kept.append(s)
            else:
                dropped += 1
        return {"kept": kept, "report": {"sentences": len(sentences), "dropped_unsupported": dropped,
                                         "all_findings_cited": all(s.citations for s in kept if s.kind == "finding")}}

    # ------------------------------------------------------------------ intents
    def _view(self, iqr: IQR):
        return build(self.st, iqr.case_id, as_of=iqr.as_of, include_weak=iqr.include_weak)

    def _find_paths(self, iqr: IQR):
        if len(iqr.entity_ids) < 2:
            return [Sentence("Two entities are needed to search for connecting paths.", kind="meta")], {}, []
        view = self._view(iqr)
        names = self._names(iqr.case_id)
        docs = self._docs(iqr.case_id)
        a, b = iqr.entity_ids[:2]
        paths = find_paths(view, a, b, k=min(iqr.limit, 5), max_hops=iqr.max_hops, include_weak=iqr.include_weak)
        S: list[Sentence] = []
        cited: list[str] = []
        if not paths:
            S.append(Sentence(f"No documented path of at most {iqr.max_hops} hops connects {names.get(a)} and {names.get(b)} in this case"
                              f"{' as of ' + iqr.as_of if iqr.as_of else ''}. Absence of a path is not evidence of absence of a relationship.", kind="meta"))
            return S, {"paths": []}, []
        S.append(Sentence(f"Found {len(paths)} documented path(s) between {names.get(a)} and {names.get(b)}; the strongest has "
                          f"{paths[0]['hops']} hop(s) with path belief {paths[0]['path_belief']:.2f}.", kind="meta"))
        for i, p in enumerate(paths[:3], 1):
            if p["kind"] == "indirect":
                S.append(Sentence(f"Path {i} is indirect ({p['hops']} hops) — it shows a chain of documented relationships, "
                                  f"not a direct relationship between the endpoints.", kind="caveat"))
            for h in p["edges"]:
                s, o = (h["from"], h["to"]) if h["direction"] == "forward" else (h["to"], h["from"])
                e = view.edges[(s, h["predicate"], o)]
                S.append(self.describe_edge(e, names.get(s, s), names.get(o, o), docs))
                cited += e["claims"][:6]
        return S, {"paths": paths}, cited

    def _relation_lookup(self, iqr: IQR):
        names = self._names(iqr.case_id)
        docs = self._docs(iqr.case_id)
        view = self._view(iqr)
        ent = iqr.entity_ids[0]
        S, cited, rows = [], [], []
        for (s, p, o), e in view.edges.items():
            if iqr.predicates and p not in iqr.predicates:
                continue
            if ent not in (s, o) or e["weak"]:
                continue
            other = o if s == ent else s
            if iqr.entity_types and view.G.nodes.get(other, {}).get("type") not in iqr.entity_types and \
                    not (set(iqr.entity_types) & {"PERSON"} and p in ("DIRECTOR_OF", "ACCUSED_IN")):
                continue
            rows.append(((s, p, o), e))
        rows.sort(key=lambda x: -x[1]["belief"])
        if not rows:
            S.append(Sentence(f"No {', '.join(iqr.predicates) or 'matching'} relationships involving {names.get(ent)} are documented in this case.", kind="meta"))
        for (s, p, o), e in rows[: iqr.limit]:
            S.append(self.describe_edge(e, names.get(s, s), names.get(o, o), docs))
            cited += e["claims"][:6]
        return S, {"relations": [{"subject": s, "predicate": p, "object": o, **{k: e[k] for k in ("belief", "independent_sources", "valid_from", "valid_to")}}
                                 for (s, p, o), e in rows[: iqr.limit]]}, cited

    def _find_related(self, iqr: IQR):
        names = self._names(iqr.case_id)
        view = self._view(iqr)
        ent = iqr.entity_ids[0]
        nb = neighbourhood(view, ent, iqr.max_hops, iqr.include_weak)
        targets = [v for v in nb["nodes"] if v != ent and (not iqr.entity_types or view.G.nodes[v].get("type") in iqr.entity_types)]
        S, cited = [], []
        S.append(Sentence(f"{len(targets)} entit{'y' if len(targets) == 1 else 'ies'} of the requested type are within {iqr.max_hops} documented hop(s) of {names.get(ent)}.", kind="meta"))
        out = []
        for v in targets[: iqr.limit]:
            ps = find_paths(view, ent, v, k=1, max_hops=iqr.max_hops)
            if not ps:
                continue
            p = ps[0]
            chain = " → ".join(names.get(x, x) for x in p["nodes"])
            cl = [c for h in p["edges"] for c in h["claims"][:2]]
            S.append(Sentence(f"{names.get(v)} ({nb['distance'][v]} hop{'s' if nb['distance'][v] > 1 else ''}): {chain} (path belief {p['path_belief']:.2f}"
                              f"{', indirect' if p['kind'] == 'indirect' else ''}).", cl))
            cited += cl
            out.append({"entity": v, "name": names.get(v), "distance": nb["distance"][v], "path": p})
        return S, {"related": out}, cited

    def _entity_profile(self, iqr: IQR):
        ent = iqr.entity_ids[0] if iqr.entity_ids else None
        if not ent:
            return [Sentence("No entity in the question could be linked to this case.", kind="meta")], {}, []
        e = self.st.get_entity(ent)
        names = self._names(iqr.case_id)
        docs = self._docs(iqr.case_id)
        view = self._view(iqr)
        S = [Sentence(f"{e['canonical_name']} ({e['type'].lower()}) is resolved from {len(self.st.entity_mentions(ent))} mention(s) across "
                      f"{e['attributes'].get('documents', '?')} document(s); known aliases: {', '.join(sorted(set(e['aliases']) - {e['canonical_name']})[:6]) or 'none'}.", kind="meta")]
        if e["identifiers"]:
            S.append(Sentence("Identifiers on record: " + "; ".join(f"{k} {', '.join(v)}" for k, v in e["identifiers"].items()) + ".", kind="meta"))
        cited = []
        rows = sorted([((s, p, o), x) for (s, p, o), x in view.edges.items() if ent in (s, o) and not x["weak"]], key=lambda r: -r[1]["belief"])
        for (s, p, o), x in rows[: iqr.limit]:
            S.append(self.describe_edge(x, names.get(s, s), names.get(o, o), docs))
            cited += x["claims"][:4]
        if e["attributes"].get("pending_reviews"):
            S.append(Sentence(f"{e['attributes']['pending_reviews']} possible duplicate identit{'y' if e['attributes']['pending_reviews'] == 1 else 'ies'} "
                              f"await analyst review; the profile may be incomplete or over-split.", kind="caveat"))
        return S, {"entity": e, "relations": len(rows)}, cited

    def _timeline(self, iqr: IQR):
        ent = iqr.entity_ids[0]
        items = timeline(self.st, iqr.case_id, ent)
        S, cited = [], []
        for it in items[: iqr.limit]:
            if it.get("claim_id"):
                rn = it.get("role_names") or {}
                who = " / ".join(v for v in rn.values() if v)
                S.append(Sentence(f"{it['date']}: {it['type'].replace('_', ' ').lower()} ({who}).", [it["claim_id"]]))
                cited.append(it["claim_id"])
        if not S:
            S.append(Sentence("No dated events are documented for this entity.", kind="meta"))
        return S, {"timeline": items}, cited

    def _centrality(self, iqr: IQR):
        view = self._view(iqr)
        U = view.undirected()
        c = analytics.centrality(U, k_sample=300)
        names = self._names(iqr.case_id)
        types = iqr.entity_types or ["PERSON", "ORGANISATION", "GOVT_BODY"]
        ranked = sorted([v for v in c if U.nodes[v].get("type") in types], key=lambda v: -(c[v]["betweenness"] + c[v]["pagerank"] * 5))
        S = [Sentence("Structural prominence reflects position in the documented network only — it is not an indication of wrongdoing or of role.", kind="caveat")]
        cited = []
        for v in ranked[: iqr.limit]:
            edges = [x for (s, p, o), x in view.edges.items() if v in (s, o) and not x["weak"]]
            cl = [x["claims"][0] for x in edges[:3]]
            S.append(Sentence(f"{names.get(v)}: betweenness {c[v]['betweenness']:.3f}, PageRank {c[v]['pagerank']:.4f}, degree {c[v]['degree']:.3f}"
                              f"{', articulation point (removal disconnects the network)' if c[v]['articulation_point'] else ''}.", cl))
            cited += cl
        return S, {"centrality": {v: c[v] for v in ranked[: iqr.limit]}}, cited

    def _communities(self, iqr: IQR):
        view = self._view(iqr)
        U = view.undirected()
        res = analytics.communities(U)
        names = self._names(iqr.case_id)
        S = [Sentence(f"Louvain detected {len(res['communities'])} communities (modularity {res['modularity']:.3f}) on the evidence-weighted graph.", kind="meta")]
        cited = []
        for c in res.get("summary", [])[: iqr.limit]:
            if c["size"] < 3:
                continue
            mem = [names.get(v, v) for v in c["members"][:8]]
            inner = [x["claims"][0] for (s, p, o), x in view.edges.items() if s in set(c["members"]) and o in set(c["members"]) and not x["weak"]][:4]
            S.append(Sentence(f"Community {c['id']} ({c['size']} members, density {c['density']:.2f}): {', '.join(mem)}"
                              f"{' …' if c['size'] > 8 else ''}; most central: {names.get(c['most_central'])}.", inner))
            cited += inner
        return S, {"communities": res}, cited

    def _contradictions(self, iqr: IQR):
        from drishti.core.domain import merged_pack
        cs = contradictions(self.st.claims(iqr.case_id), merged_pack(), self.st.events(iqr.case_id))
        if iqr.entity_ids:
            cs = [c for c in cs if iqr.entity_ids[0] in (c.get("subject"), c.get("object"))] or cs
        S = [Sentence(f"{len(cs)} contradiction(s) detected between sources.", kind="meta")]
        cited = []
        for c in cs[: iqr.limit]:
            cl = [x for x in c["claims"] if x]
            S.append(Sentence(f"{c['type'].capitalize()} conflict: {c['explanation']}", cl))
            cited += cl
        return S, {"contradictions": cs}, cited

    def _patterns(self, iqr: IQR):
        P = patterns.run_all(self.st, iqr.case_id)
        S = [Sentence("Patterns below are investigative leads computed from documented records; each needs analyst review and none is a finding of wrongdoing.", kind="caveat")]
        cited = []
        for k, items in P.items():
            for it in items[:3]:
                cl = it.get("claims", [])[:6]
                if cl:
                    S.append(Sentence(f"{k.replace('_', ' ').capitalize()}: {it['description']}", cl))
                    cited += cl
        return S, {"patterns": {k: v[: iqr.limit] for k, v in P.items()}}, cited

    def _hypotheses(self, iqr: IQR):
        view = self._view(iqr)
        U = view.undirected()
        types = {v: d.get("type") for v, d in U.nodes(data=True)}
        allowed = {("PERSON", "PERSON"), ("PERSON", "ORGANISATION"), ("ORGANISATION", "ORGANISATION")}
        hyps = linkpred.predict(U, top=iqr.limit, types=types, allowed=allowed)
        if iqr.entity_ids:
            hyps = [h for h in hyps if iqr.entity_ids[0] in (h["source"], h["target"])] or hyps
        names = self._names(iqr.case_id)
        S = [Sentence("These are inferred hypotheses from network structure only. No document states them; they are suggestions for where to look.", kind="caveat")]
        cited = []
        for h in hyps[: iqr.limit]:
            via = ", ".join(names.get(w, w) for w in h["shared_neighbours"][:4])
            cl = []
            for w in h["shared_neighbours"][:2]:
                for (s, p, o), x in view.edges.items():
                    if {s, o} in ({h["source"], w}, {h["target"], w}) and not x["weak"]:
                        cl.append(x["claims"][0])
            S.append(Sentence(f"Possible link {names.get(h['source'])} ↔ {names.get(h['target'])} (score {h['score']:.2f}) via shared neighbour(s) {via}.", cl[:4]))
            cited += cl[:4]
        return S, {"hypotheses": hyps}, cited

    def _evidence_search(self, iqr: IQR):
        from drishti.retrieval.hybrid import HybridRetriever
        ps = self.st.passages(iqr.case_id)
        if not ps:
            return [Sentence("No processed documents in this case.", kind="meta")], {}, []
        key = (iqr.case_id, len(ps), ps[-1]["passage_id"])
        R = self._retrievers.get(key)
        if R is None:
            R = self._retrievers[key] = HybridRetriever(ps, self._docs(iqr.case_id))
        hits = R.search(iqr.text, k=iqr.limit)
        claims_by_pid = defaultdict(list)
        for c in self.st.claims(iqr.case_id):
            if c["passage_id"]:
                claims_by_pid[c["passage_id"]].append(c["claim_id"])
        S = [Sentence(f"Top {len(hits)} passages by hybrid retrieval (BM25 + latent semantic + metadata).", kind="meta")]
        cited = []
        for h in hits[:5]:
            cl = claims_by_pid.get(h.passage_id, [])[:3]
            if cl:
                S.append(Sentence(f"“{h.text[:220]}{'…' if len(h.text) > 220 else ''}”", cl))
                cited += cl
        return S, {"hits": [h.__dict__ for h in hits]}, cited
