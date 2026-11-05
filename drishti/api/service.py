"""Application service: the single, permission-checked entry point used by the
REST API, the CLI and the benchmark. Every mutating call is audited."""
from __future__ import annotations

import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Any

from drishti.assistant.engine import Executor
from drishti.assistant.parser import EntityLinker, parse_question
from drishti.core.domain import merged_pack
from drishti.core.models import ReliabilityClass
from drishti.evidence.support import aggregate, contradictions
from drishti.graph import analytics, linkpred, patterns
from drishti.graph.paths import find_paths
from drishti.graph.projection import build
from drishti.graph.temporal import diff, timeline
from drishti.ingest.ingestor import Ingestor, RawItem
from drishti.pipeline.orchestrator import Pipeline
from drishti.retrieval.hybrid import HybridRetriever
from drishti.store.audit import AuditLog
from drishti.store.db import Store
from drishti.store.security import AccessDenied, Security


class Service:
    def __init__(self, store: Store, blob_root: str | Path | None = None, model_dir: str | Path | None = None, rel_mode: str = "rules"):
        self.st = store
        self.sec = Security(store)
        self.audit = AuditLog(store)
        self.blob_root = Path(blob_root or (Path(store.path).parent / "blobs" if store.path != ":memory:" else "/tmp/drishti_blobs"))
        self.model_dir = Path(model_dir) if model_dir and Path(model_dir).exists() else None
        self.rel_mode = rel_mode if self.model_dir else "rules"
        self.executor = Executor(store)
        if not store.one("SELECT 1 AS x FROM users WHERE user_id='admin'"):
            self.sec.create_user("admin", "Administrator", "admin", api_key="drishti-admin-dev-key")

    def _req(self, user: str, case: str, perm: str = "read") -> None:
        self.sec.require(user, case, perm)

    # ------------------------------------------------------------------ cases & ingestion
    def create_case(self, user: str, case_id: str, name: str, domain: str = "fusion", description: str = "") -> dict:
        u = self.st.one("SELECT * FROM users WHERE user_id=?", (user,))
        if not u or u["role"] not in ("admin", "analyst"):
            raise AccessDenied("only analysts/admins create cases")
        self.st.create_case(case_id, name, domain, description)
        self.sec.grant(case_id, user, "owner")
        self.audit.append(user, "case.create", "case", case_id, case_id, {"name": name})
        return self.st.get_case(case_id)

    def cases(self, user: str) -> list[dict]:
        ids = set(self.sec.cases_for(user))
        out = []
        for c in self.st.q("SELECT * FROM cases ORDER BY created_at"):
            if c["case_id"] in ids:
                c["documents"] = self.st.one("SELECT COUNT(*) n FROM documents WHERE case_id=?", (c["case_id"],))["n"]
                c["entities"] = self.st.one("SELECT COUNT(*) n FROM entities WHERE case_id=? AND active=1", (c["case_id"],))["n"]
                c["claims"] = self.st.one("SELECT COUNT(*) n FROM claims WHERE case_id=?", (c["case_id"],))["n"]
                out.append(c)
        return out

    def ingest(self, user: str, case: str, filename: str, data: bytes, reliability: str = "user_upload",
               publication_date: str | None = None, source_id: str = "upload", document_type: str | None = None) -> dict:
        self._req(user, case, "write")
        r = Ingestor(self.st, self.blob_root, self.audit).ingest(case, RawItem(
            data=data, title=filename, source_id=source_id, mime=mimetypes.guess_type(filename)[0],
            reliability=ReliabilityClass(reliability), publication_date=publication_date, document_type=document_type), actor=user)
        return r.__dict__

    def process(self, user: str, case: str, workers: int = 2) -> dict:
        self._req(user, case, "write")
        return Pipeline(self.st, merged_pack(), self.rel_mode, self.model_dir, workers=workers).run(case, actor=user)

    def documents(self, user: str, case: str) -> list[dict]:
        self._req(user, case)
        return [{k: v for k, v in d.items() if k != "text"} for d in self.st.documents(case)]

    def document(self, user: str, document_id: str) -> dict:
        d = self.st.get_document(document_id)
        if not d:
            raise KeyError(document_id)
        self._req(user, d["case_id"])
        d["passages"] = self.st.q("SELECT * FROM passages WHERE document_id=? ORDER BY char_start", (document_id,))
        d["claims"] = self.st.q("SELECT claim_id, predicate, subject_text, object_text, char_start, char_end, modality, confidence "
                                "FROM claims WHERE document_id=? ORDER BY char_start", (document_id,))
        d["stages"] = self.st.q("SELECT * FROM stage_runs WHERE document_id=?", (document_id,))
        return d

    # ------------------------------------------------------------------ entities & claims
    def entities(self, user: str, case: str, type_: str | None = None, q: str | None = None, limit: int = 500) -> list[dict]:
        self._req(user, case)
        es = self.st.entities(case, type_)
        if q:
            ql = q.lower()
            es = [e for e in es if ql in e["canonical_name"].lower() or any(ql in a.lower() for a in e["aliases"] or [])]
        return es[:limit]

    def entity(self, user: str, entity_id: str) -> dict:
        e = self.st.get_entity(entity_id)
        if not e:
            raise KeyError(entity_id)
        self._req(user, e["case_id"])
        docs = {d["document_id"]: d for d in self.st.documents(e["case_id"])}
        claims = [c for c in self.st.claims(e["case_id"]) if entity_id in (c["subject_entity_id"], c["object_entity_id"])]
        names = {x["entity_id"]: x["canonical_name"] for x in self.st.entities(e["case_id"])}
        groups: dict[tuple, list] = {}
        for c in claims:
            groups.setdefault((c["subject_entity_id"], c["predicate"], c["object_entity_id"]), []).append(c)
        rels = []
        for (s, p, o), cs in groups.items():
            agg = aggregate(cs, docs)
            rels.append({"subject": s, "subject_name": names.get(s), "predicate": p, "object": o, "object_name": names.get(o),
                         **{k: agg[k] for k in ("belief", "support", "contra", "independent_sources", "supporting_documents", "origin_groups")},
                         "claims": [{k: c[k] for k in ("claim_id", "evidence_text", "document_id", "page", "char_start", "char_end", "modality",
                                                       "epistemic", "status", "confidence", "valid_from", "valid_to", "reported_at")}
                                    | {"document_title": docs[c["document_id"]]["title"]} for c in cs]})
        rels.sort(key=lambda r: (r["predicate"] == "MENTIONED_WITH", -r["belief"]))
        return {**e, "mentions": self.st.entity_mentions(entity_id), "relations": rels, "timeline": timeline(self.st, e["case_id"], entity_id)}

    def claim(self, user: str, claim_id: str) -> dict:
        c = self.st.get_claim(claim_id)
        if not c:
            raise KeyError(claim_id)
        self._req(user, c["case_id"])
        d = self.st.get_document(c["document_id"])
        txt = d["text"] or ""
        a, b = c["char_start"], c["char_end"]
        return {**c, "document": {k: d[k] for k in ("document_id", "title", "reliability", "publication_date", "file_hash", "source_url", "version")},
                "context_before": txt[max(0, a - 300):a], "context_after": txt[b:b + 300],
                "span_verified": txt[a:b] == c["evidence_text"], "history": self.st.claim_history(claim_id)}

    def review_claim(self, user: str, claim_id: str, decision: str, reason: str = "") -> dict:
        c = self.st.get_claim(claim_id)
        if not c:
            raise KeyError(claim_id)
        self._req(user, c["case_id"], "review")
        if decision not in ("confirmed", "rejected", "disputed", "unreviewed"):
            raise ValueError("decision must be confirmed|rejected|disputed|unreviewed")
        epi = {"confirmed": "confirmed", "rejected": "rejected"}.get(decision, c["epistemic"] if c["epistemic"] not in ("confirmed", "rejected") else "extracted")
        self.st.update_claim(claim_id, user, reason, status=decision, epistemic=epi)
        self.audit.append(user, f"claim.{decision}", "claim", claim_id, c["case_id"], {"reason": reason})
        return self.claim(user, claim_id)

    def review_queue(self, user: str, case: str, limit: int = 100) -> list[dict]:
        self._req(user, case)
        out = []
        for d in self.st.decisions(case, "review")[:limit]:
            l = self.st.one("SELECT m.*, d.title FROM mentions m JOIN documents d USING(document_id) WHERE mention_id=?", (d["left_id"],))
            r = self.st.one("SELECT m.*, d.title FROM mentions m JOIN documents d USING(document_id) WHERE mention_id=?", (d["right_id"],))
            if l and r:
                out.append({**d, "left": {k: l[k] for k in ("mention_id", "text", "label", "title", "attributes")},
                            "right": {k: r[k] for k in ("mention_id", "text", "label", "title", "attributes")}})
        return out

    def resolve_pair(self, user: str, case: str, left: str, right: str, link: bool, reason: str) -> dict:
        self._req(user, case, "review")
        p = Pipeline(self.st, merged_pack(), self.rel_mode, self.model_dir)
        p.analyst_resolution(case, left, right, link, user, reason)
        return {"ok": True, "resolver": p.resolver.stats}

    # ------------------------------------------------------------------ graph & analytics
    def graph(self, user: str, case: str, as_of: str | None = None, include_weak: bool = False, min_belief: float = 0.0,
              types: list[str] | None = None) -> dict:
        self._req(user, case)
        v = build(self.st, case, as_of=as_of, include_weak=include_weak, min_belief=min_belief)
        U = v.undirected(include_weak=include_weak)
        cen = analytics.centrality(U, k_sample=300) if U.number_of_edges() else {}
        comm = analytics.communities(U) if U.number_of_edges() else {"communities": []}
        cid = {n: i for i, c in enumerate(comm["communities"]) for n in c}
        nodes = [{"id": n, "name": d.get("name"), "type": d.get("type"), "community": cid.get(n), **(cen.get(n) or {})}
                 for n, d in v.G.nodes(data=True) if n in U and U.degree(n) > 0 and (not types or d.get("type") in types)]
        keep = {n["id"] for n in nodes}
        edges = [{"id": f"{s}|{p}|{o}", "source": s, "target": o, "predicate": p,
                  **{k: e[k] for k in ("belief", "independent_sources", "supporting_documents", "weak", "valid_from", "valid_to", "count", "modalities", "contra")}}
                 for (s, p, o), e in v.edges.items() if s in keep and o in keep and (include_weak or not e["weak"])]
        return {"nodes": nodes, "edges": edges, "modularity": comm.get("modularity")}

    def paths(self, user: str, case: str, a: str, b: str, k: int = 5, max_hops: int = 4, include_weak: bool = False, as_of: str | None = None) -> list[dict]:
        self._req(user, case)
        v = build(self.st, case, as_of=as_of, include_weak=include_weak)
        return find_paths(v, a, b, k=k, max_hops=min(4, max_hops), include_weak=include_weak,
                          claim_lookup=lambda cid: {k2: self.st.get_claim(cid)[k2] for k2 in ("claim_id", "evidence_text", "document_id", "modality", "confidence")})

    def analytics(self, user: str, case: str) -> dict:
        self._req(user, case)
        v = build(self.st, case, include_weak=False)
        U = v.undirected()
        names = {e["entity_id"]: e["canonical_name"] for e in self.st.entities(case)}
        cen = analytics.centrality(U, k_sample=300)
        top = sorted(cen, key=lambda n: -cen[n]["betweenness"])[:15]
        res = analytics.resilience(U, steps=8) if U.number_of_nodes() > 3 else []
        rnd = analytics.resilience(U, steps=8, strategy="random") if U.number_of_nodes() > 3 else []
        lp = linkpred.train_and_evaluate(U) if U.number_of_edges() > 20 else {"auc_combined": None}
        model = lp.pop("model", None)
        types = {n: d.get("type") for n, d in U.nodes(data=True)}
        hyps = linkpred.predict(U, top=15, types=types, allowed={("PERSON", "PERSON"), ("PERSON", "ORGANISATION"), ("ORGANISATION", "ORGANISATION")}, model=model)
        for h in hyps:
            h["source_name"], h["target_name"] = names.get(h["source"]), names.get(h["target"])
            h["shared_names"] = [names.get(x) for x in h["shared_neighbours"]]
        comm = analytics.communities(U)
        for c in comm.get("summary", []):
            c["member_names"] = [names.get(m) for m in c["members"]]
            c["most_central_name"] = names.get(c["most_central"])
        return {"top_brokers": [{"entity_id": n, "name": names.get(n), **cen[n]} for n in top], "communities": comm.get("summary", []),
                "modularity": comm.get("modularity"), "resilience_targeted": res, "resilience_random": rnd,
                "link_prediction_eval": lp, "hypotheses": hyps, "components": len(analytics.components(U))}

    def patterns(self, user: str, case: str) -> dict:
        self._req(user, case)
        return patterns.run_all(self.st, case)

    def contradictions(self, user: str, case: str) -> list[dict]:
        self._req(user, case)
        return contradictions(self.st.claims(case), merged_pack(), self.st.events(case))

    def timeline(self, user: str, case: str, entity_id: str | None = None) -> list[dict]:
        self._req(user, case)
        return timeline(self.st, case, entity_id)

    def diff(self, user: str, case: str, t1: str, t2: str) -> dict:
        self._req(user, case)
        return diff(self.st, case, t1, t2)

    def search(self, user: str, case: str, q: str, k: int = 10, mode: str = "hybrid") -> list[dict]:
        self._req(user, case)
        docs = {d["document_id"]: d for d in self.st.documents(case)}
        alias = {}
        for e in self.st.entities(case):
            ns = {e["canonical_name"], *(e["aliases"] or [])}
            for n in ns:
                alias[n.lower()] = sorted(ns - {n})[:4]
        R = HybridRetriever(self.st.passages(case), docs, alias_index=alias)
        return [{**h.__dict__, "document_title": docs[h.document_id]["title"]} for h in R.search(q, k, mode=mode)]

    # ------------------------------------------------------------------ assistant
    def ask(self, user: str, case: str, question: str, use_llm: bool = False) -> dict:
        self._req(user, case)
        linker = EntityLinker(self.st.entities(case))
        iqr = parse_question(question, case, linker)
        self._req(user, iqr.case_id)  # the IQR can never widen scope
        ans = self.executor.run(iqr)
        meta = {"used": False}
        if use_llm:
            from drishti.assistant import llm
            if llm.enabled():
                ans, meta = llm.verbalise(ans)
                ans.verification = self.executor.verify(case, ans.sentences)["report"]
        self.audit.append(user, "assistant.ask", "case", case, case, {"q": question[:300], "intent": iqr.intent})
        return ans.to_dict() | {"llm": meta}

    # ------------------------------------------------------------------ audit & reports
    def audit_log(self, user: str, case: str, limit: int = 200) -> dict:
        self._req(user, case)
        return {"entries": self.audit.entries(case, limit), "chain": self.audit.verify()}

    def report(self, user: str, case: str) -> str:
        self._req(user, case, "export")
        from drishti.report.html import render_report
        html = render_report(self, user, case)
        self.audit.append(user, "report.export", "case", case, case, {})
        return html
