"""Integration tests over a generated synthetic case: pipeline, graph, assistant, security, audit."""
import sqlite3

import pytest

from drishti.api.service import Service
from drishti.assistant.iqr import IQR
from drishti.graph import patterns
from drishti.graph.paths import find_paths
from drishti.graph.projection import build
from drishti.store.audit import AuditLog
from drishti.store.security import AccessDenied


def test_pipeline_metrics_and_idempotence(processed):
    st, case, m = processed
    assert m["processed"] == m["documents"] >= 30 and m["rejected_spans"] == 0
    n_claims = len(st.claims(case))
    from drishti.core.domain import merged_pack
    from drishti.pipeline.orchestrator import Pipeline
    m2 = Pipeline(st, merged_pack(), workers=2, use_processes=False).run(case)
    assert m2["cached"] == m2["processed"]                 # parse+NER served from cache
    assert len(st.claims(case)) == n_claims                # deterministic ids -> no duplicates


def test_evidence_integrity(processed):
    st, case, _ = processed
    docs = {d["document_id"]: d["text"] for d in st.documents(case)}
    claims = st.claims(case)
    assert claims and all(docs[c["document_id"]][c["char_start"]:c["char_end"]] == c["evidence_text"] for c in claims)


def test_ocr_path_used_for_scanned_pdf(processed):
    st, case, _ = processed
    d = st.one("SELECT * FROM documents WHERE title='scanned_award_notice.pdf'")
    assert d["quality"]["ocr_pages"] == [1] and "awarded the contract" in d["text"]


def test_duplicate_bytes_are_not_reprocessed(processed, corpus, tmp_path):
    from drishti.ingest.ingestor import Ingestor, RawItem
    st, case, _ = processed
    data = (corpus / "docs" / "judgment_0.txt").read_bytes()
    r = Ingestor(st, tmp_path).ingest(case, RawItem(data=data, title="copy.txt", source_id="x"))
    assert r.status == "duplicate"


def test_paths_are_evidence_backed(processed):
    st, case, _ = processed
    v = build(st, case, include_weak=False)
    (s, p, o), e = next(((k, e) for k, e in v.edges.items() if k[1] == "DIRECTOR_OF"))
    paths = find_paths(v, s, o, k=3, max_hops=3)
    assert paths[0]["kind"] == "direct" and paths[0]["edges"][0]["claims"]


def test_planted_patterns_detected(processed, gold):
    st, case, _ = processed
    P = patterns.run_all(st, case)
    acct = {e["canonical_name"]: e["entity_id"] for e in st.entities(case, "ACCOUNT")}
    cyc = {acct[gold["entities"][g]["name"]] for g in gold["planted"]["money_cycle"]}
    assert any(set(c["accounts"]) == cyc for c in P["money_cycles"])
    assert any(c["account"] == acct[gold["entities"][gold["planted"]["structuring"]]["name"]] for c in P["structuring"])
    assert any(c["account"] == acct[gold["entities"][gold["planted"]["pass_through"]]["name"]] for c in P["pass_through"])
    assert any(c["event_date"] == gold["planted"]["call_burst"] for c in P["call_bursts"])


def test_assistant_answers_are_cited(processed):
    st, case, _ = processed
    svc = Service(st)
    org = st.entities(case, "ORGANISATION")[0]["canonical_name"]
    for q in [f"Who are the directors of {org}?", "Show suspicious money patterns", "Are there any contradictions?",
              "Which entities are most central?", f"Tell me about {org}"]:
        r = svc.ask("admin", case, q)
        assert r["verification"]["all_findings_cited"] and r["verification"]["dropped_unsupported"] == 0
        assert all(s["citations"] for s in r["sentences"] if s["kind"] == "finding")


def test_iqr_is_allowlisted():
    with pytest.raises(Exception):
        IQR(intent="drop_table", case_id="T")
    with pytest.raises(Exception):
        IQR(intent="find_paths", case_id="T", max_hops=50)
    with pytest.raises(Exception):
        IQR(intent="relation_lookup", case_id="T", predicates=["DELETE"])


def test_access_control_and_audit_chain(processed):
    st, case, _ = processed
    svc = Service(st)
    svc.sec.create_user("v", "Viewer", "viewer", api_key="k-viewer")
    with pytest.raises(AccessDenied):
        svc.graph("v", case)                      # no case grant
    svc.sec.grant(case, "v")
    assert svc.graph("v", case)["nodes"]
    cid = st.claims(case)[0]["claim_id"]
    with pytest.raises(AccessDenied):
        svc.review_claim("v", cid, "confirmed")   # viewers cannot review
    svc.review_claim("admin", cid, "confirmed", "checked against source")
    assert st.get_claim(cid)["epistemic"] == "confirmed" and st.claim_history(cid)
    audit = AuditLog(st)
    assert audit.verify()["valid"]
    with pytest.raises(sqlite3.DatabaseError):
        st.conn.execute("UPDATE audit_log SET actor='x' WHERE seq=1")


def test_audit_tamper_detected(tmp_path):
    from drishti.store.db import Store
    st = Store(tmp_path / "a.db")
    a = AuditLog(st)
    for i in range(3):
        a.append("u", "act", "obj", str(i))
    st.conn.execute("DROP TRIGGER audit_no_update")          # simulate out-of-band tampering
    st.conn.execute("UPDATE audit_log SET actor='mallory' WHERE seq=2")
    v = a.verify()
    assert not v["valid"] and v["first_bad_seq"] == 2


def test_report_renders(processed):
    st, case, _ = processed
    html = Service(st).report("admin", case)
    assert "<svg" in html and "Limitations" in html and "cited evidence" in html
