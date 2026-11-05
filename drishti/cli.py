"""DRISHTI command line.

  drishti demo                      generate a synthetic case, ingest, process, print a summary
  drishti ingest CASE DIR           ingest a folder (sidecar <file>.meta.json for provenance)
  drishti process CASE              run the pipeline
  drishti ask CASE "question"       grounded answer with citations
  drishti report CASE out.html      export the intelligence report
  drishti serve                     REST API + investigation workspace on :8000
  drishti train / bench             train learned components / run the benchmark
  drishti verify-audit              recompute the audit hash chain
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _svc(db: str, models: str | None):
    from drishti.api.service import Service
    from drishti.store.db import Store
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    return Service(Store(db), model_dir=models, rel_mode="hybrid")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="drishti")
    ap.add_argument("--db", default=os.environ.get("DRISHTI_DB", "drishti_data/drishti.db"))
    ap.add_argument("--models", default=os.environ.get("DRISHTI_MODELS", "models"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo")
    p = sub.add_parser("ingest"); p.add_argument("case"); p.add_argument("dir"); p.add_argument("--reliability", default="user_upload")
    p = sub.add_parser("process"); p.add_argument("case")
    p = sub.add_parser("ask"); p.add_argument("case"); p.add_argument("question")
    p = sub.add_parser("report"); p.add_argument("case"); p.add_argument("out")
    p = sub.add_parser("serve"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
    sub.add_parser("train")
    p = sub.add_parser("bench"); p.add_argument("--out", default="bench_results"); p.add_argument("--skip-perf", action="store_true")
    sub.add_parser("verify-audit")
    a = ap.parse_args(argv)

    if a.cmd == "train":
        from drishti.bench.train import main as train
        train(["--models", a.models])
        return 0
    if a.cmd == "bench":
        from drishti.bench.evaluate import main as bench
        bench(["--out", a.out, "--models", a.models] + (["--skip-perf"] if a.skip_perf else []))
        return 0
    if a.cmd == "serve":
        import uvicorn
        os.environ["DRISHTI_DB"], os.environ["DRISHTI_MODELS"] = a.db, a.models
        from drishti.api.app import create_app
        print("DRISHTI workspace → http://%s:%d   (dev admin API key: drishti-admin-dev-key)" % (a.host, a.port))
        uvicorn.run(create_app(a.db, a.models), host=a.host, port=a.port)
        return 0

    svc = _svc(a.db, a.models)
    if a.cmd == "demo":
        from drishti.bench.synth import generate
        from drishti.ingest.ingestor import DirectoryAdapter, Ingestor
        corpus = generate(Path(a.db).parent / "demo_corpus", 42, "test")
        if not svc.st.get_case("DEMO"):
            svc.create_case("admin", "DEMO", "Demo — fictional procurement & network case", "fusion")
        res = Ingestor(svc.st, svc.blob_root, svc.audit).ingest_all("DEMO", DirectoryAdapter(corpus / "docs"), "admin")
        m = svc.process("admin", "DEMO", workers=4)
        print(json.dumps({"ingested": len(res), **{k: v for k, v in m.items() if k != "resolver"}, "resolver": m["resolver"]}, indent=1))
        orgs = sorted(svc.entities("admin", "DEMO", "ORGANISATION"),
                      key=lambda e: -len(svc.st.claims("DEMO", predicate="DIRECTOR_OF", object_entity_id=e["entity_id"])))
        for q in ["Which entities are most central?", "Show suspicious money patterns", "Are there any contradictions?",
                  f"Who are the directors of {orgs[0]['canonical_name']}?"]:
            r = svc.ask("admin", "DEMO", q)
            print(f"\nQ: {q}\n[{r['iqr']['intent']}] {r['answer'][:900]}")
        out = Path(a.db).parent / "DEMO_report.html"
        out.write_text(svc.report("admin", "DEMO"))
        print(f"\nReport → {out}\nRun `drishti serve` and open http://127.0.0.1:8000 (API key drishti-admin-dev-key).")
        return 0
    if a.cmd == "ingest":
        from drishti.core.models import ReliabilityClass
        from drishti.ingest.ingestor import DirectoryAdapter, Ingestor
        if not svc.st.get_case(a.case):
            svc.create_case("admin", a.case, a.case)
        res = Ingestor(svc.st, svc.blob_root, svc.audit).ingest_all(a.case, DirectoryAdapter(a.dir, reliability=ReliabilityClass(a.reliability)), "admin")
        print(json.dumps([r.__dict__ for r in res], indent=1))
    elif a.cmd == "process":
        print(json.dumps(svc.process("admin", a.case, workers=4), indent=1))
    elif a.cmd == "ask":
        r = svc.ask("admin", a.case, a.question)
        print(r["answer"])
        print("\nEvidence:")
        for e in r["evidence"][:8]:
            print(f"  [{e['claim_id']}] {e['document_title']} p.{e['page']}: “{e['evidence_text'][:160]}”")
    elif a.cmd == "report":
        Path(a.out).write_text(svc.report("admin", a.case))
        print(a.out)
    elif a.cmd == "verify-audit":
        print(json.dumps(svc.audit.verify(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
