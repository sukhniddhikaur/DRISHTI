"""Idempotent, cached, resumable processing pipeline.

Per-document stages (parallel, process pool):
    parse  ->  ner
Case-level stages (sequential, need the whole case):
    gazetteer propagation -> relation/claim extraction -> entity resolution

Every stage is keyed by  sha256(input hash | stage | stage version | config).
A cached result is reused byte-for-byte; a document that failed at
`extract` never repeats `parse`/`ner`. Each run writes a `stage_runs` row
(status, attempts, error class, duration) and an audit entry.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote

from drishti.core.domain import DomainPack, merged_pack
from drishti.core.models import Passage, utcnow
from drishti.extract.calibrate import Calibrator
from drishti.extract.extractor import DocumentExtractor
from drishti.extract.ner import NER, Span, build_case_gazetteer
from drishti.extract.relations import RelationExtractor
from drishti.resolve.engine import ERConfig, Resolver, cooccurrence_context
from drishti.store.audit import AuditLog
from drishti.store.db import Store
from drishti.understand.parser import ParsedDocument, parse
from drishti.understand.quality import assess, classify

PARSE_VERSION = "parse-2.0"
TRANSIENT = (OSError, TimeoutError, ConnectionError)


def _key(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _pd_to_json(pd: ParsedDocument) -> dict[str, Any]:
    return {"text": pd.text, "passages": [p.model_dump() for p in pd.passages], "pages": pd.pages, "tables": pd.tables,
            "page_methods": pd.page_methods, "ocr_confidence": pd.ocr_confidence, "warnings": pd.warnings}


def _pd_from_json(d: dict[str, Any]) -> ParsedDocument:
    return ParsedDocument(d["text"], [Passage(**p) for p in d["passages"]], d["pages"], d["tables"],
                          {int(k): v for k, v in d["page_methods"].items()}, {int(k): v for k, v in d["ocr_confidence"].items()},
                          d["warnings"])


def _read_blob(uri: str) -> bytes:
    return Path(unquote(urlparse(uri).path)).read_bytes()


def _parse_and_ner(args: tuple[str, str, str, str]) -> dict[str, Any]:
    """Top-level (picklable) worker: bytes -> parsed doc + first-pass spans."""
    uri, doc_id, mime, title = args
    t0 = time.perf_counter()
    pd = parse(_read_blob(uri), doc_id, mime, title)
    t1 = time.perf_counter()
    spans = NER().extract(pd.text)
    t2 = time.perf_counter()
    return {"parsed": _pd_to_json(pd), "spans": [asdict(s) for s in spans], "ms_parse": (t1 - t0) * 1e3, "ms_ner": (t2 - t1) * 1e3}


class Pipeline:
    def __init__(self, store: Store, pack: DomainPack | None = None, rel_mode: str = "rules", model_dir: str | Path | None = None,
                 workers: int = 4, use_processes: bool = True, cache: bool = True, er_config: ERConfig | None = None):
        self.store, self.pack = store, pack or merged_pack()
        self.model_dir = Path(model_dir) if model_dir else None
        md = self.model_dir
        rel_file = "relation_hybrid.pkl" if rel_mode == "hybrid" else "relation_lr.pkl"
        self.rel = RelationExtractor(self.pack, rel_mode, md / rel_file if md else None)
        self.cal = Calibrator.load(md / "calibration.pkl" if md else None)
        er_cfg = er_config or ERConfig(model_path=str(md / "er_lr.pkl") if md else None)
        self.resolver = Resolver(er_cfg)
        self.workers, self.use_processes, self.cache = workers, use_processes, cache
        self.audit = AuditLog(store)
        self.metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    def _cache_get(self, key: str) -> Any | None:
        if not self.cache:
            return None
        r = self.store.one("SELECT payload FROM stage_cache WHERE cache_key=?", (key,))
        if not r:
            return None
        return r["payload"] if isinstance(r["payload"], dict) else json.loads(r["payload"])

    def _cache_put(self, key: str, payload: Any) -> None:
        with self.store.tx() as c:
            c.execute("INSERT OR REPLACE INTO stage_cache VALUES (?,?,?)", (key, json.dumps(payload, ensure_ascii=False), utcnow()))

    def _stage(self, doc_id: str, stage: str, version: str, key: str, status: str, attempts: int, ms: float,
               err: BaseException | None = None) -> None:
        with self.store.tx() as c:
            c.execute("INSERT OR REPLACE INTO stage_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (doc_id, stage, version, key, status, attempts, type(err).__name__ if err else None,
                       ("".join(traceback.format_exception_only(type(err), err)).strip()[:500] if err else None), ms, utcnow()))

    # ------------------------------------------------------------------ per-document
    def _first_pass(self, docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        todo = []
        for d in docs:
            key = _key(d["file_hash"], "parse+ner", PARSE_VERSION, NER.VERSION)
            hit = self._cache_get(key)
            if hit is not None:
                results[d["document_id"]] = hit | {"cached": True, "key": key}
                self._stage(d["document_id"], "parse+ner", PARSE_VERSION, key, "cached", 0, 0.0)
            else:
                todo.append((d, key))
        if todo:
            Ex = ProcessPoolExecutor if self.use_processes and len(todo) > 1 else ThreadPoolExecutor
            with Ex(max_workers=self.workers) as ex:
                futs = {ex.submit(_parse_and_ner, (d["storage_uri"], d["document_id"], d["mime"], d["title"])): (d, key) for d, key in todo}
                for f, (d, key) in futs.items():
                    attempts, err = 1, None
                    while True:
                        try:
                            r = f.result() if attempts == 1 else _parse_and_ner((d["storage_uri"], d["document_id"], d["mime"], d["title"]))
                            break
                        except TRANSIENT as e:  # retry transient failures with backoff
                            err = e
                            if attempts >= 3:
                                r = None
                                break
                            time.sleep(0.2 * 2 ** attempts)
                            attempts += 1
                        except Exception as e:  # permanent: classify and continue with other documents
                            err, r = e, None
                            break
                    if r is None:
                        self._stage(d["document_id"], "parse+ner", PARSE_VERSION, key, "failed", attempts, 0.0, err)
                        self.store.set_doc_status(d["document_id"], "failed")
                        self.audit.append("pipeline", "stage.failed", "document", d["document_id"], d["case_id"],
                                          {"stage": "parse+ner", "error": type(err).__name__})
                        continue
                    self._cache_put(key, r)
                    self._stage(d["document_id"], "parse+ner", PARSE_VERSION, key, "done", attempts, r["ms_parse"] + r["ms_ner"])
                    results[d["document_id"]] = r | {"cached": False, "key": key}
        return results

    # ------------------------------------------------------------------ case-level
    def run(self, case_id: str, actor: str = "pipeline") -> dict[str, Any]:
        t0 = time.perf_counter()
        docs = self.store.documents(case_id)
        first = self._first_pass(docs)
        t1 = time.perf_counter()
        spans_by_doc = {did: [Span(**s) for s in r["spans"]] for did, r in first.items()}
        gaz = build_case_gazetteer(spans_by_doc.values())
        gaz_hash = _key(json.dumps(sorted(gaz.items())))[:16]
        ner = NER(gazetteer=gaz)
        extractor = DocumentExtractor(self.pack, ner, self.rel, self.cal)
        n_claims = rejected = 0
        for d in docs:
            r = first.get(d["document_id"])
            if r is None:
                continue
            ts = time.perf_counter()
            key = _key(r["key"], "extract", extractor.version, gaz_hash, self.pack.name)
            try:
                pd = _pd_from_json(r["parsed"])
                spans = spans_by_doc[d["document_id"]]
                # second NER pass: case-level gazetteer propagation, then overlap resolution
                spans = NER._resolve_overlaps(spans + ner._gazetteer_pass(pd.text, spans))
                NER._attach_identifiers(pd.text, spans)
                from drishti.resolve.normalize import normalize
                for s in spans:
                    s.normalized = s.normalized or normalize(s.label, s.text)
                out = extractor.run(d, pd, spans)
                q = assess(pd)
                dtype, probs = classify(pd, self.pack, d["title"])
                with self.store.tx() as c:
                    c.execute("UPDATE documents SET text=?, quality=?, document_type=CASE WHEN document_type='unknown' THEN ? ELSE document_type END, "
                              "processing_status='done', metadata=json_set(COALESCE(metadata,'{}'),'$.class_probs',json(?)) WHERE document_id=?",
                              (pd.text, json.dumps(q), dtype, json.dumps(probs), d["document_id"]))
                self.store.replace_doc_outputs(d["document_id"], pd.passages, out.mentions, out.claims, out.events)
                n_claims += len(out.claims)
                rejected += out.rejected
                self._stage(d["document_id"], "extract", extractor.version, key, "done", 1, (time.perf_counter() - ts) * 1e3)
            except Exception as e:
                self._stage(d["document_id"], "extract", extractor.version, key, "failed", 1, (time.perf_counter() - ts) * 1e3, e)
                self.store.set_doc_status(d["document_id"], "failed")
                self.audit.append(actor, "stage.failed", "document", d["document_id"], case_id, {"stage": "extract", "error": repr(e)[:200]})
        t2 = time.perf_counter()
        er = self.resolve(case_id, actor)
        t3 = time.perf_counter()
        self.metrics = {"documents": len(docs), "processed": len(first), "cached": sum(1 for r in first.values() if r.get("cached")),
                        "claims": n_claims, "rejected_spans": rejected, "ms_first_pass": round((t1 - t0) * 1e3, 1),
                        "ms_extract": round((t2 - t1) * 1e3, 1), "ms_resolve": round((t3 - t2) * 1e3, 1),
                        "ms_total": round((t3 - t0) * 1e3, 1), "resolver": er,
                        "docs_per_minute": round(len(docs) / max(1e-9, t3 - t0) * 60, 1)}
        self.audit.append(actor, "pipeline.run", "case", case_id, case_id,
                          {k: v for k, v in self.metrics.items() if k != "resolver"} | {"extractor": extractor.version})
        return self.metrics

    def resolve(self, case_id: str, actor: str = "pipeline") -> dict[str, Any]:
        mentions = self.store.mentions(case_id)
        analyst = [d for d in self.store.decisions(case_id) if d["outcome"] in ("analyst_link", "analyst_split")]
        ents, mapping, decisions = self.resolver.resolve(case_id, mentions, cooccurrence_context(mentions), analyst)
        keep = {d["decision_id"] for d in analyst}
        with self.store.tx() as c:
            c.execute("DELETE FROM resolution_decisions WHERE case_id=? AND outcome NOT IN ('analyst_link','analyst_split')", (case_id,))
            for d in decisions:
                if d["decision_id"] not in keep:
                    self.store._upsert("resolution_decisions", d, c)
        self.store.save_entities(ents, mapping)
        self.audit.append(actor, "resolve.run", "case", case_id, case_id, dict(self.resolver.stats))
        return dict(self.resolver.stats)

    # ------------------------------------------------------------------ analyst actions
    def analyst_resolution(self, case_id: str, left_mention: str, right_mention: str, link: bool, actor: str, reason: str) -> None:
        from drishti.core.ids import stable_id
        rec = {"decision_id": stable_id("DEC", case_id, "analyst", left_mention, right_mention), "case_id": case_id,
               "left_id": left_mention, "right_id": right_mention, "score": 1.0 if link else 0.0, "features": {},
               "outcome": "analyst_link" if link else "analyst_split", "actor": actor, "reason": reason, "created_at": utcnow()}
        self.store.upsert_many("resolution_decisions", [rec])
        self.audit.append(actor, "resolve.analyst_link" if link else "resolve.analyst_split", "mention_pair",
                          f"{left_mention}|{right_mention}", case_id, {"reason": reason})
        self.resolve(case_id, actor)
