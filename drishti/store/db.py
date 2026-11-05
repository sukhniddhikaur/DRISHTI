"""Relational source of truth (SQLite here; schema is Postgres-portable).

The graph is *derived* from this store. Every table that carries knowledge
also carries provenance (document, span, extractor version). Writes that change
knowledge or access go through the hash-chained audit log.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from drishti.core.models import Claim, DocumentRecord, Entity, Event, Mention, Passage, utcnow

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS cases (
  case_id TEXT PRIMARY KEY, name TEXT NOT NULL, domain TEXT NOT NULL,
  created_at TEXT NOT NULL, description TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS sources (
  source_id TEXT PRIMARY KEY, name TEXT, kind TEXT, reliability TEXT, access_method TEXT, url TEXT
);
CREATE TABLE IF NOT EXISTS documents (
  document_id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL REFERENCES cases(case_id),
  source_id TEXT NOT NULL, title TEXT, document_type TEXT, mime TEXT,
  publication_date TEXT, retrieval_date TEXT, file_hash TEXT NOT NULL,
  storage_uri TEXT, source_url TEXT, access_method TEXT, reliability TEXT,
  version INTEGER DEFAULT 1, parent_document_id TEXT, processing_status TEXT,
  text TEXT, quality TEXT, metadata TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_doc_case_hash ON documents(case_id, file_hash);
CREATE INDEX IF NOT EXISTS ix_doc_source ON documents(case_id, source_id);

CREATE TABLE IF NOT EXISTS passages (
  passage_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
  page INTEGER, kind TEXT, char_start INTEGER, char_end INTEGER, text TEXT
);
CREATE INDEX IF NOT EXISTS ix_passage_doc ON passages(document_id);

CREATE TABLE IF NOT EXISTS mentions (
  mention_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
  passage_id TEXT, label TEXT, text TEXT, char_start INTEGER, char_end INTEGER,
  normalized TEXT, attributes TEXT, extractor TEXT, extractor_version TEXT, confidence REAL
);
CREATE INDEX IF NOT EXISTS ix_mention_doc ON mentions(document_id);
CREATE INDEX IF NOT EXISTS ix_mention_norm ON mentions(label, normalized);

CREATE TABLE IF NOT EXISTS entities (
  entity_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, type TEXT, canonical_name TEXT,
  aliases TEXT, identifiers TEXT, attributes TEXT, active INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_entity_case ON entities(case_id, type);

-- Reversible resolution: mappings are never destroyed, only superseded.
CREATE TABLE IF NOT EXISTS mention_entity (
  mention_id TEXT NOT NULL, entity_id TEXT NOT NULL, decision_id TEXT,
  active INTEGER DEFAULT 1, created_at TEXT, PRIMARY KEY (mention_id, entity_id)
);
CREATE INDEX IF NOT EXISTS ix_me_entity ON mention_entity(entity_id, active);

CREATE TABLE IF NOT EXISTS resolution_decisions (
  decision_id TEXT PRIMARY KEY, case_id TEXT, left_id TEXT, right_id TEXT,
  score REAL, features TEXT, outcome TEXT, actor TEXT, reason TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_rd_case ON resolution_decisions(case_id, outcome);

CREATE TABLE IF NOT EXISTS claims (
  claim_id TEXT PRIMARY KEY, case_id TEXT NOT NULL, predicate TEXT NOT NULL,
  subject_mention_id TEXT, object_mention_id TEXT, subject_entity_id TEXT, object_entity_id TEXT,
  subject_text TEXT, object_text TEXT, document_id TEXT NOT NULL, passage_id TEXT,
  evidence_text TEXT NOT NULL, char_start INTEGER, char_end INTEGER, page INTEGER,
  modality TEXT, epistemic TEXT, valid_from TEXT, valid_to TEXT, reported_at TEXT, ingested_at TEXT,
  extractor TEXT, extractor_version TEXT, raw_score REAL, confidence REAL, attributes TEXT, status TEXT
);
CREATE INDEX IF NOT EXISTS ix_claim_case ON claims(case_id, predicate);
CREATE INDEX IF NOT EXISTS ix_claim_subj ON claims(subject_entity_id);
CREATE INDEX IF NOT EXISTS ix_claim_obj ON claims(object_entity_id);
CREATE INDEX IF NOT EXISTS ix_claim_doc ON claims(document_id);

CREATE TABLE IF NOT EXISTS claim_history (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, claim_id TEXT, field TEXT, old TEXT, new TEXT, actor TEXT, reason TEXT, at TEXT
);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, case_id TEXT, event_type TEXT, date TEXT, document_id TEXT, passage_id TEXT,
  evidence_text TEXT, char_start INTEGER, char_end INTEGER, roles TEXT, attributes TEXT
);

CREATE TABLE IF NOT EXISTS stage_runs (
  document_id TEXT, stage TEXT, version TEXT, cache_key TEXT, status TEXT, attempts INTEGER,
  error_class TEXT, error TEXT, duration_ms REAL, updated_at TEXT, PRIMARY KEY (document_id, stage)
);
CREATE TABLE IF NOT EXISTS stage_cache (cache_key TEXT PRIMARY KEY, payload TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY, name TEXT, role TEXT, api_key_hash TEXT UNIQUE, created_at TEXT
);
CREATE TABLE IF NOT EXISTS case_acl (case_id TEXT, user_id TEXT, permission TEXT, PRIMARY KEY(case_id, user_id));

CREATE TABLE IF NOT EXISTS audit_log (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, actor TEXT, action TEXT, object_type TEXT,
  object_id TEXT, case_id TEXT, payload TEXT, prev_hash TEXT, hash TEXT
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""

_JSON_COLS = {"quality", "metadata", "attributes", "aliases", "identifiers", "roles", "features", "payload"}


def _enc(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    if hasattr(v, "value"):
        return v.value
    return v


def _row(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    for k in _JSON_COLS & d.keys():
        if isinstance(d[k], str) and d[k]:
            d[k] = json.loads(d[k])
    return d


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---------------------------------------------------------------- basics
    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN")
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def q(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [_row(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.q(sql, params)
        return rows[0] if rows else None

    def _upsert(self, table: str, rec: dict[str, Any], conn: sqlite3.Connection | None = None) -> None:
        cols = list(rec)
        sql = (f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
               f"ON CONFLICT DO UPDATE SET " + ",".join(f"{c}=excluded.{c}" for c in cols))
        (conn or self.conn).execute(sql, [_enc(rec[c]) for c in cols])

    def upsert_many(self, table: str, recs: list[dict[str, Any]]) -> None:
        if not recs:
            return
        with self.tx() as c:
            for r in recs:
                self._upsert(table, r, c)

    # ---------------------------------------------------------------- cases
    def create_case(self, case_id: str, name: str, domain: str, description: str = "") -> None:
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO cases VALUES (?,?,?,?,?)", (case_id, name, domain, utcnow(), description))

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM cases WHERE case_id=?", (case_id,))

    # ---------------------------------------------------------------- documents
    def find_document_by_hash(self, case_id: str, file_hash: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM documents WHERE case_id=? AND file_hash=?", (case_id, file_hash))

    def latest_version(self, case_id: str, source_id: str, title: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM documents WHERE case_id=? AND source_id=? AND title=? ORDER BY version DESC LIMIT 1",
                        (case_id, source_id, title))

    def save_document(self, doc: DocumentRecord) -> None:
        self.upsert_many("documents", [doc.model_dump()])

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM documents WHERE document_id=?", (document_id,))

    def documents(self, case_id: str) -> list[dict[str, Any]]:
        return self.q("SELECT * FROM documents WHERE case_id=? ORDER BY document_id", (case_id,))

    def set_doc_status(self, document_id: str, status: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE documents SET processing_status=? WHERE document_id=?", (status, document_id))

    # ---------------------------------------------------------------- extraction outputs
    def replace_doc_outputs(self, document_id: str, passages: list[Passage], mentions: list[Mention],
                            claims: list[Claim], events: list[Event]) -> None:
        """Idempotent per-document write: same doc + same model version -> same rows."""
        with self.tx() as c:
            c.execute("DELETE FROM passages WHERE document_id=?", (document_id,))
            c.execute("DELETE FROM mentions WHERE document_id=?", (document_id,))
            c.execute("DELETE FROM claims WHERE document_id=? AND epistemic NOT IN ('confirmed','rejected')", (document_id,))
            c.execute("DELETE FROM events WHERE document_id=?", (document_id,))
            for p in passages:
                self._upsert("passages", p.model_dump(), c)
            for m in mentions:
                self._upsert("mentions", m.model_dump(), c)
            for cl in claims:
                # never overwrite an analyst decision
                ex = c.execute("SELECT epistemic FROM claims WHERE claim_id=?", (cl.claim_id,)).fetchone()
                if ex and ex[0] in ("confirmed", "rejected"):
                    continue
                self._upsert("claims", cl.model_dump(), c)
            for e in events:
                self._upsert("events", e.model_dump(), c)

    def passages(self, case_id: str) -> list[dict[str, Any]]:
        return self.q("SELECT p.* FROM passages p JOIN documents d USING(document_id) WHERE d.case_id=? "
                      "ORDER BY p.document_id, p.char_start", (case_id,))

    def mentions(self, case_id: str, labels: Iterable[str] | None = None) -> list[dict[str, Any]]:
        sql = "SELECT m.* FROM mentions m JOIN documents d USING(document_id) WHERE d.case_id=?"
        params: list[Any] = [case_id]
        if labels:
            labels = list(labels)
            sql += f" AND m.label IN ({','.join('?' * len(labels))})"
            params += labels
        return self.q(sql + " ORDER BY m.mention_id", params)

    def claims(self, case_id: str, **filters: Any) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM claims WHERE case_id=?", [case_id]
        for k, v in filters.items():
            if isinstance(v, (list, tuple, set)):
                v = list(v)
                sql += f" AND {k} IN ({','.join('?' * len(v))})"
                params += v
            else:
                sql += f" AND {k}=?"
                params.append(v)
        return self.q(sql + " ORDER BY claim_id", params)

    def get_claim(self, claim_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM claims WHERE claim_id=?", (claim_id,))

    def events(self, case_id: str) -> list[dict[str, Any]]:
        return self.q("SELECT * FROM events WHERE case_id=? ORDER BY date, event_id", (case_id,))

    # ---------------------------------------------------------------- entities
    def save_entities(self, entities: list[Entity], mapping: dict[str, tuple[str, str]]) -> None:
        """mapping: mention_id -> (entity_id, decision_id)."""
        with self.tx() as c:
            case_ids = {e.case_id for e in entities}
            for cid in case_ids:
                c.execute("UPDATE entities SET active=0 WHERE case_id=?", (cid,))
                c.execute("UPDATE mention_entity SET active=0 WHERE entity_id IN (SELECT entity_id FROM entities WHERE case_id=?)", (cid,))
            for e in entities:
                d = e.model_dump()
                d.pop("mention_ids")
                d["active"] = 1
                self._upsert("entities", d, c)
            now = utcnow()
            for mid, (eid, did) in mapping.items():
                self._upsert("mention_entity", {"mention_id": mid, "entity_id": eid, "decision_id": did,
                                                "active": 1, "created_at": now}, c)
            # propagate resolution onto claims (subject/object entity ids)
            for cid in case_ids:
                c.execute("""UPDATE claims SET
                    subject_entity_id=(SELECT entity_id FROM mention_entity me WHERE me.mention_id=claims.subject_mention_id AND me.active=1),
                    object_entity_id=(SELECT entity_id FROM mention_entity me WHERE me.mention_id=claims.object_mention_id AND me.active=1)
                    WHERE case_id=?""", (cid,))

    def entities(self, case_id: str, type_: str | None = None) -> list[dict[str, Any]]:
        if type_:
            return self.q("SELECT * FROM entities WHERE case_id=? AND active=1 AND type=? ORDER BY entity_id", (case_id, type_))
        return self.q("SELECT * FROM entities WHERE case_id=? AND active=1 ORDER BY entity_id", (case_id,))

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        return self.one("SELECT * FROM entities WHERE entity_id=?", (entity_id,))

    def entity_mentions(self, entity_id: str) -> list[dict[str, Any]]:
        return self.q("SELECT m.* FROM mentions m JOIN mention_entity me USING(mention_id) "
                      "WHERE me.entity_id=? AND me.active=1", (entity_id,))

    def decisions(self, case_id: str, outcome: str | None = None) -> list[dict[str, Any]]:
        if outcome:
            return self.q("SELECT * FROM resolution_decisions WHERE case_id=? AND outcome=? ORDER BY score DESC", (case_id, outcome))
        return self.q("SELECT * FROM resolution_decisions WHERE case_id=? ORDER BY score DESC", (case_id,))

    # ---------------------------------------------------------------- claim review
    def update_claim(self, claim_id: str, actor: str, reason: str, **fields: Any) -> None:
        with self.tx() as c:
            old = c.execute("SELECT * FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
            if old is None:
                raise KeyError(claim_id)
            for k, v in fields.items():
                c.execute(f"UPDATE claims SET {k}=? WHERE claim_id=?", (_enc(v), claim_id))
                c.execute("INSERT INTO claim_history(claim_id,field,old,new,actor,reason,at) VALUES (?,?,?,?,?,?,?)",
                          (claim_id, k, str(old[k]), str(_enc(v)), actor, reason, utcnow()))

    def claim_history(self, claim_id: str) -> list[dict[str, Any]]:
        return self.q("SELECT * FROM claim_history WHERE claim_id=? ORDER BY seq", (claim_id,))
