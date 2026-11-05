"""Tamper-evident, append-only audit log (chain of custody).

Each entry's hash = SHA-256(prev_hash || canonical JSON of the entry). SQLite
triggers forbid UPDATE/DELETE; `verify()` recomputes the chain so any
out-of-band edit to the file (e.g. with a hex editor) is detected and the first
broken sequence number is reported.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from drishti.core.models import utcnow
from drishti.store.db import Store

GENESIS = "0" * 64


def _digest(prev: str, entry: dict[str, Any]) -> str:
    body = json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256((prev + body).encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, store: Store):
        self.store = store

    def append(self, actor: str, action: str, object_type: str, object_id: str,
               case_id: str | None = None, payload: dict[str, Any] | None = None) -> str:
        with self.store.tx() as c:
            last = c.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
            prev = last[0] if last else GENESIS
            entry = {"at": utcnow(), "actor": actor, "action": action, "object_type": object_type,
                     "object_id": object_id, "case_id": case_id, "payload": payload or {}}
            h = _digest(prev, entry)
            c.execute("INSERT INTO audit_log(at,actor,action,object_type,object_id,case_id,payload,prev_hash,hash) "
                      "VALUES (?,?,?,?,?,?,?,?,?)",
                      (entry["at"], actor, action, object_type, object_id, case_id,
                       json.dumps(entry["payload"], sort_keys=True, ensure_ascii=False), prev, h))
            return h

    def verify(self) -> dict[str, Any]:
        rows = self.store.q("SELECT * FROM audit_log ORDER BY seq")
        prev = GENESIS
        for r in rows:
            entry = {"at": r["at"], "actor": r["actor"], "action": r["action"], "object_type": r["object_type"],
                     "object_id": r["object_id"], "case_id": r["case_id"], "payload": r["payload"] or {}}
            if r["prev_hash"] != prev or _digest(prev, entry) != r["hash"]:
                return {"valid": False, "entries": len(rows), "first_bad_seq": r["seq"]}
            prev = r["hash"]
        return {"valid": True, "entries": len(rows), "head": prev}

    def entries(self, case_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        if case_id:
            return self.store.q("SELECT * FROM audit_log WHERE case_id=? ORDER BY seq DESC LIMIT ?", (case_id, limit))
        return self.store.q("SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,))
