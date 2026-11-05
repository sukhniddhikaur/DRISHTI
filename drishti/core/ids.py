"""Deterministic, content-addressed identifiers.

Deterministic IDs make every stage idempotent: re-running extraction on the
same document with the same model version produces the same mention/claim IDs,
so retries upsert instead of duplicating.
"""
from __future__ import annotations

import hashlib


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_id(prefix: str, *parts: object, n: int = 12) -> str:
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{h[:n].upper()}"
