"""Role-based + per-case access control.

Roles define *what* a user may do; the case ACL defines *where*. Every API and
assistant query is scoped through `require()`, so a generated query can never
widen its own scope.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

from drishti.core.models import utcnow
from drishti.store.db import Store

ROLE_PERMS: dict[str, set[str]] = {
    "admin": {"read", "write", "review", "admin", "export"},
    "analyst": {"read", "write", "review", "export"},
    "viewer": {"read"},
}


class AccessDenied(PermissionError):
    pass


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class Security:
    def __init__(self, store: Store):
        self.store = store

    def create_user(self, user_id: str, name: str, role: str, api_key: str | None = None) -> str:
        if role not in ROLE_PERMS:
            raise ValueError(f"unknown role {role}")
        key = api_key or secrets.token_urlsafe(24)
        with self.store.tx() as c:
            c.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?,?)", (user_id, name, role, _hash_key(key), utcnow()))
        return key

    def grant(self, case_id: str, user_id: str, permission: str = "member") -> None:
        with self.store.tx() as c:
            c.execute("INSERT OR REPLACE INTO case_acl VALUES (?,?,?)", (case_id, user_id, permission))

    def authenticate(self, api_key: str) -> dict | None:
        h = _hash_key(api_key)
        for u in self.store.q("SELECT * FROM users"):
            if hmac.compare_digest(u["api_key_hash"], h):
                return u
        return None

    def cases_for(self, user_id: str) -> list[str]:
        u = self.store.one("SELECT * FROM users WHERE user_id=?", (user_id,))
        if u and u["role"] == "admin":
            return [r["case_id"] for r in self.store.q("SELECT case_id FROM cases")]
        return [r["case_id"] for r in self.store.q("SELECT case_id FROM case_acl WHERE user_id=?", (user_id,))]

    def require(self, user_id: str, case_id: str, perm: str = "read") -> None:
        u = self.store.one("SELECT * FROM users WHERE user_id=?", (user_id,))
        if u is None:
            raise AccessDenied("unknown user")
        if perm not in ROLE_PERMS[u["role"]]:
            raise AccessDenied(f"role '{u['role']}' lacks '{perm}'")
        if u["role"] != "admin" and case_id not in self.cases_for(user_id):
            raise AccessDenied(f"user '{user_id}' has no access to case '{case_id}'")
