"""Deterministic claims from structured records (CDR, bank statements, registers).

Header synonyms are matched fuzzily so exports from different telecom
operators / banks map onto one schema. Rows become STATED claims whose
evidence is the exact table-row passage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from drishti.extract.identifiers import norm_amount, norm_phone
from drishti.extract.temporal import find_dates

SYN = {
    "a_party": ("calling number", "calling", "a party", "a_party", "msisdn", "caller", "source number", "from number"),
    "b_party": ("called number", "called", "b party", "b_party", "receiver", "destination number", "to number"),
    "call_time": ("start time", "call date", "date time", "datetime", "timestamp", "call start", "date"),
    "duration": ("duration", "dur", "call duration", "seconds"),
    "imei": ("imei",),
    "cell": ("cell id", "cell", "first cell id", "tower", "cgi"),
    "from_acct": ("from account", "debit account", "remitter account", "sender account", "from_account", "source account"),
    "to_acct": ("to account", "credit account", "beneficiary account", "to_account", "receiver account"),
    "from_name": ("remitter name", "sender name", "from name", "account holder", "remitter"),
    "to_name": ("beneficiary name", "receiver name", "to name", "beneficiary"),
    "amount": ("amount", "txn amount", "transaction amount", "amount (inr)", "amount inr", "value"),
    "txn_date": ("txn date", "transaction date", "value date", "date", "posting date"),
    "utr": ("utr", "reference", "ref no", "transaction id", "txn id", "rrn"),
    "mode": ("mode", "channel", "txn type", "type"),
    "phone": ("mobile number", "phone", "mobile", "msisdn", "phone number"),
    "subscriber": ("subscriber name", "subscriber", "customer name", "name"),
    "address": ("address", "subscriber address"),
}


def _map_header(header: list[str]) -> dict[str, int]:
    norm = [re.sub(r"[^a-z0-9 ()_]", "", h.lower()).strip() for h in header]
    out: dict[str, int] = {}
    for key, syns in SYN.items():
        for i, h in enumerate(norm):
            if i in out.values():
                continue
            if h in syns or any(h == s or (len(s) > 4 and s in h) for s in syns):
                out[key] = i
                break
    return out


def table_kind(header: list[str]) -> tuple[str | None, dict[str, int]]:
    m = _map_header(header)
    if {"a_party", "b_party"} <= m.keys():
        return "cdr", m
    if {"from_acct", "to_acct", "amount"} <= m.keys():
        return "bank", m
    if {"phone", "subscriber"} <= m.keys():
        return "subscriber", m
    return None, m


@dataclass
class StructuredRow:
    kind: str
    passage_id: str
    fields: dict[str, Any]


def parse_rows(tables: list[dict[str, Any]]) -> list[StructuredRow]:
    out: list[StructuredRow] = []
    for t in tables:
        kind, m = table_kind(t["header"])
        if not kind:
            continue
        for row, pid in zip(t["rows"], t["passage_ids"]):
            if pid is None:
                continue
            g = {k: (row[i].strip() if i < len(row) else "") for k, i in m.items()}
            f: dict[str, Any] = {"raw": g}
            if kind == "cdr":
                f["a"], f["b"] = norm_phone(g.get("a_party", "")), norm_phone(g.get("b_party", ""))
                ds = find_dates(g.get("call_time", ""))
                f["date"] = ds[0].iso if ds else None
                f["time"] = g.get("call_time", "")
                try:
                    f["duration"] = int(float(g.get("duration", "0") or 0))
                except ValueError:
                    f["duration"] = 0
                f["imei"], f["cell"] = g.get("imei") or None, g.get("cell") or None
                if not (f["a"] and f["b"]):
                    continue
            elif kind == "bank":
                f["from"], f["to"] = re.sub(r"\D", "", g.get("from_acct", "")), re.sub(r"\D", "", g.get("to_acct", ""))
                amt = re.sub(r"[^\d.]", "", g.get("amount", ""))
                f["amount"] = norm_amount(amt, None) if amt else 0
                ds = find_dates(g.get("txn_date", ""))
                f["date"] = ds[0].iso if ds else None
                f["utr"], f["mode"] = g.get("utr") or None, g.get("mode") or None
                f["from_name"], f["to_name"] = g.get("from_name") or None, g.get("to_name") or None
                if not (f["from"] and f["to"]):
                    continue
            elif kind == "subscriber":
                f["phone"], f["name"], f["address"] = norm_phone(g.get("phone", "")), g.get("subscriber"), g.get("address")
                if not f["phone"]:
                    continue
            out.append(StructuredRow(kind, pid, f))
    return out
