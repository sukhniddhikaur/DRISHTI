"""Rule + checksum validated identifier extraction (Indian public-record formats).

Precision matters more than recall here: identifiers are the strongest
entity-resolution signal, so a false identifier can cause a false merge.
Each pattern is followed by a structural validator (and a checksum where the
format defines one). Aadhaar-like numbers are detected with the Verhoeff
checksum *only to redact them* — DRISHTI never stores them in clear.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator

STATE_CODES = set("AN AP AR AS BR CH CG DD DL DN GA GJ HP HR JH JK KA KL LA LD MH ML MN MP MZ NL OD OR PB PY RJ SK TN TR TS UK UP WB".split())

# PAN 4th character encodes the holder type: a free entity-type signal for ER/NER.
PAN_HOLDER = {"P": "PERSON", "C": "ORGANISATION", "H": "HUF", "F": "FIRM", "A": "AOP", "T": "TRUST",
              "B": "BOI", "L": "LOCAL_AUTHORITY", "J": "ARTIFICIAL_JURIDICAL", "G": "GOVT_BODY"}

_GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_checksum(first14: str) -> str:
    """Official GSTIN mod-36 check character."""
    total = 0
    for i, ch in enumerate(first14):
        v = _GST_CHARS.index(ch) * (2 if i % 2 else 1)
        total += v // 36 + v % 36
    return _GST_CHARS[(36 - total % 36) % 36]


def valid_gstin(s: str) -> bool:
    return (bool(re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]", s))
            and 1 <= int(s[:2]) <= 38 and valid_pan(s[2:12]) and gstin_checksum(s[:14]) == s[14])


def valid_pan(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{3}[PCHFATBLJG][A-Z]\d{4}[A-Z]", s))


def valid_cin(s: str) -> bool:
    m = re.fullmatch(r"([LU])(\d{5})([A-Z]{2})(\d{4})([A-Z]{3})(\d{6})", s)
    return bool(m) and m.group(3) in STATE_CODES | {"TG"} and 1850 <= int(m.group(4)) <= 2030 and \
        m.group(5) in {"PTC", "PLC", "GOI", "SGC", "FTC", "GAP", "GAT", "NPL", "ULL", "ULT", "FLC", "OPC"}


def valid_ifsc(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", s))


def valid_vehicle(s: str) -> bool:
    m = re.fullmatch(r"([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{4})", s)
    return bool(m) and m.group(1) in STATE_CODES and int(m.group(4)) > 0


_VD = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
       [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
       [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
       [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_VP = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
       [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
       [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_ok(num: str) -> bool:
    c = 0
    for i, d in enumerate(reversed(num)):
        c = _VD[c][_VP[i % 8][int(d)]]
    return c == 0


def norm_phone(s: str) -> str | None:
    d = re.sub(r"\D", "", s)
    if len(d) == 12 and d.startswith("91"):
        d = d[2:]
    elif len(d) == 11 and d.startswith("0"):
        d = d[1:]
    return "+91" + d if len(d) == 10 and d[0] in "6789" else None


_UNITS = {"lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000, "crore": 10_000_000, "crores": 10_000_000,
          "cr": 10_000_000, "million": 1_000_000, "mn": 1_000_000, "billion": 1_000_000_000, "thousand": 1000, "k": 1000}


def norm_amount(num: str, unit: str | None) -> int:
    v = float(num.replace(",", ""))
    return int(round(v * _UNITS.get((unit or "").lower().rstrip("."), 1)))


@dataclass
class IdMatch:
    label: str          # PAN | GSTIN | CIN | DIN | IFSC | PHONE | VEHICLE | ACCOUNT | EMAIL | MONEY | CASE | CONTRACT | AADHAAR
    start: int
    end: int
    text: str
    normalized: str
    attrs: dict


_PATTERNS: list[tuple[str, re.Pattern, Callable[[re.Match], tuple[str, dict] | None]]] = []


def _p(label: str, rx: str, flags: int = 0):
    def deco(fn):
        _PATTERNS.append((label, re.compile(rx, flags), fn))
        return fn
    return deco


@_p("GSTIN", r"\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
def _gst(m):
    s = m.group(0)
    return (s, {"pan": s[2:12], "state_code": s[:2], "holder_type": PAN_HOLDER.get(s[5])}) if valid_gstin(s) else None


@_p("CIN", r"\b[LU]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b")
def _cin(m):
    s = m.group(0)
    return (s, {"listed": s[0] == "L", "state": s[6:8], "year": int(s[8:12]), "class": s[12:15]}) if valid_cin(s) else None


@_p("PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b")
def _pan(m):
    s = m.group(0)
    return (s, {"holder_type": PAN_HOLDER.get(s[3])}) if valid_pan(s) else None


@_p("DIN", r"\bDIN[:\s.-]*(?:No\.?\s*)?(\d{8})\b", re.I)
def _din(m):
    return m.group(1), {}


@_p("IFSC", r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
def _ifsc(m):
    s = m.group(0)
    return (s, {"bank_code": s[:4]}) if valid_ifsc(s) and not valid_pan(s[:10]) else None


@_p("EMAIL", r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
def _email(m):
    return m.group(0).lower(), {}


@_p("VEHICLE", r"\b([A-Z]{2})[\s-]?(\d{1,2})[\s-]?([A-Z]{1,3})[\s-]?(\d{4})\b")
def _veh(m):
    s = "".join(m.groups())
    return (s, {"state": m.group(1)}) if valid_vehicle(s) else None


@_p("ACCOUNT", r"(?:A/?c|Account|Acct)\.?\s*(?:No\.?|Number|#)?\s*[:\-]?\s*(\d{9,18})\b", re.I)
def _acct(m):
    return m.group(1), {}


@_p("AADHAAR", r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")
def _aadhaar(m):
    d = re.sub(r"\s", "", m.group(0))
    return ("XXXX-XXXX-" + d[-4:], {"redacted": True}) if verhoeff_ok(d) else None


@_p("PHONE", r"(?<![\w/])(?:\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
def _phone(m):
    n = norm_phone(m.group(0))
    return (n, {}) if n else None


@_p("MONEY", r"(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d+)?)\s*(crores?|cr\.?|lakhs?|lacs?|million|mn|billion|thousand)?\b", re.I)
def _money(m):
    v = norm_amount(m.group(1), m.group(2))
    return (str(v), {"amount_inr": v}) if v > 0 else None


@_p("MONEY", r"\b([\d,]+(?:\.\d+)?)\s*(crores?|lakhs?)\s*(?:rupees)?\b", re.I)
def _money2(m):
    v = norm_amount(m.group(1), m.group(2))
    return (str(v), {"amount_inr": v}) if v > 0 else None


@_p("CASE", r"\bFIR\s*(?:No\.?|Number)?\s*[:\-]?\s*(\d{1,5})\s*/\s*(\d{2,4})\b", re.I)
def _fir(m):
    yr = m.group(2) if len(m.group(2)) == 4 else "20" + m.group(2)
    return f"FIR-{int(m.group(1))}/{yr}", {"case_kind": "FIR", "year": int(yr)}


@_p("CASE", r"\b((?:W\.?P\.?|C\.?S\.?|Crl\.?\s?A\.?|CRL\.?M\.?C\.?|O\.?S\.?|Civil Appeal|Writ Petition|Suit|Arb\.?\s?P\.?|CS\s?\(COMM\))"
            r"\s*(?:\([A-Za-z]+\))?\s*No\.?\s*\d{1,6}\s*(?:of|/)\s*\d{4})", re.I)
def _court(m):
    s = re.sub(r"\s+", " ", m.group(1)).strip()
    return re.sub(r"[\s.]", "", s.upper()), {"case_kind": "COURT", "display": s}


@_p("CONTRACT", r"\b(?:Tender|NIT|Work Order|Contract|LoA|Letter of Award)\s*(?:No\.?|ID|Reference)\s*[:\-]?\s*([A-Z0-9][A-Z0-9/_.-]{3,40}[A-Z0-9])", re.I)
def _contract(m):
    return m.group(1).upper(), {}


PRIORITY = {"GSTIN": 10, "CIN": 10, "AADHAAR": 9, "ACCOUNT": 8, "PAN": 8, "IFSC": 8, "DIN": 7, "CASE": 7, "CONTRACT": 7,
            "EMAIL": 6, "VEHICLE": 6, "MONEY": 5, "PHONE": 5}


def find_identifiers(text: str, start: int = 0, end: int | None = None) -> Iterator[IdMatch]:
    end = len(text) if end is None else end
    seg = text[start:end]
    found: list[IdMatch] = []
    for label, rx, fn in _PATTERNS:
        for m in rx.finditer(seg):
            r = fn(m)
            if r is None:
                continue
            norm, attrs = r
            # for DIN/ACCOUNT/FIR the span is the full cue+number, which is the readable evidence
            if label in ("ACCOUNT", "DIN"):  # span = the number itself, not the cue words
                a, b = m.start(1), m.end(1)
            else:
                a, b = m.start(), m.end()
            found.append(IdMatch(label, start + a, start + b, seg[a:b], norm, attrs))
    # resolve overlaps: higher priority, then longer
    found.sort(key=lambda x: (-PRIORITY.get(x.label, 0), -(x.end - x.start), x.start))
    taken: list[IdMatch] = []
    for f in found:
        if all(f.end <= t.start or f.start >= t.end for t in taken):
            taken.append(f)
    yield from sorted(taken, key=lambda x: x.start)
