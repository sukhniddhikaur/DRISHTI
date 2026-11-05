"""Offset-preserving sentence segmentation tuned for Indian legal/corporate text.

Never re-joins or normalises the underlying string: every returned span indexes
straight into the canonical document text, so evidence can be highlighted at
its exact position.
"""
from __future__ import annotations

import re

ABBREV = {
    "mr", "mrs", "ms", "dr", "smt", "shri", "sri", "km", "kum", "pvt", "no", "nos", "vs", "v", "u/s",
    "sec", "s", "r", "w", "d", "st", "inc", "corp", "dept", "govt", "hon'ble", "ld", "adv", "rs", "inr", "approx",
    "e.g", "i.e", "etc", "viz", "jr", "sr", "a/c", "ph", "mob", "regd", "addl", "asst", "insp", "si", "asi", "const",
}
# legal-form abbreviations end a sentence when a new capitalised word follows ("... Ltd. Mr. X")
SOFT_ABBREV = {"ltd", "inc", "corp", "llp", "co", "limited"}
CONTINUE = {"ltd", "limited", "pvt", "private", "llp", "and", "inc"}
_BOUNDARY = re.compile(r"[.!?;](?=[\"')\]]*\s+(?:[A-Z0-9\"'(\[]|$))|\n{2,}")


def sentences(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    end = len(text) if end is None else end
    spans: list[tuple[int, int]] = []
    s = start
    for m in _BOUNDARY.finditer(text, start, end):
        cut = m.end()
        if m.group(0) == ".":
            # look back at the token before the period
            j = m.start() - 1
            while j >= s and not text[j].isspace() and text[j] not in "(":
                j -= 1
            tok = text[j + 1:m.start()].lower().strip("(")
            nxt = re.match(r"\s*([A-Za-z]+)", text[m.end():m.end() + 20])
            nxt_w = nxt.group(1) if nxt else ""
            if tok in SOFT_ABBREV and nxt_w[:1].isupper() and nxt_w.lower() not in CONTINUE:
                _push(text, s, cut, spans)
                s = cut
                continue
            if "." in tok or tok in ABBREV or (len(tok) == 1 and tok.isalpha()) or re.fullmatch(r"\d+", tok or "x") and _next_is_digit(text, m.end()):
                continue
        if m.group(0) == ";":
            continue  # keep clauses together: relations often span ';'
        _push(text, s, cut, spans)
        s = cut
    _push(text, s, end, spans)
    return spans


def _next_is_digit(text: str, i: int) -> bool:
    while i < len(text) and text[i].isspace():
        i += 1
    return i < len(text) and text[i].isdigit()


def _push(text: str, a: int, b: int, out: list[tuple[int, int]]) -> None:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    if b > a:
        out.append((a, b))


TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'./-]*|[^\sA-Za-z0-9]")


def tokens(text: str, start: int = 0, end: int | None = None) -> list[tuple[str, int, int]]:
    end = len(text) if end is None else end
    return [(m.group(0), m.start(), m.end()) for m in TOKEN.finditer(text, start, end)]
