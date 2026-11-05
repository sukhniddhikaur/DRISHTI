"""Date normalisation and valid-time interval extraction.

Distinguishes the three clocks the blueprint requires:
  * valid time   — when the relationship held   (from the sentence)
  * report time  — when the source said it      (document publication date)
  * ingest time  — when DRISHTI learned it      (set by the pipeline)

Indian conventions: numeric dates are DD/MM/YYYY; financial years run
1 April – 31 March ("FY 2023-24").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9

_ORD = r"(?:st|nd|rd|th)?"
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
DATE_RXS = [
    ("dmy_text", re.compile(rf"\b(\d{{1,2}}){_ORD}(?:\s+day\s+of)?[\s-]+{_MON}\.?,?[\s-]+(\d{{4}})\b", re.I)),
    ("mdy_text", re.compile(rf"\b{_MON}\.?\s+(\d{{1,2}}){_ORD},?\s+(\d{{4}})\b", re.I)),
    ("iso", re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b")),
    ("dmy_num", re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")),
    ("my_text", re.compile(rf"\b{_MON}\.?,?\s+(\d{{4}})\b", re.I)),
    ("fy", re.compile(r"\b(?:FY|F\.Y\.|financial year)\s*(\d{4})\s*[-–/]\s*(\d{2,4})\b", re.I)),
]


@dataclass
class DateMatch:
    start: int
    end: int
    text: str
    iso: str
    end_iso: str | None = None  # for ranges such as a financial year
    precision: str = "day"


def _mk(y: int, m: int, d: int) -> str | None:
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def find_dates(text: str, start: int = 0, end: int | None = None) -> list[DateMatch]:
    end = len(text) if end is None else end
    out: list[DateMatch] = []
    for kind, rx in DATE_RXS:
        for m in rx.finditer(text, start, end):
            if any(not (m.end() <= o.start or m.start() >= o.end) for o in out):
                continue
            g = m.groups()
            iso = end_iso = None
            prec = "day"
            if kind == "dmy_text":
                iso = _mk(int(g[2]), MONTHS[g[1].lower()[:3]], int(g[0]))
            elif kind == "mdy_text":
                iso = _mk(int(g[2]), MONTHS[g[0].lower()[:3]], int(g[1]))
            elif kind == "iso":
                iso = _mk(int(g[0]), int(g[1]), int(g[2]))
            elif kind == "dmy_num":
                iso = _mk(int(g[2]), int(g[1]), int(g[0]))
            elif kind == "my_text":
                iso, prec = _mk(int(g[1]), MONTHS[g[0].lower()[:3]], 1), "month"
            elif kind == "fy":
                y1 = int(g[0])
                y2 = int(g[1]) if len(g[1]) == 4 else (y1 // 100) * 100 + int(g[1])
                if y2 == y1 + 1:
                    iso, end_iso, prec = _mk(y1, 4, 1), _mk(y2, 3, 31), "fy"
            if iso:
                out.append(DateMatch(m.start(), m.end(), m.group(0), iso, end_iso, prec))
    return sorted(out, key=lambda d: d.start)


START_CUES = ("appointed", "w.e.f", "with effect from", "since", "from", "joined", "inducted", "incorporated", "began", "commenced")
END_CUES = ("resigned", "ceased", "stepped down", "vacated", "till", "until", "upto", "up to", "removed", "retired", "ended")


def interval_for(sentence: str, dates: list[DateMatch], offset: int) -> tuple[str | None, str | None, str | None]:
    """Return (valid_from, valid_to, point_date) using cue words preceding each date."""
    vf = vt = point = None
    low = sentence.lower()
    for d in dates:
        pre = low[max(0, d.start - offset - 40):d.start - offset]
        if d.end_iso:  # financial-year range
            vf, vt = vf or d.iso, d.end_iso
            continue
        if any(c in pre for c in END_CUES):
            vt = d.iso
        elif any(c in pre for c in START_CUES):
            vf = d.iso
        else:
            point = point or d.iso
    if any(c in low for c in ("resigned", "ceased", "stepped down", "vacated")) and point and not vt:
        vt, point = point, None
    if any(c in low for c in ("appointed", "inducted", "joined")) and point and not vf:
        vf = point
    return vf, vt, point
