"""Parsing-quality assessment and document classification."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from drishti.core.domain import DomainPack
from drishti.understand.parser import ParsedDocument, _garbage_ratio

_WORD = re.compile(r"[a-z]{2,}")
# A small English function-word list: real prose has many; OCR noise has few.
_FUNCTION = set("the of and to in a is was for on by with as at from that this be are an or which has have were its it not".split())


def assess(pd: ParsedDocument) -> dict[str, Any]:
    text = pd.text
    words = _WORD.findall(text.lower())
    fn_ratio = (sum(w in _FUNCTION for w in words) / len(words)) if words else 0.0
    norm = [re.sub(r"\W+", " ", p.text.lower()).strip() for p in pd.passages]
    counts = Counter(n for n in norm if len(n) > 30)
    dup = sum(c - 1 for c in counts.values())
    ocr_pages = [p for p, m in pd.page_methods.items() if m == "ocr"]
    low_ocr = [p for p, c in pd.ocr_confidence.items() if c < 0.75]
    tabular = sum(p.kind == "table_row" for p in pd.passages)
    flags = []
    if not text.strip():
        flags.append("empty")
    if _garbage_ratio(text) > 0.08:
        flags.append("high_garbage")
    if words and fn_ratio < 0.05 and tabular < len(pd.passages) / 2:
        flags.append("low_linguistic_signal")
    if low_ocr:
        flags.append("low_ocr_confidence")
    if dup:
        flags.append("duplicate_sections")
    flags += [f"warning:{w}" for w in pd.warnings]
    return {
        "chars": len(text), "passages": len(pd.passages), "pages": pd.pages, "tables": len(pd.tables),
        "ocr_pages": ocr_pages, "ocr_confidence": pd.ocr_confidence, "garbage_ratio": round(_garbage_ratio(text), 4),
        "function_word_ratio": round(fn_ratio, 4), "duplicate_passages": dup, "flags": flags,
        "needs_review": bool({"empty", "high_garbage", "low_ocr_confidence", "low_linguistic_signal"} & set(flags)),
    }


def classify(pd: ParsedDocument, pack: DomainPack, title: str = "") -> tuple[str, dict[str, float]]:
    """Cue-weighted log-odds classifier over the pack's document classes.

    Each cue contributes log(1 + tf) weighted by an IDF-like specificity
    (cues that appear in many classes count less). Returns label + softmax
    scores so the UI can show *why* a document was typed the way it was.
    """
    low = (title + "\n" + pd.text[:20000]).lower()
    cue_classes = Counter(c for cues in pack.document_classes.values() for c in cues)
    raw: dict[str, float] = {}
    for cls, cues in pack.document_classes.items():
        s = 0.0
        for c in cues:
            tf = low.count(c)
            if tf:
                s += math.log1p(tf) * (1.0 / cue_classes[c])
        raw[cls] = s
    if pd.tables and len(pd.tables[0]["header"]) >= 3:
        hdr = " ".join(pd.tables[0]["header"]).lower()
        if any(k in hdr for k in ("calling", "called", "imei", "b_party", "a_party")):
            raw["cdr"] = raw.get("cdr", 0) + 5
        if any(k in hdr for k in ("utr", "ifsc", "debit", "credit", "from_account", "beneficiary")):
            raw["bank_statement"] = raw.get("bank_statement", 0) + 5
    if not any(raw.values()):
        return "unknown", {}
    m = max(raw.values())
    z = sum(math.exp(v - m) for v in raw.values())
    probs = {k: round(math.exp(v - m) / z, 4) for k, v in raw.items()}
    return max(probs, key=probs.get), probs
