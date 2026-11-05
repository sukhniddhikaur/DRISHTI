"""Scope-limited negation, hedging and attribution detection.

The cue must sit inside the clause that contains the relation trigger (no
sentence-wide bag-of-words), otherwise "X, who was never charged, is a director
of Y" would wrongly negate the directorship.
"""
from __future__ import annotations

import re

from drishti.core.models import Modality

NEG = re.compile(r"\b(?:not|never|no(?![.\w])|neither|nor|denied|denies|deny|refuted|disputed that|isn't|wasn't|aren't|weren't|"
                 r"has not|had not|did not|does not|cannot|without any)\b|n't\b", re.I)
HEDGE = re.compile(r"\b(?:allegedly|alleged|reportedly|purportedly|suspected|suspect(?:s|ed)? that|believed to|"
                   r"may have|might have|could have|is said to|are said to|rumou?red|presumably|apparently|possibly|"
                   r"likely|unconfirmed|prima facie)\b", re.I)
REPORT = re.compile(r"\b(?:according to|as per|as reported|reported that|sources said|it is learnt|it is learned|"
                    r"claimed that|stated that|said that|told reporters|in a statement)\b", re.I)
CLAUSE_BREAK = re.compile(r"[,;:]\s*(?:who|which|whereas|while|but|although|though)\b|\bbut\b|;", re.I)


def clause_window(text: str, s0: int, s1: int, anchor: int) -> tuple[int, int]:
    """Smallest clause around `anchor` bounded by relative-clause / contrast breaks."""
    left, right = s0, s1
    for m in CLAUSE_BREAK.finditer(text, s0, s1):
        if m.end() <= anchor:
            left = m.end()
        elif m.start() >= anchor:
            right = m.start()
            break
    return left, right


REL_CLAUSE = re.compile(r",\s*(?:who|which|whose|whom|where)\b[^,;]*,", re.I)


def _mask_relative(text: str, s0: int, s1: int) -> str:
    """Blank out embedded relative clauses (", who was never charged,") keeping offsets."""
    if not REL_CLAUSE.search(text, s0, s1):
        return text  # common case: no copy
    seg = REL_CLAUSE.sub(lambda m: " " * len(m.group(0)), text[s0:s1])
    return text[:s0] + seg + text[s1:]


def detect(text: str, s0: int, s1: int, trigger_pos: int, subj_end: int, obj_start: int) -> tuple[Modality, list[str]]:
    text = _mask_relative(text, s0, s1)
    a, b = clause_window(text, s0, s1, trigger_pos)
    # negation must precede the trigger closely (within 30 chars) or sit between the arguments
    lo = max(a, trigger_pos - 30)
    neg_zone = text[lo:max(trigger_pos, obj_start)]
    cues: list[str] = []
    neg = NEG.search(neg_zone)
    if neg and not re.search(r"\bno\.\s*\d|\bnot only\b|\bno longer\b", text[lo:lo + 60], re.I):
        cues.append("neg:" + neg.group(0).lower())
        return Modality.NEGATED, cues
    clause = text[a:b]
    h = HEDGE.search(clause)
    if h:
        cues.append("hedge:" + h.group(0).lower())
        return Modality.HEDGED, cues
    r = REPORT.search(text[s0:s1])
    if r:
        cues.append("report:" + r.group(0).lower())
        return Modality.REPORTED, cues
    return Modality.ASSERTED, cues


def ended(text: str, s0: int, s1: int) -> bool:
    return bool(re.search(r"\b(?:no longer|resigned|ceased|stepped down|vacated|removed as|retired as)\b", text[s0:s1], re.I))
