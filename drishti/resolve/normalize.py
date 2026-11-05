"""Name normalisation, suffix canonicalisation and Indic-aware phonetic keys."""
from __future__ import annotations

import re
import unicodedata

HONORIFICS = {"mr", "mrs", "ms", "smt", "shri", "sri", "dr", "kum", "km", "late", "sh", "kumari", "shrimati", "md", "prof",
              "adv", "sardar", "s"}

ORG_SUFFIXES = [
    (r"\bprivate\s+limited\b|\bpvt\.?\s*ltd\.?|\bpvt\.?\s+limited\b|\bprivate\s+ltd\.?", "pvt ltd"),
    (r"\blimited\b|\bltd\.?(?=\s|$)", "ltd"),
    (r"\bl\.?l\.?p\.?(?=\s|$)", "llp"),
    (r"\bincorporated\b|\binc\.?(?=\s|$)", "inc"),
    (r"\bcorporation\b|\bcorp\.?(?=\s|$)", "corp"),
    (r"&\s*co\.?(?=\s|$)|\band\s+company\b", "co"),
]
# "Infra" ~ "Infrastructure" etc.: common Indian corporate abbreviations
ORG_TOKEN_ALIASES = {"infra": "infrastructure", "infrastructures": "infrastructure", "intl": "international",
                     "inds": "industries", "ind": "industries", "engg": "engineering", "constn": "construction",
                     "constructions": "construction", "mfg": "manufacturing", "tech": "technologies", "technology": "technologies",
                     "svcs": "services", "dev": "developers", "bldrs": "builders", "natl": "national", "pharma": "pharmaceuticals",
                     "enterprise": "enterprises", "trader": "traders", "bros": "brothers", "assoc": "associates"}
ORG_STOP_PREFIX = {"m/s", "ms", "messrs", "the", "m/s."}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def org_parts(name: str) -> tuple[str, str]:
    """Return (core name, canonical legal-form suffix)."""
    s = strip_accents(name).lower().strip()
    suffix = ""
    for rx, canon in ORG_SUFFIXES:
        m = re.search(rx + r"\s*$", s)
        if m:
            suffix, s = canon, s[:m.start()]
            break
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    toks = [t for t in s.split() if t]
    while toks and toks[0] in ORG_STOP_PREFIX | {"m", "s"}:
        toks.pop(0)
    toks = [ORG_TOKEN_ALIASES.get(t, t) for t in toks]
    return " ".join(toks).replace("&", "and"), suffix


def person_core(name: str) -> str:
    s = strip_accents(name).lower()
    s = re.sub(r"[.,]", " ", s)
    toks = [t for t in s.split() if t]
    while toks and toks[0] in HONORIFICS:
        toks.pop(0)
    return " ".join(toks)


def normalize(label: str, text: str) -> str:
    if label in ("ORGANISATION", "GOVT_BODY"):
        core, suf = org_parts(text)
        return f"{core} {suf}".strip() if label == "ORGANISATION" else core
    if label == "PERSON":
        return person_core(text)
    if label == "LOCATION":
        return re.sub(r"\s+", " ", strip_accents(text).lower().replace("district", "")).strip()
    return re.sub(r"\s+", " ", text.strip().lower())


_PHON_RULES = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("ou", "u"), ("ph", "f"), ("sh", "s"), ("th", "t"), ("dh", "d"),
               ("kh", "k"), ("bh", "b"), ("gh", "g"), ("ch", "c"), ("jh", "j"), ("w", "v"), ("z", "j"), ("q", "k"),
               ("ck", "k"), ("x", "ks"), ("y", "i")]


def indic_phonetic(token: str) -> str:
    """Transliteration-robust key: Mohammad/Mohammed/Muhammad -> 'mhmd'."""
    t = re.sub(r"[^a-z]", "", strip_accents(token).lower())
    if not t:
        return ""
    for a, b in _PHON_RULES:
        t = t.replace(a, b)
    t = re.sub(r"(.)\1+", r"\1", t)
    head, rest = t[0], re.sub(r"[aeiouh]", "", t[1:])
    if head in "aeiou":
        head = "a"
    return head + rest


def phonetic_key(name: str) -> str:
    return " ".join(indic_phonetic(t) for t in name.split() if t)


def initials_key(name: str) -> str:
    """'Rajesh Kumar Sharma' -> 'rk sharma'; 'R. K. Sharma' -> 'rk sharma'."""
    toks = person_core(name).split()
    if not toks:
        return ""
    return "".join(t[0] for t in toks[:-1]) + " " + toks[-1]
