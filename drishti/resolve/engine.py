"""Entity resolution: profiles -> blocking -> pair scoring -> constrained clustering.

Key properties
--------------
* Sub-quadratic: candidate pairs come from multi-key blocking + MinHash LSH;
  oversized blocks are dropped as stop-blocks and reported.
* Precision-first: identifier *conflicts* (two different PANs/DINs/CINs) are
  hard cannot-link constraints; auto-links require a high threshold; the
  ambiguous band goes to a human review queue.
* No naive transitivity: clusters merge only if the *average* linkage across
  all cross pairs stays high and no cannot-link is violated — A~B and B~C does
  not drag A and C together when A≁C.
* Reversible: every decision is logged with its features; analyst must-link /
  cannot-link decisions are replayed as constraints on every re-run.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from drishti.core.ids import stable_id
from drishti.core.models import Entity, utcnow
from drishti.resolve.normalize import indic_phonetic, initials_key, org_parts, person_core, phonetic_key
from drishti.resolve.similarity import MinHashLSH, jaro_winkler, ngrams, soft_token_overlap, token_jaccard

EXACT_TYPES = {"PHONE", "ACCOUNT", "VEHICLE", "CASE", "CONTRACT"}
FUZZY_TYPES = {"PERSON", "ORGANISATION"}
SIMPLE_TYPES = {"GOVT_BODY", "LOCATION"}
STRONG_IDS = ("PAN", "DIN", "CIN", "GSTIN", "AADHAAR")
SOFT_IDS = ("PHONE", "EMAIL", "ACCOUNT")
FEATURES = ["jw", "soft_tok", "tok_jacc", "tfidf", "phon_eq", "initials_ok", "suffix_eq", "suffix_conflict",
            "strong_id_match", "soft_id_match", "ctx_jacc", "loc_overlap", "same_doc", "len_ratio", "first_tok_eq",
            "last_tok_eq"]


@dataclass
class Profile:
    pid: str
    label: str
    names: Counter = field(default_factory=Counter)      # surface forms
    norms: Counter = field(default_factory=Counter)      # normalised forms
    ids: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    context: set[str] = field(default_factory=set)       # co-mentioned normalised names
    locations: set[str] = field(default_factory=set)
    docs: set[str] = field(default_factory=set)
    mention_ids: list[str] = field(default_factory=list)
    aliases: set[str] = field(default_factory=set)
    holder_type: str | None = None

    @property
    def norm(self) -> str:
        return self.norms.most_common(1)[0][0]


@dataclass
class ERConfig:
    t_high: float = 0.85
    t_low: float = 0.45
    avg_link_factor: float = 0.8
    max_block: int = 150
    model_path: str | None = None


class Resolver:
    VERSION = "er-2.0"

    def __init__(self, cfg: ERConfig | None = None):
        self.cfg = cfg or ERConfig()
        self.model = None
        if self.cfg.model_path and Path(self.cfg.model_path).exists():
            with open(self.cfg.model_path, "rb") as f:
                self.model = pickle.load(f)
        self.stats: dict[str, Any] = {}

    # ================================================================== profiles
    @staticmethod
    def build_profiles(mentions: list[dict[str, Any]], cooc: dict[str, set[str]]) -> list[Profile]:
        """One profile per (doc, label, normalised form): one-sense-per-discourse."""
        groups: dict[tuple, Profile] = {}
        for m in mentions:
            key = (m["document_id"], m["label"], m["normalized"] or m["text"].lower())
            p = groups.get(key)
            if p is None:
                p = groups[key] = Profile(pid=stable_id("PRF", *key), label=m["label"])
            p.names[m["text"]] += 1
            p.norms[key[2]] += 1
            p.docs.add(m["document_id"])
            p.mention_ids.append(m["mention_id"])
            attrs = m.get("attributes") or {}
            for t, vals in (attrs.get("identifiers") or {}).items():
                p.ids[t].update(vals)
            for al in attrs.get("aliases", []):
                p.aliases.add(al)
            if attrs.get("pan_holder_type"):
                p.holder_type = attrs["pan_holder_type"]
            p.context |= cooc.get(m["mention_id"], set())
        for p in groups.values():
            p.context.discard(f"{p.label}:{p.norm}")
            p.locations = {c for c in p.context if c.startswith("LOCATION:")}
        return list(groups.values())

    # ================================================================== blocking
    def block(self, profiles: list[Profile]) -> set[tuple[int, int]]:
        keys: dict[str, list[int]] = defaultdict(list)
        lsh = MinHashLSH()
        for i, p in enumerate(profiles):
            ks = set()
            for t in STRONG_IDS + SOFT_IDS:
                for v in p.ids.get(t, ()):
                    ks.add(f"id:{t}:{v}")
            for n in list(p.norms) + [a.lower() for a in p.aliases]:
                if p.label == "ORGANISATION":
                    core, _ = org_parts(n)
                    toks = core.split()
                    if toks:
                        ks.add(f"o1:{toks[0]}")
                        ks.add(f"o4:{core[:4]}")
                        ks.add("osort:" + " ".join(sorted(toks)))
                else:
                    toks = person_core(n).split()
                    if toks:
                        ks.add(f"sur:{indic_phonetic(toks[-1])}")
                        ks.add(f"ini:{initials_key(n)}")
                        ks.add(f"fn:{indic_phonetic(toks[0])}:{toks[-1][:1]}")
                for bi, bk in lsh.bands(lsh.signature(ngrams(n))):
                    ks.add(f"lsh:{bi}:{bk.hex()}")
            for k in ks:
                keys[f"{p.label}|{k}"].append(i)
        pairs: set[tuple[int, int]] = set()
        stop = 0
        for k, idx in keys.items():
            idx = sorted(set(idx))
            if len(idx) > self.cfg.max_block and not k.split("|", 1)[1].startswith("id:"):
                stop += 1
                continue
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    pairs.add((idx[x], idx[y]))
        n = len(profiles)
        self.stats.update(profiles=n, candidate_pairs=len(pairs), naive_pairs=n * (n - 1) // 2, stop_blocks=stop,
                          reduction_ratio=round(1 - len(pairs) / max(1, n * (n - 1) // 2), 4))
        return pairs

    # ================================================================== features
    def _tfidf(self, profiles: list[Profile]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        docs = [" ".join(p.norms) for p in profiles]
        if not docs:
            return None
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        try:
            return vec.fit_transform(docs)
        except ValueError:
            return None

    @staticmethod
    def conflict(a: Profile, b: Profile) -> str | None:
        if a.label != b.label:
            return "type"
        for t in ("PAN", "DIN", "CIN", "AADHAAR"):
            if a.ids.get(t) and b.ids.get(t) and not (a.ids[t] & b.ids[t]):
                return f"id_conflict:{t}"
        if a.holder_type and b.holder_type and a.holder_type != b.holder_type:
            return "pan_holder_type"
        if a.label == "ORGANISATION":
            sa, sb = org_parts(a.norm)[1], org_parts(b.norm)[1]
            if sa and sb and sa != sb and {sa, sb} != {"pvt ltd", "ltd"}:
                return "legal_form"
        if a.docs == b.docs and len(a.docs) == 1 and a.label == "PERSON":
            # two different full names in the same document are usually different people
            ta, tb = person_core(a.norm).split(), person_core(b.norm).split()
            if len(ta) >= 2 and len(tb) >= 2 and ta[0] != tb[0] and not (ta[0][:1] == tb[0][:1] and min(len(ta[0]), len(tb[0])) <= 2):
                return "same_doc_distinct"
        return None

    def features(self, a: Profile, b: Profile, X=None, i: int = 0, j: int = 0) -> dict[str, float]:
        if a.label == "ORGANISATION":
            na = [org_parts(n)[0] for n in a.norms] + [x.lower() for x in a.aliases]
            nb = [org_parts(n)[0] for n in b.norms] + [x.lower() for x in b.aliases]
        else:
            na = [person_core(n) for n in a.norms] + [person_core(x) for x in a.aliases]
            nb = [person_core(n) for n in b.norms] + [person_core(x) for x in b.aliases]
        na, nb = [x for x in na if x] or [a.norm], [x for x in nb if x] or [b.norm]
        jw = max(jaro_winkler(x, y) for x in na for y in nb)
        st = max(soft_token_overlap(x, y) for x in na for y in nb)
        tj = max(token_jaccard(x, y) for x in na for y in nb)
        tf = 0.0
        if X is not None:
            tf = float((X[i] @ X[j].T).toarray()[0, 0])
        ph = float(any(phonetic_key(x) == phonetic_key(y) for x in na for y in nb))
        ini = 0.0
        if a.label == "PERSON":
            for x in na:
                for y in nb:
                    tx, ty = x.split(), y.split()
                    if tx and ty and indic_phonetic(tx[-1]) == indic_phonetic(ty[-1]):
                        fx, fy = [t[0] for t in tx[:-1]], [t[0] for t in ty[:-1]]
                        if fx and fy and (fx[:len(fy)] == fy or fy[:len(fx)] == fx):
                            ini = 1.0
        sa, sb = org_parts(a.norm)[1], org_parts(b.norm)[1]
        sid = sum(len(a.ids.get(t, set()) & b.ids.get(t, set())) for t in STRONG_IDS)
        soid = sum(len(a.ids.get(t, set()) & b.ids.get(t, set())) for t in SOFT_IDS)
        ctx = len(a.context & b.context) / len(a.context | b.context) if a.context | b.context else 0.0
        la, lb = len(na[0]), len(nb[0])
        f0a, f0b = na[0].split(), nb[0].split()
        return {
            "jw": jw, "soft_tok": st, "tok_jacc": tj, "tfidf": tf, "phon_eq": ph, "initials_ok": ini,
            "suffix_eq": float(bool(sa) and sa == sb), "suffix_conflict": float(bool(sa and sb and sa != sb)),
            "strong_id_match": float(min(sid, 2)), "soft_id_match": float(min(soid, 2)),
            "ctx_jacc": ctx, "loc_overlap": float(bool(a.locations & b.locations)), "same_doc": float(bool(a.docs & b.docs)),
            "len_ratio": min(la, lb) / max(la, lb, 1),
            "first_tok_eq": float(bool(f0a and f0b and f0a[0] == f0b[0])),
            "last_tok_eq": float(bool(f0a and f0b and f0a[-1] == f0b[-1])),
        }

    def score(self, f: dict[str, float]) -> float:
        if self.model is not None:
            x = np.array([[f[k] for k in FEATURES]])
            return float(self.model.predict_proba(x)[0, 1])
        # blueprint weighted model: S = w1*name + w2*identifier + w3*address + w4*context (+ structure terms)
        name = 0.4 * f["jw"] + 0.35 * f["soft_tok"] + 0.25 * f["tfidf"]
        if f["initials_ok"] and f["last_tok_eq"]:
            name = max(name, 0.88)
        if f["tok_jacc"] < 0.5 and not f["initials_ok"]:
            name -= 0.2 * (1 - f["tok_jacc"])   # e.g. same brand, different business word
        s = name + 0.08 * f["ctx_jacc"] + 0.04 * min(1.0, f["soft_id_match"]) + 0.02 * f["loc_overlap"] \
            + 0.02 * f["phon_eq"] - 0.25 * f["suffix_conflict"]
        if f["strong_id_match"]:
            s = max(s, 0.97)
        return float(max(0.0, min(1.0, s)))

    # ================================================================== clustering
    def resolve(self, case_id: str, mentions: list[dict[str, Any]], cooc: dict[str, set[str]],
                analyst: Iterable[dict[str, Any]] = ()) -> tuple[list[Entity], dict[str, tuple[str, str]], list[dict[str, Any]]]:
        """Returns (entities, mention->(entity, decision) mapping, decision log rows)."""
        decisions: list[dict[str, Any]] = []
        mapping: dict[str, tuple[str, str]] = {}
        entities: list[Entity] = []
        now = utcnow()
        must: set[frozenset] = set()
        cannot: set[frozenset] = set()
        for d in analyst:
            pair = frozenset((d["left_id"], d["right_id"]))
            (must if d["outcome"] == "analyst_link" else cannot).add(pair)

        # ---------------- exact-value types
        by_val: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for m in mentions:
            if m["label"] in EXACT_TYPES | SIMPLE_TYPES:
                by_val[(m["label"], m["normalized"] or m["text"].lower())].append(m)
        for (lab, val), ms in by_val.items():
            eid = stable_id("ENT", case_id, lab, val)
            did = stable_id("DEC", eid, "exact")
            entities.append(Entity(entity_id=eid, case_id=case_id, type=lab, canonical_name=Counter(m["text"] for m in ms).most_common(1)[0][0] if lab in SIMPLE_TYPES else val,
                                   aliases=sorted({m["text"] for m in ms}), mention_ids=[m["mention_id"] for m in ms],
                                   attributes={"resolution": "exact_value", "documents": len({m['document_id'] for m in ms})}))
            for m in ms:
                mapping[m["mention_id"]] = (eid, did)

        # ---------------- fuzzy types
        fm = [m for m in mentions if m["label"] in FUZZY_TYPES]
        profiles = self.build_profiles(fm, cooc)
        pairs = self.block(profiles)
        X = self._tfidf(profiles)
        scored: dict[tuple[int, int], float] = {}
        feats: dict[tuple[int, int], dict[str, float]] = {}
        blocked: dict[tuple[int, int], str] = {}
        pidx = {p.pid: i for i, p in enumerate(profiles)}
        mention_to_prof = {mid: p.pid for p in profiles for mid in p.mention_ids}

        def prof_pair(i: int, j: int) -> frozenset:
            return frozenset((profiles[i].pid, profiles[j].pid))

        analyst_must = {frozenset(mention_to_prof.get(x, x) for x in pr) for pr in must}
        analyst_cannot = {frozenset(mention_to_prof.get(x, x) for x in pr) for pr in cannot}
        for i, j in pairs:
            a, b = profiles[i], profiles[j]
            c = self.conflict(a, b)
            if c:
                blocked[(i, j)] = c
                continue
            f = self.features(a, b, X, i, j)
            feats[(i, j)] = f
            scored[(i, j)] = self.score(f)
        for pr in analyst_must:
            if len(pr) == 2 and all(x in pidx for x in pr):
                i, j = sorted(pidx[x] for x in pr)
                scored[(i, j)] = 1.0
                blocked.pop((i, j), None)

        parent = list(range(len(profiles)))
        members: dict[int, set[int]] = {i: {i} for i in range(len(profiles))}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def pair_score(i: int, j: int) -> float | None:
            k = (min(i, j), max(i, j))
            if k in blocked:
                return None
            if k not in scored:
                a, b = profiles[k[0]], profiles[k[1]]
                c = self.conflict(a, b)
                if c:
                    blocked[k] = c
                    return None
                feats[k] = self.features(a, b, X, k[0], k[1])
                scored[k] = self.score(feats[k])
            return scored[k]

        review: list[tuple[int, int]] = []
        for (i, j), s in sorted(scored.items(), key=lambda kv: -kv[1]):
            ri, rj = find(i), find(j)
            outcome = "separate"
            reason = ""
            if ri == rj:
                continue
            forced = prof_pair(i, j) in analyst_must
            if prof_pair(i, j) in analyst_cannot:
                outcome, reason = "separate", "analyst cannot-link"
            elif s >= self.cfg.t_high or forced:
                # constrained average-linkage check across the two clusters
                cross = [pair_score(x, y) for x in members[ri] for y in members[rj]]
                if any(c is None for c in cross) and not forced:
                    outcome, reason = "review", "cluster-level conflict"
                elif any(prof_pair(x, y) in analyst_cannot for x in members[ri] for y in members[rj]):
                    outcome, reason = "separate", "analyst cannot-link (cluster)"
                else:
                    vals = [c for c in cross if c is not None]
                    avg = sum(vals) / len(vals) if vals else s
                    if forced or avg >= self.cfg.t_high * self.cfg.avg_link_factor:
                        parent[rj] = ri
                        members[ri] |= members.pop(rj)
                        outcome, reason = ("analyst_link" if forced else "auto_link"), f"avg_link={avg:.3f}"
                    else:
                        outcome, reason = "review", f"chaining prevented avg_link={avg:.3f}"
            elif s >= self.cfg.t_low:
                outcome = "review"
            if outcome == "review":
                review.append((i, j))
            if outcome != "separate" or s >= self.cfg.t_low - 0.1:
                decisions.append({"decision_id": stable_id("DEC", case_id, profiles[i].pid, profiles[j].pid),
                                  "case_id": case_id, "left_id": profiles[i].mention_ids[0], "right_id": profiles[j].mention_ids[0],
                                  "score": round(s, 4), "features": {k: round(v, 4) for k, v in feats.get((i, j), {}).items()},
                                  "outcome": outcome, "actor": "resolver:" + self.VERSION, "reason": reason, "created_at": now})
        for (i, j), why in blocked.items():
            if why.startswith("id_conflict") and jaro_winkler(profiles[i].norm, profiles[j].norm) > 0.85:
                decisions.append({"decision_id": stable_id("DEC", case_id, profiles[i].pid, profiles[j].pid),
                                  "case_id": case_id, "left_id": profiles[i].mention_ids[0], "right_id": profiles[j].mention_ids[0],
                                  "score": 0.0, "features": {}, "outcome": "cannot_link", "actor": "resolver:" + self.VERSION,
                                  "reason": why, "created_at": now})

        for root, mem in members.items():
            ps = [profiles[k] for k in mem]
            names = Counter()
            for p in ps:
                names.update(p.names)
            label = ps[0].label
            canon = self._canonical(label, names)
            mids = sorted(m for p in ps for m in p.mention_ids)
            eid = stable_id("ENT", case_id, label, mids[0])
            ids: dict[str, list[str]] = defaultdict(list)
            for p in ps:
                for t, vs in p.ids.items():
                    ids[t] = sorted(set(ids[t]) | vs)
            pending = sum(1 for (i, j) in review if find(i) == root or find(j) == root)
            entities.append(Entity(entity_id=eid, case_id=case_id, type=label, canonical_name=canon,
                                   aliases=sorted(set(names) | {a for p in ps for a in p.aliases}), identifiers=dict(ids),
                                   mention_ids=mids,
                                   attributes={"resolution": "clustered" if len(ps) > 1 else "singleton", "profiles": len(ps),
                                               "documents": len({d for p in ps for d in p.docs}), "pending_reviews": pending}))
            for p in ps:
                did = stable_id("DEC", eid, "member", p.pid)
                for mid in p.mention_ids:
                    mapping[mid] = (eid, did)
        self.stats.update(entities=len(entities), review_pairs=len(review), blocked_pairs=len(blocked),
                          auto_links=sum(d["outcome"] == "auto_link" for d in decisions))
        return entities, mapping, decisions

    @staticmethod
    def _canonical(label: str, names: Counter) -> str:
        def key(n: str):
            if label == "ORGANISATION":
                core, suf = org_parts(n)
                return (bool(suf), names[n], len(n))
            return (len(person_core(n).split()), names[n], -len(n) + len(person_core(n)))
        best = max(names, key=key)
        return re.sub(r"^(?:Mr|Mrs|Ms|Smt|Shri|Sri|Dr|Kum|Km|Sh)\.?\s+", "", best) if label == "PERSON" else best


def cooccurrence_context(mentions: list[dict[str, Any]], window: int = 250) -> dict[str, set[str]]:
    """Normalised names appearing near each mention (context similarity signal)."""
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for m in mentions:
        by_doc[m["document_id"]].append(m)
    ctx: dict[str, set[str]] = {}
    for ms in by_doc.values():
        ms.sort(key=lambda m: m["char_start"])
        for m in ms:
            ctx[m["mention_id"]] = {f'{o["label"]}:{o["normalized"]}' for o in ms if o is not m and abs(o["char_start"] - m["char_start"]) <= window
                                    and o["label"] in ("ORGANISATION", "PERSON", "GOVT_BODY", "LOCATION", "PHONE")}
    return ctx
