"""Schema-constrained relation extraction: rules, a learned classifier, or both.

Candidate generation   ordered mention pairs inside one sentence whose types
                       satisfy *some* relation's signature (closed schema).
Rule layer             trigger lexicon with direction (active / passive /
                       by-passive / right-context), coordination handling and
                       intervening-argument checks.
Learned layer          multinomial logistic regression over sparse lexical +
                       structural features; labels are direction-aware
                       ("DIRECTOR_OF>" = left mention is subject).
Constrained decoding   a predicted label is only accepted if the argument
                       types satisfy the schema; otherwise it is discarded.
Calibration            raw scores -> probabilities with isotonic regression
                       fitted on a held-out dev split (see calibrate.py).
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drishti.core.domain import DomainPack, RelationSpec
from drishti.extract.modality import detect, ended
from drishti.extract.ner import Span
from drishti.core.models import Modality

NODE_LABELS = {"PERSON", "ORGANISATION", "GOVT_BODY", "LOCATION", "CASE", "CONTRACT", "PHONE", "ACCOUNT", "VEHICLE"}
ACTORS = {"PERSON", "ORGANISATION", "GOVT_BODY"}

# right-context rules: (predicate, between regex, right-context regex, swap)  swap => right mention is subject
HON = r"(?:(?:mr|ms|mrs|smt|shri|sri|dr|kum)\.?\s*)?"
RIGHT_RULES = [
    ("DIRECTOR_OF", r"\b(?:appointed|inducted|re-appointed|elected)\s*" + HON + "$", r"^\s*(?:as|to the post of)\s+(?:an?\s+)?(?:additional |independent |whole-time |managing |non-executive |executive )?director", True),
    ("AWARDED_CONTRACT_TO", r"\b(?:awarded|issued|granted)\s*" + HON + "$", r"^\s*(?:the|a)\s+(?:contract|work order|tender|letter of award)", False),
    ("ASSOCIATE_OF", r"\band\s*$", r"^\s*(?:are|were)\s+(?:known\s+)?associates", False),
]


@dataclass
class RelHit:
    predicate: str
    subj: Span
    obj: Span
    s0: int
    s1: int
    raw_score: float
    extractor: str
    trigger: str = ""
    trigger_pos: int = 0
    modality: Modality = Modality.ASSERTED
    cues: list[str] = field(default_factory=list)
    ended: bool = False


def _types_ok(spec: RelationSpec, subj: Span, obj: Span) -> bool:
    return subj.label in spec.subject_types and obj.label in spec.object_types


def _find_trigger(hay: str, triggers: tuple[str, ...]) -> tuple[str, int] | None:
    best = None
    for t in triggers:
        for m in re.finditer(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", hay):
            if best is None or m.start() > best[1]:  # closest to the right mention
                best = (t, m.start())
    return best


class RelationExtractor:
    VERSION = "rel-2.0"

    def __init__(self, pack: DomainPack, mode: str = "rules", model_path: str | Path | None = None,
                 emit_cooccurrence: bool = True):
        self.pack, self.mode, self.emit_cooc = pack, mode, emit_cooccurrence
        self.add_threshold = 0.6
        self.model = None
        if mode in ("learned", "hybrid") and model_path and Path(model_path).exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        if mode in ("learned", "hybrid") and self.model is None:
            self.mode = "rules"
        self._sig = {(s, o) for spec in pack.relations.values() if spec.name != "MENTIONED_WITH"
                     for s in spec.subject_types for o in spec.object_types}

    # ------------------------------------------------------------------ candidates
    def candidates(self, text: str, spans: list[Span], sents: list[tuple[int, int]]) -> list[tuple[Span, Span, int, int]]:
        out = []
        nodes = [s for s in spans if s.label in NODE_LABELS and not s.attrs.get("kin_reference")]
        for s0, s1 in sents:
            inside = [s for s in nodes if s.start >= s0 and s.end <= s1]
            for i, a in enumerate(inside):
                for b in inside[i + 1:]:
                    if a.normalized == b.normalized and a.label == b.label:
                        continue
                    if ((a.label, b.label) in self._sig or (b.label, a.label) in self._sig
                            or (a.label in ACTORS and b.label in ACTORS)):
                        out.append((a, b, s0, s1))
        return out

    # ------------------------------------------------------------------ rules
    def rule_hits(self, text: str, a: Span, b: Span, s0: int, s1: int, spans: list[Span]) -> list[RelHit]:
        low = text.lower()
        between = low[a.end:b.start]
        if len(between) > 160:
            return []
        hits: list[RelHit] = []
        inner = [s for s in spans if s.start >= a.end and s.end <= b.start and s.label in NODE_LABELS]
        for spec in self.pack.relations.values():
            if spec.name == "MENTIONED_WITH" or not spec.triggers and not spec.passive_triggers:
                continue
            for triggers, forward in ((spec.triggers, True), (spec.passive_triggers, False)):
                if not triggers:
                    continue
                subj, obj = (a, b) if forward else (b, a)
                if not _types_ok(spec, subj, obj) and not (spec.symmetric and _types_ok(spec, obj, subj)):
                    continue
                found = _find_trigger(between, triggers)
                if not found:
                    continue
                trig, pos = found
                tpos = a.end + pos
                # an argument of the right type sitting between trigger and right mention steals the relation
                right_type = obj.label if forward else subj.label
                if any(s.start > tpos and s.label == right_type for s in inner):
                    continue
                score = 0.92 - 0.0015 * len(between)
                # an uncoordinated same-type mention between left arg and trigger => weaker
                left_type = a.label
                for s in inner:
                    if s.end <= tpos and s.label == left_type:
                        gap = low[s.end:tpos]
                        if not re.match(r"^\s*(?:,|and|&)?\s*(?:mr\.?|ms\.?|smt\.?|shri)?\s*", gap) or len(gap) > 60:
                            score -= 0.15
                hits.append(self._finish(text, spec.name, subj, obj, s0, s1, score, "rules", trig, tpos))
        # by-passive: "<trigger> A by B"
        if re.fullmatch(r"\s*,?\s*by\s+(?:the\s+)?", between):
            left = low[max(s0, a.start - 70):a.start]
            for spec in self.pack.relations.values():
                found = _find_trigger(left, spec.triggers)
                if found and _types_ok(spec, b, a):
                    hits.append(self._finish(text, spec.name, b, a, s0, s1, 0.88, "rules:by-passive", found[0],
                                             max(s0, a.start - 70) + found[1]))
        # right-context rules
        right = re.sub(r"^\s*\([^)]{0,60}\)", "", low[b.end:min(s1, b.end + 120)])
        for pred, brx, rrx, swap in RIGHT_RULES:
            if pred not in self.pack.relations:
                continue
            spec = self.pack.relations[pred]
            if re.search(brx, between) and re.search(rrx, right):
                subj, obj = (b, a) if swap else (a, b)
                if _types_ok(spec, subj, obj) or (spec.symmetric and _types_ok(spec, obj, subj)):
                    hits.append(self._finish(text, pred, subj, obj, s0, s1, 0.86, "rules:right-context", pred.lower(), b.end))
        # dedupe by predicate keeping best
        best: dict[tuple[str, int, int], RelHit] = {}
        for h in hits:
            k = (h.predicate, h.subj.start, h.obj.start)
            if k not in best or h.raw_score > best[k].raw_score:
                best[k] = h
        return list(best.values())

    def _finish(self, text: str, pred: str, subj: Span, obj: Span, s0: int, s1: int, score: float, ex: str,
                trig: str, tpos: int) -> RelHit:
        first, second = (subj, obj) if subj.start < obj.start else (obj, subj)
        mod, cues = detect(text, s0, s1, tpos, first.end, second.start)
        if mod == Modality.HEDGED:
            score -= 0.1
        return RelHit(pred, subj, obj, s0, s1, max(0.05, min(0.99, score)), ex, trig, tpos, mod, cues, ended(text, s0, s1))

    # ------------------------------------------------------------------ learned
    @staticmethod
    def features(text: str, a: Span, b: Span, s0: int, s1: int, spans: list[Span], rule_preds: set[str]) -> dict[str, Any]:
        low = text.lower()
        f: dict[str, Any] = {f"types={a.label}>{b.label}": 1}
        # replace inner mentions with type placeholders
        inner = sorted([s for s in spans if s.start >= a.end and s.end <= b.start], key=lambda s: s.start)
        seg, cur = [], a.end
        for s in inner:
            seg.append(low[cur:s.start])
            seg.append(f" <{s.label}> ")
            cur = s.end
        seg.append(low[cur:b.start])
        toks = re.findall(r"<[A-Z_]+>|[a-z]+", "".join(seg))
        for t in toks[:14]:
            f[f"bw={t}"] = 1
        for x, y in zip(toks[:13], toks[1:14]):
            f[f"bb={x}_{y}"] = 1
        if toks:
            f[f"first={toks[0]}"] = 1
            f[f"last={toks[-1]}"] = 1
        for t in re.findall(r"[a-z]+", low[max(s0, a.start - 40):a.start])[-3:]:
            f[f"lw={t}"] = 1
        for t in re.findall(r"[a-z]+", low[b.end:min(s1, b.end + 40)])[:3]:
            f[f"rw={t}"] = 1
        n = len(toks)
        f[f"dist={min(n // 3, 6)}"] = 1
        f[f"inner={min(len(inner), 3)}"] = 1
        for p in rule_preds:
            f[f"rule={p}"] = 1
        return f

    # ------------------------------------------------------------------ main
    def extract(self, text: str, spans: list[Span], sents: list[tuple[int, int]]) -> list[RelHit]:
        out: list[RelHit] = []
        for a, b, s0, s1 in self.candidates(text, spans, sents):
            rules = self.rule_hits(text, a, b, s0, s1, spans)
            hits = rules
            if self.mode in ("learned", "hybrid") and self.model is not None:
                hits = self._learned(text, a, b, s0, s1, spans, rules)
            if hits:
                out += hits
            elif self.emit_cooc and a.label in ACTORS and b.label in ACTORS:
                out.append(RelHit("MENTIONED_WITH", a, b, s0, s1, 0.3, "cooccurrence"))
        return out

    def _learned(self, text: str, a: Span, b: Span, s0: int, s1: int, spans: list[Span], rules: list[RelHit]) -> list[RelHit]:
        """learned: model decides alone. hybrid: rule hits are kept (precision) and the model may
        (a) re-score them and (b) add relations the lexicon missed, if p >= add_threshold (recall)."""
        rule_preds = {f"{h.predicate}{'>' if h.subj is a else '<'}" for h in rules}
        feats = self.features(text, a, b, s0, s1, spans, rule_preds if self.mode == "hybrid" else set())
        vec, clf = self.model["vec"], self.model["clf"]
        probs = dict(zip(clf.classes_, clf.predict_proba(vec.transform([feats]))[0]))
        out: list[RelHit] = []
        if self.mode == "hybrid":
            for h in rules:
                p = probs.get(f"{h.predicate}{'>' if h.subj is a else '<'}", 0.0)
                h.raw_score = round(0.6 * h.raw_score + 0.4 * max(p, 0.5), 4)
                h.extractor = "hybrid"
                out.append(h)
        threshold = self.add_threshold if self.mode == "hybrid" else 0.35
        for lab, p in sorted(probs.items(), key=lambda x: -x[1])[:2]:
            if lab == "NONE" or p < threshold:
                continue
            pred, direction = lab[:-1], lab[-1]
            if pred not in self.pack.relations or any(h.predicate == pred for h in out):
                continue
            spec = self.pack.relations[pred]
            subj, obj = (a, b) if direction == ">" else (b, a)
            if not _types_ok(spec, subj, obj) and not (spec.symmetric and _types_ok(spec, obj, subj)):
                continue  # constrained decoding: schema-violating predictions are discarded
            r = next((h for h in rules if h.predicate == pred and h.subj is subj), None)
            tpos = r.trigger_pos if r else (a.end + b.start) // 2
            h = self._finish(text, pred, subj, obj, s0, s1, float(p), "learned", r.trigger if r else "", tpos)
            h.raw_score = float(p)
            out.append(h)
        return out
