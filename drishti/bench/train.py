"""Train the learned components on disjoint synthetic worlds.

    python -m drishti.bench.train --models models

  train world (seed 7, train templates only)  -> relation classifier, ER pair model
  dev world   (seed 11, train templates only) -> isotonic calibration
  test world  (seed 42, all templates)        -> evaluation only (evaluate.py)
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from drishti.bench.evaluate import SYMMETRIC, Alignment, _gold_rel_set, run_system
from drishti.bench.metrics import overlap
from drishti.bench.synth import generate
from drishti.core.domain import merged_pack
from drishti.extract.calibrate import Calibrator
from drishti.extract.ner import Span
from drishti.extract.relations import RelationExtractor
from drishti.resolve.engine import FEATURES, Resolver, cooccurrence_context
from drishti.understand.segment import sentences


def _spans_from_db(st, document_id: str) -> list[Span]:
    return [Span(m["label"], m["char_start"], m["char_end"], m["text"], m["confidence"], m["extractor"], m["normalized"], m["attributes"] or {})
            for m in st.q("SELECT * FROM mentions WHERE document_id=? ORDER BY char_start", (document_id,))]


def relation_dataset(al: Alignment, with_rules: bool) -> tuple[list[dict], list[str]]:
    rex = RelationExtractor(merged_pack(), "rules")
    X, y = [], []
    for name, gd in al.gold["documents"].items():
        if name not in al.maps:
            continue
        d = al.docs[name]
        text = d["text"]
        spans = _spans_from_db(al.st, d["document_id"])
        om = al.maps[name]
        grels = [(r["pred"], om.span(*r["subj_span"]), om.span(*r["obj_span"])) for r in gd["rels"]]
        for a, b, s0, s1 in rex.candidates(text, spans, sentences(text)):
            lab = "NONE"
            for pred, ss, os_ in grels:
                if overlap(ss, (a.start, a.end)) and overlap(os_, (b.start, b.end)):
                    lab = pred + ">"
                elif overlap(ss, (b.start, b.end)) and overlap(os_, (a.start, a.end)):
                    lab = pred + ("<" if pred not in SYMMETRIC else ">")
            rules = rex.rule_hits(text, a, b, s0, s1, spans) if with_rules else []
            rp = {f"{h.predicate}{'>' if h.subj is a else '<'}" for h in rules}
            X.append(RelationExtractor.features(text, a, b, s0, s1, spans, rp))
            y.append(lab)
    return X, y


def train_relation(aligns: list[Alignment], out: Path) -> dict:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    rep = {}
    for fname, with_rules in (("relation_lr.pkl", False), ("relation_hybrid.pkl", True)):
        X, y = [], []
        for al in aligns:
            x_, y_ = relation_dataset(al, with_rules)
            X += x_
            y += y_
        vec = DictVectorizer()
        Xv = vec.fit_transform(X)
        clf = LogisticRegression(C=3.0, max_iter=3000).fit(Xv, y)
        with open(out / fname, "wb") as f:
            pickle.dump({"vec": vec, "clf": clf, "labels": sorted(set(y))}, f)
        rep[fname] = {"examples": len(y), "label_counts": dict(Counter(y)), "features": len(vec.vocabulary_),
                      "train_accuracy": round(float(clf.score(Xv, y)), 4)}
    return rep


def train_er(aligns: list[Alignment], out: Path) -> dict:
    from sklearn.linear_model import LogisticRegression

    X, y = [], []
    for al in aligns:
        x_, y_ = er_dataset(al)
        X += x_
        y += y_
    X, y = np.array(X), np.array(y)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y)
    with open(out / "er_lr.pkl", "wb") as f:
        pickle.dump(clf, f)
    return {"pairs": int(len(y)), "positives": int(y.sum()), "train_accuracy": round(float(clf.score(X, y)), 4),
            "coefficients": {k: round(float(c), 3) for k, c in zip(FEATURES, clf.coef_[0])}}


def er_dataset(al: Alignment) -> tuple[list, list]:
    res = Resolver()
    ms = al.st.mentions(al.case, ["PERSON", "ORGANISATION"])
    profiles = res.build_profiles(ms, cooccurrence_context(al.st.mentions(al.case)))
    pairs = res.block(profiles)
    Xt = res._tfidf(profiles)
    X, y = [], []
    gold_of = []
    for p in profiles:
        g = Counter(al.m2g.get(m) for m in p.mention_ids if al.m2g.get(m)).most_common(1)
        gold_of.append(g[0][0] if g else None)
    for i, j in pairs:
        if gold_of[i] is None or gold_of[j] is None or res.conflict(profiles[i], profiles[j]):
            continue
        f = res.features(profiles[i], profiles[j], Xt, i, j)
        X.append([f[k] for k in FEATURES])
        y.append(int(gold_of[i] == gold_of[j]))
    return X, y


def train_calibration(al: Alignment, out: Path) -> dict:
    gold = _gold_rel_set(al.gold)
    title = {d["document_id"]: t for t, d in al.docs.items()}
    ex, sc, y = [], [], []
    for c in al.st.claims(al.case):
        if c["epistemic"] == "stated" or title.get(c["document_id"]) not in al.gold_spans or c["predicate"] == "MENTIONED_WITH":
            continue
        s, o = al.m2g.get(c["subject_mention_id"], "?"), al.m2g.get(c["object_mention_id"], "?")
        if c["predicate"] in SYMMETRIC:
            s, o = sorted((s, o))
        ex.append(c["extractor"])
        sc.append(c["raw_score"])
        y.append(int((title[c["document_id"]], s, c["predicate"], o) in gold))
    cal = Calibrator()
    rep = cal.fit(ex, sc, y)
    cal.save(out / "calibration.pkl")
    return rep


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--work", default="bench_results/train")
    a = ap.parse_args(argv)
    out, work = Path(a.models), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("relation_lr.pkl", "relation_hybrid.pkl", "er_lr.pkl", "calibration.pkl"):
        (out / f).unlink(missing_ok=True)
    report = {}
    aligns = []
    for seed, noise in ((7, 0.0), (8, 0.0), (9, 0.03)):   # noise augmentation for OCR robustness
        tr = generate(work / f"corpus_train_{seed}", seed, "train", noise=noise)
        st, cid, _ = run_system(tr, work / f"run_train_{seed}", "rules", None)
        aligns.append(Alignment(st, cid, json.loads((tr / "gold.json").read_text())))
    report["relation"] = train_relation(aligns, out)
    report["er"] = train_er(aligns, out)
    dv = generate(work / "corpus_dev", 11, "train")
    st, cid, _ = run_system(dv, work / "run_dev", "hybrid", out)
    al = Alignment(st, cid, json.loads((dv / "gold.json").read_text()))
    report["calibration"] = train_calibration(al, out)
    (out / "training_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (v if k != "relation" else {f: {x: y for x, y in r.items() if x != "label_counts"} for f, r in v.items()})
                      for k, v in report.items()}, indent=1))
    return report


if __name__ == "__main__":
    main()
