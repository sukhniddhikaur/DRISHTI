"""DRISHTI benchmark: every experiment from the blueprint's evaluation table.

    python -m drishti.bench.evaluate --out bench_results

Runs on a freshly generated *test* world (held-out templates, disjoint names)
and compares DRISHTI against baselines that reproduce the prototype's logic.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from drishti.bench import baselines
from drishti.bench.metrics import OffsetMap, match_spans, mrr, ndcg_at_k, overlap, pairwise_er, prf, recall_at_k
from drishti.bench.synth import generate
from drishti.core.domain import merged_pack
from drishti.evidence.support import aggregate, contradictions
from drishti.extract.calibrate import ece
from drishti.graph import analytics, linkpred, patterns
from drishti.graph.paths import find_paths
from drishti.graph.projection import build
from drishti.ingest.ingestor import DirectoryAdapter, Ingestor
from drishti.pipeline.orchestrator import Pipeline
from drishti.retrieval.hybrid import HybridRetriever
from drishti.store.db import Store

NER_LABELS = {"PERSON", "ORGANISATION", "GOVT_BODY", "LOCATION", "PHONE", "ACCOUNT", "VEHICLE", "CASE"}
SYMMETRIC = {"ASSOCIATE_OF"}


# ============================================================================ system run
def run_system(corpus: Path, work: Path, rel_mode: str = "rules", model_dir: Path | None = None, workers: int = 4,
               processes: bool = True, cache: bool = True, case_id: str = "BENCH") -> tuple[Store, str, dict[str, Any]]:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    st = Store(work / "drishti.db")
    st.create_case(case_id, "Benchmark", "fusion")
    Ingestor(st, work / "blobs").ingest_all(case_id, DirectoryAdapter(corpus / "docs"))
    pipe = Pipeline(st, merged_pack(), rel_mode=rel_mode, model_dir=model_dir, workers=workers, use_processes=processes, cache=cache)
    m = pipe.run(case_id)
    return st, case_id, m


class Alignment:
    """Links system documents/mentions/entities to gold documents/mentions/entities."""

    def __init__(self, st: Store, case_id: str, gold: dict[str, Any]):
        self.st, self.case, self.gold = st, case_id, gold
        self.docs = {d["title"]: d for d in st.documents(case_id)}
        self.maps: dict[str, OffsetMap] = {}
        self.m2g: dict[str, str] = {}      # system mention -> gold gid
        self.gold_spans: dict[str, list[tuple[int, int, str, str]]] = {}
        for name, gd in gold["documents"].items():
            d = self.docs.get(name)
            if not d or gd["flags"].get("structured"):
                continue
            om = OffsetMap(gd["text"], d["text"] or "")
            self.maps[name] = om
            spans = []
            for gm in gd["mentions"]:
                a, b = om.span(gm["start"], gm["end"])
                spans.append((a, b, gm["label"], gm["gid"]))
            self.gold_spans[name] = spans
            for m in st.q("SELECT * FROM mentions WHERE document_id=?", (d["document_id"],)):
                best, bo = None, 0
                for a, b, lab, gid in spans:
                    o = overlap((a, b), (m["char_start"], m["char_end"]))
                    if o > bo and o >= 0.5 * min(b - a, m["char_end"] - m["char_start"]):
                        best, bo = gid, o
                if best:
                    self.m2g[m["mention_id"]] = best
        # structured docs: gold by normalised value / exact name
        by_name = defaultdict(list)
        for gid, e in gold["entities"].items():
            key = e["name"][3:] if e["type"] == "PHONE" else e["name"]
            by_name[(e["type"], key.lower())].append(gid)
        for name, gd in gold["documents"].items():
            d = self.docs.get(name)
            if not d or not gd["flags"].get("structured"):
                continue
            for m in st.q("SELECT * FROM mentions WHERE document_id=?", (d["document_id"],)):
                key = m["normalized"][3:] if m["label"] == "PHONE" else m["text"]
                c = by_name.get((m["label"], key.lower()))
                if c and len(c) == 1:
                    self.m2g[m["mention_id"]] = c[0]
        self.me = {r["mention_id"]: r["entity_id"] for r in st.q(
            "SELECT me.mention_id, me.entity_id FROM mention_entity me JOIN entities e USING(entity_id) WHERE e.case_id=? AND me.active=1", (case_id,))}
        votes = defaultdict(Counter)
        for mid, gid in self.m2g.items():
            if mid in self.me:
                votes[gid][self.me[mid]] += 1
        self.g2e = {g: c.most_common(1)[0][0] for g, c in votes.items()}
        evotes = defaultdict(Counter)
        for mid, gid in self.m2g.items():
            if mid in self.me:
                evotes[self.me[mid]][gid] += 1
        self.e2g = {e: c.most_common(1)[0][0] for e, c in evotes.items()}


# ============================================================================ experiments
def eval_ner(al: Alignment) -> dict[str, Any]:
    res = {}
    for system in ("drishti", "regex_baseline"):
        tot = {"strict": [0, 0, 0], "relaxed": [0, 0, 0]}
        per_label: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for name, gspans in al.gold_spans.items():
            d = al.docs[name]
            gold = [(a, b, lab) for a, b, lab, _ in gspans if lab in NER_LABELS]
            if system == "drishti":
                pred = [(m["char_start"], m["char_end"], m["label"]) for m in al.st.q("SELECT * FROM mentions WHERE document_id=?", (d["document_id"],))
                        if m["label"] in NER_LABELS]
            else:
                pred = [(x["start"], x["end"], x["label"]) for x in baselines.regex_ner(d["text"]) if x["label"] in NER_LABELS]
            for mode in ("strict", "relaxed"):
                tp, fp, fn, per = match_spans(gold, pred, mode == "strict")
                tot[mode] = [tot[mode][0] + tp, tot[mode][1] + fp, tot[mode][2] + fn]
                if mode == "relaxed":
                    for lab, v in per.items():
                        for i, k in enumerate(("tp", "fp", "fn")):
                            per_label[lab][i] += v[k]
        res[system] = {"strict": prf(*tot["strict"]), "relaxed": prf(*tot["relaxed"]),
                       "per_label_relaxed": {k: prf(*v) for k, v in sorted(per_label.items())}}
    return res


def _gold_rel_set(gold: dict[str, Any], with_modality: bool = False) -> set[tuple]:
    out = set()
    for name, gd in gold["documents"].items():
        for r in gd["rels"]:
            s, o = r["subj"], r["obj"]
            if r["pred"] in SYMMETRIC:
                s, o = sorted((s, o))
            out.add((name, s, r["pred"], o) + ((r["modality"],) if with_modality else ()))
    return out


def eval_relations(al: Alignment, label: str) -> dict[str, Any]:
    gold = _gold_rel_set(al.gold)
    gold_mod = {k[:4]: k[4] for k in _gold_rel_set(al.gold, True)}
    pred, pred_mod = set(), {}
    title = {d["document_id"]: t for t, d in al.docs.items()}
    for c in al.st.claims(al.case):
        if c["predicate"] == "MENTIONED_WITH" or c["epistemic"] == "stated":
            continue
        name = title.get(c["document_id"])
        if name not in al.gold_spans:
            continue
        s, o = al.m2g.get(c["subject_mention_id"]), al.m2g.get(c["object_mention_id"])
        if not s or not o:
            s, o = s or f"?{c['subject_mention_id']}", o or f"?{c['object_mention_id']}"
        if c["predicate"] in SYMMETRIC:
            s, o = sorted((s, o))
        k = (name, s, c["predicate"], o)
        pred.add(k)
        pred_mod[k] = c["modality"]
    tp = pred & gold
    per = {}
    for p in sorted({k[2] for k in gold | pred}):
        g = {k for k in gold if k[2] == p}
        q = {k for k in pred if k[2] == p}
        per[p] = prf(len(g & q), len(q - g), len(g - q))
    mod_ok = sum(1 for k in tp if pred_mod.get(k) == gold_mod.get(k))
    mods = Counter(gold_mod[k] for k in tp)
    non_asserted = [k for k in tp if gold_mod.get(k) != "asserted"]
    return {"system": label, "overall": prf(len(tp), len(pred - gold), len(gold - pred)), "per_predicate": per,
            "modality_accuracy": round(mod_ok / max(1, len(tp)), 4),
            "non_asserted_modality_accuracy": round(sum(pred_mod.get(k) == gold_mod[k] for k in non_asserted) / max(1, len(non_asserted)), 4),
            "matched_by_gold_modality": dict(mods)}


def eval_untyped_edges(al: Alignment) -> dict[str, Any]:
    """Prototype co-occurrence edges vs DRISHTI typed claims, judged as untyped document-level links."""
    gold = {(k[0],) + tuple(sorted((k[1], k[3]))) for k in _gold_rel_set(al.gold)}
    by_doc = defaultdict(list)
    for name, spans in al.gold_spans.items():
        did = al.docs[name]["document_id"]
        for m in al.st.q("SELECT * FROM mentions WHERE document_id=?", (did,)):
            g = al.m2g.get(m["mention_id"])
            by_doc[name].append(g or f"?{m['mention_id']}")
    cooc = baselines.cooccurrence_edges(by_doc)
    title = {d["document_id"]: t for t, d in al.docs.items()}
    ours = set()
    for c in al.st.claims(al.case):
        if c["predicate"] == "MENTIONED_WITH" or title.get(c["document_id"]) not in al.gold_spans:
            continue
        s = al.m2g.get(c["subject_mention_id"], f"?{c['subject_mention_id']}")
        o = al.m2g.get(c["object_mention_id"], f"?{c['object_mention_id']}")
        ours.add((title[c["document_id"]],) + tuple(sorted((s, o))))
    return {"cooccurrence_baseline": prf(len(cooc & gold), len(cooc - gold), len(gold - cooc)),
            "drishti_typed_claims": prf(len(ours & gold), len(ours - gold), len(gold - ours))}


def eval_er(al: Alignment) -> dict[str, Any]:
    ms = [m for m in al.st.mentions(al.case, ["PERSON", "ORGANISATION"]) if m["mention_id"] in al.m2g and m["mention_id"] in al.me]
    gold_types = {g: e["type"] for g, e in al.gold["entities"].items()}
    ms = [m for m in ms if gold_types.get(al.m2g[m["mention_id"]]) == m["label"]]
    gold = {m["mention_id"]: al.m2g[m["mention_id"]] for m in ms}
    t0 = time.perf_counter()
    exact = baselines.exact_er(ms)
    t1 = time.perf_counter()
    fuzzy = baselines.naive_fuzzy_er(ms)
    t2 = time.perf_counter()
    ours = {m["mention_id"]: al.me[m["mention_id"]] for m in ms}
    reviews = al.st.decisions(al.case, "review")
    # review-queue precision: how many queued pairs are true matches (useful work for analysts)
    rq = [(d["left_id"], d["right_id"]) for d in reviews if d["left_id"] in al.m2g and d["right_id"] in al.m2g]
    rq_true = sum(al.m2g[a] == al.m2g[b] for a, b in rq)
    return {"mentions_evaluated": len(ms),
            "exact_string": pairwise_er(exact, gold),
            "naive_fuzzy_transitive": pairwise_er(fuzzy, gold) | {"ms": round((t2 - t1) * 1e3, 1), "comparisons": len(ms) * (len(ms) - 1) // 2},
            "drishti": pairwise_er(ours, gold),
            "review_queue": {"pairs": len(reviews), "evaluable": len(rq), "true_match_fraction": round(rq_true / max(1, len(rq)), 4)}}


QTPL = {
    "DIRECTOR_OF": lambda s, o: f"who serves on the board of {o}",
    "AWARDED_CONTRACT_TO": lambda s, o: f"which firm won work from {s}",
    "SUBSIDIARY_OF": lambda s, o: f"parent company of {s}",
    "ACCUSED_IN": lambda s, o: f"persons named in {o}",
    "USES_PHONE": lambda s, o: f"mobile used by {s}",
    "ASSOCIATE_OF": lambda s, o: f"associates of {s}",
    "OWNS_VEHICLE": lambda s, o: f"vehicle belonging to {s}",
    "HOLDS_ACCOUNT": lambda s, o: f"bank details of {s}",
    "LOCATED_IN": lambda s, o: f"where is {s} based",
    "FILED_CASE_AGAINST": lambda s, o: f"litigation between {s} and {o}",
    "ACQUIRED": lambda s, o: f"takeover of {o}",
    "SHAREHOLDER_OF": lambda s, o: f"equity holders of {o}",
}


def eval_retrieval(al: Alignment, seed: int = 7) -> dict[str, Any]:
    passages = [p for p in al.st.passages(al.case) if p["kind"] != "table_row"]
    docs = {d["document_id"]: d for d in al.docs.values()}
    ents = al.gold["entities"]

    def gname(g):
        e = ents[g]
        if e["type"] == "CASE":
            n, y = e["name"][4:].split("/")
            return f"FIR No. {n}/{y}"
        return e["name"]

    q2rel: dict[str, set[str]] = defaultdict(set)
    pos = defaultdict(list)
    for p in passages:
        pos[p["document_id"]].append(p)
    for name, gd in al.gold["documents"].items():
        if name not in al.maps:
            continue
        did = al.docs[name]["document_id"]
        for r in gd["rels"]:
            if r["pred"] not in QTPL:
                continue
            a, _ = al.maps[name].span(*r["sentence"])
            for p in pos[did]:
                if p["char_start"] <= a < p["char_end"] + 1:
                    q2rel[QTPL[r["pred"]](gname(r["subj"]), gname(r["obj"]))].add(p["passage_id"])
                    break
    qs = sorted(q2rel.items())
    random.Random(seed).shuffle(qs)
    dev, test = qs[::2], qs[1::2]
    alias = {}
    for e in al.st.entities(al.case):
        names = {e["canonical_name"]} | set(e["aliases"] or [])
        for n in names:
            alias[n.lower()] = sorted(names - {n})[:4]
    R = HybridRetriever(passages, docs, alias_index=alias)
    tuned = R.tune(dev)

    def run(mode, expand=True, fusion="weighted"):
        R.fusion = fusion
        r5 = n10 = rr = 0.0
        for q, rel in test:
            ranked = [h.passage_id for h in R.search(q, 10, mode=mode, expand=expand)]
            r5 += recall_at_k(ranked, rel, 5)
            n10 += ndcg_at_k(ranked, rel, 10)
            rr += mrr(ranked, rel)
        n = max(1, len(test))
        return {"recall@5": round(r5 / n, 4), "ndcg@10": round(n10 / n, 4), "mrr": round(rr / n, 4)}

    out = {"queries_dev": len(dev), "queries_test": len(test), "passages": len(passages), "tuned_weights": tuned,
           "bm25": run("bm25", False), "semantic_lsa": run("semantic", False), "hybrid_no_expansion": run("hybrid", False),
           "hybrid_rrf": run("hybrid", True, "rrf"), "hybrid_tuned_entity_expansion": run("hybrid", True)}
    return out


def eval_graph(al: Alignment, seed: int = 7) -> dict[str, Any]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    view = build(al.st, al.case, include_weak=False)
    U = view.undirected()
    res: dict[str, Any] = {"nodes": U.number_of_nodes(), "edges": U.number_of_edges()}
    # ---- communities vs gold gangs
    gp = [(g, gang) for gang, mem in al.gold["groups"].items() for g in mem if g in al.g2e and al.g2e[g] in U]
    comm = analytics.communities(U)
    lab = {v: i for i, c in enumerate(comm["communities"]) for v in c}
    y_true = [gang for _, gang in gp]
    y_pred = [lab[al.g2e[g]] for g, _ in gp]
    cc = {v: i for i, c in enumerate(analytics.components(U)) for v in c}
    y_cc = [cc[al.g2e[g]] for g, _ in gp]
    res["communities"] = {"members_evaluated": len(gp), "modularity": comm["modularity"],
                          "louvain_ARI": round(adjusted_rand_score(y_true, y_pred), 4), "louvain_NMI": round(normalized_mutual_info_score(y_true, y_pred), 4),
                          "connected_components_ARI": round(adjusted_rand_score(y_true, y_cc), 4),
                          "prototype_no_clustering_ARI": round(adjusted_rand_score(y_true, [0] * len(y_true)), 4)}
    # ---- link prediction (edge hold-out)
    lp = linkpred.train_and_evaluate(U, seed=seed)
    lp.pop("model", None)
    res["link_prediction"] = lp
    # ---- path queries vs gold graph; compare with co-occurrence graph
    G = nx.Graph()
    for r in al.gold["relations"]:
        G.add_edge(r["subj"], r["obj"])
    present = [g for g in G if g in al.g2e and al.g2e[g] in U]
    rnd = random.Random(seed)
    pos_pairs, neg_pairs = [], []
    lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=6))
    for _ in range(4000):
        a, b = rnd.sample(present, 2)
        d = lengths.get(a, {}).get(b)
        if d in (2, 3) and len(pos_pairs) < 60:
            pos_pairs.append((a, b))
        elif (d is None or d > 5) and len(neg_pairs) < 60:
            neg_pairs.append((a, b))
        if len(pos_pairs) >= 60 and len(neg_pairs) >= 60:
            break
    # co-occurrence graph (prototype): entities in the same document are linked
    C = nx.Graph()
    for name, spans in al.gold_spans.items():
        ids = sorted({g for *_, g in spans})
        C.add_edges_from((ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)))

    def judge_ours(pairs):
        found = prec_hops = tot_hops = 0
        for a, b in pairs:
            ps = find_paths(view, al.g2e[a], al.g2e[b], k=1, max_hops=3)
            if ps:
                found += 1
                for h in ps[0]["edges"]:
                    tot_hops += 1
                    ga, gb = al.e2g.get(h["from"]), al.e2g.get(h["to"])
                    prec_hops += bool(ga and gb and G.has_edge(ga, gb))
        return found, prec_hops, tot_hops

    def judge_cooc(pairs):
        found = prec = tot = 0
        for a, b in pairs:
            if a in C and b in C:
                try:
                    p = nx.shortest_path(C, a, b)
                except nx.NetworkXNoPath:
                    continue
                if len(p) - 1 <= 3:
                    found += 1
                    for u, v in zip(p, p[1:]):
                        tot += 1
                        prec += G.has_edge(u, v)
        return found, prec, tot

    out = {}
    for nm, fn in (("drishti_evidence_graph", judge_ours), ("cooccurrence_graph", judge_cooc)):
        fp_, hp, th = fn(pos_pairs)
        fn_, _, _ = fn(neg_pairs)
        out[nm] = {"connected_pairs_found": round(fp_ / max(1, len(pos_pairs)), 4), "hop_precision": round(hp / max(1, th), 4),
                   "false_connection_rate": round(fn_ / max(1, len(neg_pairs)), 4)}
    res["path_queries"] = out | {"positive_pairs": len(pos_pairs), "negative_pairs": len(neg_pairs)}
    # ---- resilience (targeted vs random)
    inv = U.subgraph([v for v in U if U.nodes[v].get("type") in ("PERSON", "PHONE", "ACCOUNT", "CASE", "VEHICLE")]).copy()
    if inv.number_of_nodes() > 5:
        t = analytics.resilience(inv, steps=5, strategy="betweenness")
        r = analytics.resilience(inv, steps=5, strategy="random")
        res["resilience_5_removals"] = {"targeted_lcc": t[-1]["lcc_fraction"], "random_lcc": r[-1]["lcc_fraction"],
                                        "targeted_efficiency": t[-1]["efficiency_ratio"], "random_efficiency": r[-1]["efficiency_ratio"]}
    return res


def eval_patterns(al: Alignment) -> dict[str, Any]:
    P = patterns.run_all(al.st, al.case)
    pl = al.gold["planted"]
    ents = al.gold["entities"]
    acct = {e["canonical_name"]: e["entity_id"] for e in al.st.entities(al.case, "ACCOUNT")}
    phone = {e["canonical_name"]: e["entity_id"] for e in al.st.entities(al.case, "PHONE")}
    names = lambda gids: {acct.get(ents[g]["name"]) for g in gids}
    det = {}
    cyc = names(pl["money_cycle"])
    det["money_cycle"] = any(set(c["accounts"]) == cyc for c in P["money_cycles"])
    det["structuring"] = any(c["account"] == acct.get(ents[pl["structuring"]]["name"]) for c in P["structuring"])
    det["pass_through"] = any(c["account"] == acct.get(ents[pl["pass_through"]]["name"]) for c in P["pass_through"])
    sp = phone.get(ents[pl["shared_phone"]["phone"]]["name"])
    det["shared_phone"] = any(c.get("identifier") == sp for c in P["shared_identifiers"])
    det["call_burst"] = any(c["event_date"] == pl["call_burst"] for c in P["call_bursts"])
    sup = {al.g2e.get(g) for g in pl["interlock"]["suppliers"]}
    det["director_interlock"] = any(set(c["suppliers"]) == sup for c in P["director_interlocks"])
    return {"planted_detected": det, "recall": round(sum(det.values()) / len(det), 4),
            "findings_per_detector": {k: len(v) for k, v in P.items()}}


def eval_evidence(al: Alignment) -> dict[str, Any]:
    docs = {d["document_id"]: d for d in al.st.documents(al.case)}
    claims = al.st.claims(al.case)
    unsupported = sum(1 for c in claims if (docs[c["document_id"]]["text"] or "")[c["char_start"]:c["char_end"]] != c["evidence_text"])
    title = {d["document_id"]: t for t, d in al.docs.items()}
    # source independence on edges supported by >1 document
    view = build(al.st, al.case, include_weak=False)
    multi = [e for e in view.edges.values() if e["supporting_documents"] > 1 and not e["weak"]]
    synd = [e for k, e in view.edges.items() if any(title.get(d) == "news_contract_a_syndicated.txt"
                                                    for g in e["origin_groups"] for d in g["documents"]) and e["predicate"] == "AWARDED_CONTRACT_TO"]
    con = contradictions(claims, merged_pack(), al.st.events(al.case))
    con_docs = [{title.get(al.st.get_claim(c)["document_id"]) for c in x["claims"] if c and al.st.get_claim(c)} for x in con]
    return {"claims": len(claims), "unsupported_claims": unsupported, "evidence_coverage": round(1 - unsupported / max(1, len(claims)), 4),
            "edges_with_multiple_documents": len(multi),
            "mean_documents_per_multi_edge": round(float(np.mean([e["supporting_documents"] for e in multi])) if multi else 0, 3),
            "mean_independent_origins_per_multi_edge": round(float(np.mean([e["independent_sources"] for e in multi])) if multi else 0, 3),
            "syndicated_award_edge": [{"documents": e["supporting_documents"], "independent_origins": e["independent_sources"],
                                       "groups": [g["reason"] for g in e["origin_groups"]]} for e in synd],
            "contradictions": Counter(x["type"] for x in con),
            "planted_temporal_contradiction_found": any("news_profile.txt" in d for d, x in zip(con_docs, con) if x["type"] == "temporal"),
            "planted_amount_conflict_found": any(any(t and t.startswith("news_contract_a") for t in d) for d, x in zip(con_docs, con) if x["type"] == "attribute")}


def eval_calibration(al: Alignment) -> dict[str, Any]:
    gold = _gold_rel_set(al.gold)
    title = {d["document_id"]: t for t, d in al.docs.items()}
    raw, cal, y = [], [], []
    for c in al.st.claims(al.case):
        if c["epistemic"] == "stated" or title.get(c["document_id"]) not in al.gold_spans or c["predicate"] == "MENTIONED_WITH":
            continue
        s, o = al.m2g.get(c["subject_mention_id"], "?"), al.m2g.get(c["object_mention_id"], "?")
        if c["predicate"] in SYMMETRIC:
            s, o = sorted((s, o))
        y.append(int((title[c["document_id"]], s, c["predicate"], o) in gold))
        raw.append(c["raw_score"])
        cal.append(c["confidence"])
    return {"claims": len(y), "accuracy": round(float(np.mean(y)) if y else 0, 4), "ece_raw_score": round(ece(np.array(raw), np.array(y)), 4),
            "ece_calibrated": round(ece(np.array(cal), np.array(y)), 4)}


def eval_robustness(out: Path, model_dir: Path | None, seed: int, levels=(0.0, 0.01, 0.03, 0.06)) -> list[dict[str, Any]]:
    """OCR-noise sweep: character confusions applied everywhere, including entity names."""
    rows = []
    mode = "hybrid" if model_dir and (model_dir / "relation_hybrid.pkl").exists() else "rules"
    for lv in levels:
        corpus = generate(out / f"corpus_noise_{lv}", seed, "test", noise=lv)
        gold = json.loads((corpus / "gold.json").read_text())
        st, cid, _ = run_system(corpus, out / f"work_noise_{lv}", mode, model_dir)
        al = Alignment(st, cid, gold)
        n, r, e = eval_ner(al), eval_relations(al, mode), eval_er(al)
        rows.append({"noise": lv, "ner_relaxed_f1": n["drishti"]["relaxed"]["f1"], "ner_baseline_f1": n["regex_baseline"]["relaxed"]["f1"],
                     "relation_f1": r["overall"]["f1"], "er_pairwise_f1": e["drishti"]["f1"], "er_fuzzy_baseline_f1": e["naive_fuzzy_transitive"]["f1"]})
    return rows


def eval_performance(out: Path, model_dir: Path | None) -> dict[str, Any]:
    """Throughput on a larger corpus: the four robustness corpora (distinct bytes) pooled together."""
    pool = out / "corpus_perf" / "docs"
    if pool.parent.exists():
        shutil.rmtree(pool.parent)
    pool.mkdir(parents=True)
    for src in sorted(out.glob("corpus_noise_*")):
        tag = src.name.split("_")[-1]
        for f in (src / "docs").iterdir():
            shutil.copy(f, pool / f"n{tag}_{f.name}" if not f.name.endswith(".meta.json") else pool / f"n{tag}_{f.name}")
    corpus = pool.parent
    res: dict[str, Any] = {"documents": len([f for f in pool.iterdir() if not f.name.endswith(".meta.json")])}
    for name, workers, procs in (("sequential_cold", 1, False), ("threads_4_cold", 4, False), ("processes_4_cold", 4, True)):
        _, _, m = run_system(corpus, out / f"perf_{name}", "rules", model_dir, workers=workers, processes=procs, cache=True)
        res[name] = {k: m[k] for k in ("ms_total", "ms_first_pass", "docs_per_minute")}
    st, cid, _ = run_system(corpus, out / "perf_warm", "rules", model_dir, workers=4, processes=False)
    m2 = Pipeline(st, merged_pack(), "rules", model_dir, workers=4, use_processes=False).run(cid)
    res["warm_cache_rerun"] = {k: m2[k] for k in ("ms_total", "ms_first_pass", "docs_per_minute", "cached")}
    from drishti.api.service import Service
    svc = Service(st)
    lat = []
    orgs = st.entities(cid, "ORGANISATION")[:20]
    for e in orgs:
        t0 = time.perf_counter()
        svc.ask("admin", cid, "how is Public Works Department connected to " + e["canonical_name"])
        lat.append((time.perf_counter() - t0) * 1e3)
    res["assistant_query_latency_ms"] = {"median": round(float(np.median(lat)), 1), "p95": round(float(np.percentile(lat, 95)), 1), "n": len(lat)}
    return res


# ============================================================================ main
def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="models")
    ap.add_argument("--skip-perf", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = generate(out / "corpus_test", a.seed, "test")
    gold = json.loads((corpus / "gold.json").read_text())
    model_dir = Path(a.models) if Path(a.models).exists() else None
    results: dict[str, Any] = {"corpus": {"documents": len(gold["documents"]),
                                          "gold_mentions": sum(len(d["mentions"]) for d in gold["documents"].values()),
                                          "gold_relations_in_text": sum(len(d["rels"]) for d in gold["documents"].values()),
                                          "gold_entities": len(gold["entities"])}}
    rel_results, er_main = [], None
    for mode in ("rules", "learned", "hybrid"):
        if mode != "rules" and not (model_dir and (model_dir / "relation_lr.pkl").exists()):
            continue
        st, cid, metrics = run_system(corpus, out / f"work_{mode}", mode, model_dir)
        al = Alignment(st, cid, gold)
        rel_results.append(eval_relations(al, mode))
        if mode == "hybrid" or (mode == "rules" and not (model_dir and (model_dir / "relation_lr.pkl").exists())):
            er_main = (st, cid, al, metrics)
        if mode == "rules":
            results["ner"] = eval_ner(al)
            results["untyped_edges"] = eval_untyped_edges(al)
            rules_al = (st, cid, al, metrics)
    st, cid, al, metrics = er_main or rules_al
    results["relation_extraction"] = rel_results
    results["entity_resolution"] = eval_er(al) | {"blocking": metrics["resolver"]}
    if er_main is not None:
        st0, cid0, _ = run_system(corpus, out / "work_heuristic", "rules", None)
        results["entity_resolution"]["drishti_weighted_no_model"] = eval_er(Alignment(st0, cid0, gold))["drishti"]
    results["retrieval"] = eval_retrieval(al)
    results["graph"] = eval_graph(al)
    results["patterns"] = eval_patterns(al)
    results["evidence"] = eval_evidence(al)
    results["calibration"] = eval_calibration(al)
    if not a.skip_perf:
        results["robustness"] = eval_robustness(out, model_dir, a.seed)
        results["performance"] = eval_performance(out, model_dir)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    from drishti.bench.report import write_markdown
    write_markdown(results, out / "RESULTS.md")
    print(json.dumps(results, indent=1, default=str))
    return results


if __name__ == "__main__":
    main()
