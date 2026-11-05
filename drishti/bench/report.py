"""Render benchmark results as a Markdown report."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _f(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def write_markdown(r: dict[str, Any], path: Path) -> None:
    L: list[str] = ["# DRISHTI benchmark results", "",
                    "Synthetic, fully fictional test world with exact ground truth (held-out templates, disjoint names from the training world). "
                    "These numbers measure the pipeline on controlled data; they are **not** estimates of real-world accuracy.", ""]
    c = r["corpus"]
    L += [f"Corpus: {c['documents']} documents · {c['gold_mentions']} gold mentions · {c['gold_relations_in_text']} gold textual relations · "
          f"{c['gold_entities']} gold entities", ""]
    if "ner" in r:
        L += ["## 1. Entity extraction", "", "| System | Strict P | Strict R | Strict F1 | Relaxed P | Relaxed R | Relaxed F1 |", "|---|---|---|---|---|---|---|"]
        for k, v in r["ner"].items():
            s, x = v["strict"], v["relaxed"]
            L.append(f"| {k} | {_f(s['precision'])} | {_f(s['recall'])} | {_f(s['f1'])} | {_f(x['precision'])} | {_f(x['recall'])} | {_f(x['f1'])} |")
        L += ["", "Per label (DRISHTI, relaxed):", "", "| Label | P | R | F1 |", "|---|---|---|---|"]
        for k, v in r["ner"]["drishti"]["per_label_relaxed"].items():
            L.append(f"| {k} | {_f(v['precision'])} | {_f(v['recall'])} | {_f(v['f1'])} |")
        L.append("")
    if r.get("relation_extraction"):
        L += ["## 2. Relation extraction (typed, document level)", "", "| Mode | P | R | F1 | Modality acc. | Non-asserted modality acc. |", "|---|---|---|---|---|---|"]
        for x in r["relation_extraction"]:
            o = x["overall"]
            L.append(f"| {x['system']} | {_f(o['precision'])} | {_f(o['recall'])} | {_f(o['f1'])} | {_f(x['modality_accuracy'])} | {_f(x['non_asserted_modality_accuracy'])} |")
        best = r["relation_extraction"][-1]
        L += ["", f"Per predicate ({best['system']}):", "", "| Predicate | P | R | F1 |", "|---|---|---|---|"]
        for k, v in best["per_predicate"].items():
            L.append(f"| {k} | {_f(v['precision'])} | {_f(v['recall'])} | {_f(v['f1'])} |")
        L.append("")
    if "untyped_edges" in r:
        u = r["untyped_edges"]
        L += ["### Prototype co-occurrence edges vs DRISHTI claims (untyped)", "", "| System | P | R | F1 |", "|---|---|---|---|"]
        for k, v in u.items():
            L.append(f"| {k} | {_f(v['precision'])} | {_f(v['recall'])} | {_f(v['f1'])} |")
        L.append("")
    if "entity_resolution" in r:
        e = r["entity_resolution"]
        L += ["## 3. Entity resolution", "", f"Mentions evaluated: {e['mentions_evaluated']}", "",
              "| System | Pairwise P | Pairwise R | Pairwise F1 | B³ F1 | False-merge rate | Fragmented gold |", "|---|---|---|---|---|---|---|"]
        for k in ("exact_string", "naive_fuzzy_transitive", "drishti"):
            v = e[k]
            L.append(f"| {k} | {_f(v['precision'])} | {_f(v['recall'])} | {_f(v['f1'])} | {_f(v['bcubed_f1'])} | {_f(v['false_merge_rate'])} | {_f(v['fragmented_gold_rate'])} |")
        if "drishti_weighted_no_model" in e:
            v = e["drishti_weighted_no_model"]
            L.append(f"| drishti (weighted formula, no learned model) | {_f(v['precision'])} | {_f(v['recall'])} | {_f(v['f1'])} | {_f(v['bcubed_f1'])} | {_f(v['false_merge_rate'])} | {_f(v['fragmented_gold_rate'])} |")
        b = e["blocking"]
        L += ["", f"Blocking: {b['candidate_pairs']} candidate pairs vs {b['naive_pairs']} naive (reduction {b['reduction_ratio']:.1%}); "
              f"{b.get('review_pairs', 0)} pairs routed to human review; review-queue true-match fraction {e['review_queue']['true_match_fraction']}.", ""]
    if "retrieval" in r:
        x = r["retrieval"]
        L += ["## 4. Evidence retrieval", "", f"{x['queries_test']} test queries (paraphrased, no trigger words), {x['passages']} passages. "
              f"Tuned weights: α={x['tuned_weights']['alpha']}, β={x['tuned_weights']['beta']}, γ={x['tuned_weights']['gamma']}.", "",
              "| Method | Recall@5 | nDCG@10 | MRR |", "|---|---|---|---|"]
        for k in ("bm25", "semantic_lsa", "hybrid_no_expansion", "hybrid_rrf", "hybrid_tuned_entity_expansion"):
            v = x[k]
            L.append(f"| {k} | {_f(v['recall@5'])} | {_f(v['ndcg@10'])} | {_f(v['mrr'])} |")
        L.append("")
    if "graph" in r:
        g = r["graph"]
        c = g["communities"]
        L += ["## 5. Graph intelligence", "", f"Graph: {g['nodes']} nodes, {g['edges']} evidence edges (weak co-occurrence excluded).", "",
              "| Community method | ARI vs gold groups | NMI |", "|---|---|---|",
              f"| Louvain (evidence-weighted) | {_f(c['louvain_ARI'])} | {_f(c['louvain_NMI'])} |",
              f"| Connected components | {_f(c['connected_components_ARI'])} | – |",
              f"| Prototype (no clustering) | {_f(c['prototype_no_clustering_ARI'])} | – |", ""]
        lp = g["link_prediction"]
        if lp.get("auc_combined") is not None:
            L += [f"Link prediction (edge hold-out ROC-AUC): combined {lp['auc_combined']}, Adamic-Adar {lp.get('auc_aa')}, "
                  f"resource allocation {lp.get('auc_ra')}, preferential attachment {lp.get('auc_pa_log')}.", ""]
        pq = g["path_queries"]
        L += ["| Path discovery (≤3 hops) | Connected pairs found | Hop precision | False-connection rate |", "|---|---|---|---|"]
        for k in ("drishti_evidence_graph", "cooccurrence_graph"):
            v = pq[k]
            L.append(f"| {k} | {_f(v['connected_pairs_found'])} | {_f(v['hop_precision'])} | {_f(v['false_connection_rate'])} |")
        if "resilience_5_removals" in g:
            rr = g["resilience_5_removals"]
            L += ["", f"Resilience after 5 removals — targeted (betweenness): LCC {rr['targeted_lcc']}, efficiency {rr['targeted_efficiency']}; "
                  f"random: LCC {rr['random_lcc']}, efficiency {rr['random_efficiency']}."]
        L.append("")
    if "patterns" in r:
        p = r["patterns"]
        L += ["## 6. Planted pattern detection", "", "| Pattern | Detected |", "|---|---|"]
        for k, v in p["planted_detected"].items():
            L.append(f"| {k} | {'yes' if v else 'no'} |")
        L += ["", f"Findings per detector: {p['findings_per_detector']}", ""]
    if "evidence" in r:
        e = r["evidence"]
        L += ["## 7. Evidence integrity", "", f"- Claims: {e['claims']}; unsupported (evidence span ≠ source text): **{e['unsupported_claims']}**; coverage {e['evidence_coverage']:.0%}",
              f"- Edges supported by >1 document: {e['edges_with_multiple_documents']}; mean documents {e['mean_documents_per_multi_edge']} vs mean independent origins {e['mean_independent_origins_per_multi_edge']}",
              f"- Syndicated-news award edge: {e['syndicated_award_edge']}",
              f"- Contradictions found: {dict(e['contradictions'])}; planted temporal contradiction found: {e['planted_temporal_contradiction_found']}; planted amount conflict found: {e['planted_amount_conflict_found']}", ""]
    if "calibration" in r:
        c = r["calibration"]
        L += ["## 8. Confidence calibration", "", f"{c['claims']} textual claims, accuracy {c['accuracy']}. ECE raw score {c['ece_raw_score']} → calibrated {c['ece_calibrated']}.", ""]
    if "robustness" in r:
        L += ["## 9a. Robustness to OCR-style noise (applied inside entity names too)", "",
              "| Char noise | NER F1 | NER baseline F1 | Relation F1 | ER F1 | ER fuzzy baseline F1 |", "|---|---|---|---|---|---|"]
        for x in r["robustness"]:
            L.append(f"| {x['noise']:.0%} | {_f(x['ner_relaxed_f1'])} | {_f(x['ner_baseline_f1'])} | {_f(x['relation_f1'])} | {_f(x['er_pairwise_f1'])} | {_f(x['er_fuzzy_baseline_f1'])} |")
        L.append("")
    if "performance" in r:
        p = r["performance"]
        L += ["## 9. Processing efficiency", "", f"Pooled corpus of {p.get('documents')} documents (incl. one scanned PDF per sub-corpus).", "",
              "| Run | Total ms | Parse+NER ms | Docs/min |", "|---|---|---|---|"]
        for k in ("sequential_cold", "threads_4_cold", "processes_4_cold", "warm_cache_rerun"):
            v = p[k]
            L.append(f"| {k} | {v['ms_total']} | {v['ms_first_pass']} | {v['docs_per_minute']} |")
        q = p["assistant_query_latency_ms"]
        L += ["", f"Assistant path query latency: median {q['median']} ms, p95 {q['p95']} ms (n={q['n']}).", ""]
    path.write_text("\n".join(L))
