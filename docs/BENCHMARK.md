# DRISHTI benchmark results

Synthetic, fully fictional test world with exact ground truth (held-out templates, disjoint names from the training world). These numbers measure the pipeline on controlled data; they are **not** estimates of real-world accuracy.

Corpus: 35 documents · 507 gold mentions · 209 gold textual relations · 168 gold entities

## 1. Entity extraction

| System | Strict P | Strict R | Strict F1 | Relaxed P | Relaxed R | Relaxed F1 |
|---|---|---|---|---|---|---|
| drishti | 0.960 | 0.953 | 0.956 | 0.996 | 0.988 | 0.992 |
| regex_baseline | 0.654 | 0.600 | 0.625 | 0.826 | 0.757 | 0.790 |

Per label (DRISHTI, relaxed):

| Label | P | R | F1 |
|---|---|---|---|
| ACCOUNT | 1.000 | 0.875 | 0.933 |
| CASE | 0.909 | 1.000 | 0.952 |
| GOVT_BODY | 1.000 | 1.000 | 1.000 |
| LOCATION | 1.000 | 1.000 | 1.000 |
| ORGANISATION | 1.000 | 1.000 | 1.000 |
| PERSON | 1.000 | 0.974 | 0.987 |
| PHONE | 1.000 | 1.000 | 1.000 |
| VEHICLE | 1.000 | 1.000 | 1.000 |

## 2. Relation extraction (typed, document level)

| Mode | P | R | F1 | Modality acc. | Non-asserted modality acc. |
|---|---|---|---|---|---|
| rules | 1.000 | 0.943 | 0.971 | 1.000 | 1.000 |
| learned | 0.934 | 0.734 | 0.822 | 1.000 | 1.000 |
| hybrid | 0.989 | 0.943 | 0.965 | 1.000 | 1.000 |

Per predicate (hybrid):

| Predicate | P | R | F1 |
|---|---|---|---|
| ACCUSED_IN | 1.000 | 0.467 | 0.636 |
| ACQUIRED | 1.000 | 1.000 | 1.000 |
| ASSOCIATE_OF | 1.000 | 0.875 | 0.933 |
| AWARDED_CONTRACT_TO | 1.000 | 1.000 | 1.000 |
| DIRECTOR_OF | 0.971 | 1.000 | 0.986 |
| FILED_CASE_AGAINST | 1.000 | 1.000 | 1.000 |
| HOLDS_ACCOUNT | 1.000 | 0.875 | 0.933 |
| LOCATED_IN | 1.000 | 1.000 | 1.000 |
| OWNS_VEHICLE | 1.000 | 1.000 | 1.000 |
| SHAREHOLDER_OF | 1.000 | 1.000 | 1.000 |
| SUBSIDIARY_OF | 1.000 | 1.000 | 1.000 |
| USES_PHONE | 1.000 | 1.000 | 1.000 |

### Prototype co-occurrence edges vs DRISHTI claims (untyped)

| System | P | R | F1 |
|---|---|---|---|
| cooccurrence_baseline | 0.119 | 0.984 | 0.213 |
| drishti_typed_claims | 1.000 | 0.943 | 0.971 |

## 3. Entity resolution

Mentions evaluated: 430

| System | Pairwise P | Pairwise R | Pairwise F1 | B³ F1 | False-merge rate | Fragmented gold |
|---|---|---|---|---|---|---|
| exact_string | 1.000 | 0.851 | 0.919 | 0.928 | 0.000 | 0.312 |
| naive_fuzzy_transitive | 0.800 | 0.926 | 0.858 | 0.922 | 0.101 | 0.188 |
| drishti | 0.999 | 0.994 | 0.996 | 0.994 | 0.014 | 0.025 |
| drishti (weighted formula, no learned model) | 0.994 | 0.987 | 0.991 | 0.989 | 0.028 | 0.037 |

Blocking: 887 candidate pairs vs 28441 naive (reduction 96.9%); 8 pairs routed to human review; review-queue true-match fraction 0.5.

## 4. Evidence retrieval

51 test queries (paraphrased, no trigger words), 172 passages. Tuned weights: α=0, β=0.7, γ=0.3.

| Method | Recall@5 | nDCG@10 | MRR |
|---|---|---|---|
| bm25 | 0.880 | 0.837 | 0.797 |
| semantic_lsa | 0.904 | 0.843 | 0.802 |
| hybrid_no_expansion | 0.904 | 0.843 | 0.802 |
| hybrid_rrf | 0.919 | 0.849 | 0.808 |
| hybrid_tuned_entity_expansion | 0.924 | 0.843 | 0.799 |

## 5. Graph intelligence

Graph: 166 nodes, 318 evidence edges (weak co-occurrence excluded).

| Community method | ARI vs gold groups | NMI |
|---|---|---|
| Louvain (evidence-weighted) | 0.526 | 0.704 |
| Connected components | 0.000 | – |
| Prototype (no clustering) | 0.000 | – |

Link prediction (edge hold-out ROC-AUC): combined 0.791, Adamic-Adar 0.684, resource allocation 0.684, preferential attachment 0.7768.

| Path discovery (≤3 hops) | Connected pairs found | Hop precision | False-connection rate |
|---|---|---|---|
| drishti_evidence_graph | 0.567 | 1.000 | 0.017 |
| cooccurrence_graph | 0.550 | 0.131 | 0.617 |

Resilience after 5 removals — targeted (betweenness): LCC 0.5575, efficiency 0.7936; random: LCC 0.6106, efficiency 0.9858.

## 6. Planted pattern detection

| Pattern | Detected |
|---|---|
| money_cycle | yes |
| structuring | yes |
| pass_through | yes |
| shared_phone | yes |
| call_burst | yes |
| director_interlock | yes |

Findings per detector: {'money_cycles': 1, 'pass_through': 1, 'structuring': 1, 'call_bursts': 1, 'shared_identifiers': 2, 'director_interlocks': 3}

## 7. Evidence integrity

- Claims: 856; unsupported (evidence span ≠ source text): **0**; coverage 100%
- Edges supported by >1 document: 29; mean documents 2.172 vs mean independent origins 1.966
- Syndicated-news award edge: [{'documents': 3, 'independent_origins': 1, 'groups': ['attributed to tender']}]
- Contradictions found: {'temporal': 1, 'attribute': 1}; planted temporal contradiction found: True; planted amount conflict found: True

## 8. Confidence calibration

200 textual claims, accuracy 0.99. ECE raw score 0.1204 → calibrated 0.0767.

## 9a. Robustness to OCR-style noise (applied inside entity names too)

| Char noise | NER F1 | NER baseline F1 | Relation F1 | ER F1 | ER fuzzy baseline F1 |
|---|---|---|---|---|---|
| 0% | 0.992 | 0.790 | 0.965 | 0.996 | 0.858 |
| 1% | 0.989 | 0.789 | 0.951 | 0.978 | 0.850 |
| 3% | 0.979 | 0.782 | 0.916 | 0.963 | 0.843 |
| 6% | 0.967 | 0.782 | 0.869 | 0.923 | 0.831 |

## 9. Processing efficiency

Pooled corpus of 140 documents (incl. one scanned PDF per sub-corpus).

| Run | Total ms | Parse+NER ms | Docs/min |
|---|---|---|---|
| sequential_cold | 28637.9 | 18785.6 | 259.8 |
| threads_4_cold | 25964.0 | 16418.8 | 286.6 |
| processes_4_cold | 25478.8 | 15781.1 | 292.0 |
| warm_cache_rerun | 9732.5 | 45.9 | 764.4 |

Assistant path query latency: median 106.3 ms, p95 115.3 ms (n=20).
