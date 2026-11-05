# DRISHTI 2.0 — Evidence-First Intelligence & Knowledge-Graph Platform

DRISHTI turns fragmented, multi-format documents (court judgments, company filings, tender notices, annual reports, news, FIRs, chargesheets, call-detail records, bank statements, registers) into an explainable, continuously evolving intelligence graph. It keeps the provenance and uncertainty of every extracted fact.

> **Core question:** how can a large set of fragmented, unstructured documents become an accurate, explainable, continuously evolving intelligence graph, while every fact keeps its provenance and its uncertainty?

Every fact DRISHTI shows is a **claim** that points to an exact, verified character span of an immutable source document. Entities are resolved **reversibly**. The graph is a **bitemporal projection** of claims, never the source of truth. Every analytical result (paths, communities, link hypotheses, money-flow patterns, contradictions) comes back with its evidence and an honest label for its uncertainty.

```
pip install -e ".[fast,dev]"      # Tesseract is needed only for scanned documents
drishti demo                      # generates a fictional case, ingests 35 mixed documents, processes, asks questions, writes a report
drishti serve                     # http://127.0.0.1:8000  — API key: drishti-admin-dev-key
pytest -q                         # 30 tests
drishti train && drishti bench    # retrain the learned components; regenerate docs/BENCHMARK.md
```

## What is in the box

| Layer | What it does | Module |
|---|---|---|
| Ingestion | Source adapters; SHA-256 content addressing; duplicate detection before processing; versioning with parent links; read-only evidence blobs | `ingest/` |
| Document understanding | Cost cascade: native text, then OCR (Tesseract + OpenCV deskew/denoise) only for pages that need it; table rows preserved; passages with char offsets; quality flags; explainable document classifier | `understand/` |
| Identifiers | GSTIN (official mod-36 checksum), PAN (holder-type char as an entity-type signal), CIN, IFSC, DIN, vehicle, +91 phone, accounts, FIR/court case numbers, tender refs, ₹ lakh/crore amounts; Aadhaar detected by Verhoeff checksum **only to redact it** | `extract/identifiers.py` |
| NER | Grammar heads for legal forms and government bodies; honorific, role, kinship and alias cues; case-level gazetteer propagation ("ABC Infra" ← "ABC Infrastructure Pvt Ltd"); identifier-to-owner attachment that skips "s/o …" | `extract/ner.py` |
| Relations | Closed schema with direction (active / passive / by-passive / right-context); learned logistic-regression classifier with constrained decoding; hybrid union | `extract/relations.py` |
| Modality and time | Scope-limited negation (relative clauses masked; "No." ≠ no), hedging, attribution; valid-time intervals (w.e.f., resigned, FY 2023-24); report and ingest times | `extract/modality.py`, `extract/temporal.py` |
| Structured records | CDR, bank and subscriber tables become deterministic STATED claims (fuzzy header mapping across operators and banks) | `extract/structured.py` |
| Calibration | Isotonic regression per extractor family fitted on a held-out dev world; ECE reported | `extract/calibrate.py` |
| Entity resolution | Multi-key blocking + MinHash LSH (~97% fewer comparisons); Indic phonetic keys; 16 features; learned pair model; identifier cannot-links; constrained average-linkage clustering (no naive transitivity); human review queue; analyst decisions replayed | `resolve/` |
| Evidence engine | Origin groups (version chains, near-duplicate text, attribution); noisy-OR belief across *independent* origins only; polarity, functional, temporal and amount contradictions | `evidence/support.py` |
| Graph intelligence | Bitemporal projection (`as_of`, `known_at`); evidence-weighted betweenness, PageRank, eigenvector, k-core, articulation points; Louvain; Yen k-shortest evidence paths; link prediction with hold-out AUC; adaptive resilience simulation | `graph/` |
| Pattern leads | Temporal money cycles, pass-through/mule accounts, structuring below reporting thresholds, call bursts around incidents, shared identifiers, director interlocks — explicit thresholds and claim-level evidence | `graph/patterns.py` |
| Retrieval | BM25 (from scratch) + LSA (word + char TF-IDF → SVD) + metadata; fusion weights tuned on dev queries; RRF; entity-alias query expansion | `retrieval/hybrid.py` |
| Assistant | NL → allowlisted **IQR** → ACL-checked executor → confidence-aware, citation-verified answer; optional LLM verbaliser whose output is re-verified | `assistant/` |
| Security and custody | Hashed API keys; admin/analyst/viewer roles; per-case ACL re-checked for every assistant query; append-only, hash-chained audit log that detects tampering | `store/` |
| Interfaces | FastAPI REST API; investigation workspace (graph, evidence panel with highlighted source spans, review queues, leads, audit); self-contained HTML intelligence report; CLI | `api/`, `report/`, `cli.py` |
| Benchmark | Synthetic world generator with exact ground truth, held-out templates, hard negatives and planted patterns; baselines that reproduce the prototype; 9 experiment groups | `bench/` |

## Benchmark highlights

The test world is synthetic and fictional, with exact ground truth. Its templates are held out and its names are disjoint from the training worlds. The full tables are in [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

| Experiment | Baseline (prototype-style) | DRISHTI 2.0 |
|---|---|---|
| Entity extraction F1 (relaxed / strict) | regex: 0.790 / 0.625 | **0.992 / 0.956** |
| Relation edges, precision (untyped) | co-occurrence: 0.119 | **1.000** (typed F1 0.97 rules, 0.965 hybrid) |
| Modality accuracy on negated/hedged/reported | – | **1.000** |
| Entity resolution pairwise F1 · false-merge rate | naive fuzzy + transitive: 0.858 · 0.101 | **0.996 · 0.014** |
| ER comparisons (blocking + LSH) | 28,441 | **887** (−96.9%) |
| ER F1 under 6% OCR-style noise | fuzzy: 0.831 | **0.923** |
| Path discovery: hop precision · false connections | co-occurrence graph: 0.131 · 0.617 | **1.000 · 0.017** |
| Communities vs gold groups (ARI) | no clustering: 0.0 | **0.53** (NMI 0.70) |
| Link prediction hold-out ROC-AUC | Adamic-Adar alone: 0.68 | **0.79** (combined) |
| Planted patterns detected | keyword list | **6 / 6** |
| Unsupported claims (span ≠ source) | – | **0 of 856** |
| Syndicated news + official record (3 docs) | counted as 3 sources | **1 independent origin** |
| Calibration ECE (raw → isotonic) | – | 0.120 → 0.077 |
| Retrieval Recall@5 | BM25: 0.880 | **0.924** (hybrid + entity expansion) |
| Warm-cache re-run, parse+NER (140 docs) | 18.8 s | **46 ms** |

**Reading these numbers honestly.** The extraction rules and the synthetic templates have the same author, so in-template results are optimistic. That is why the test world uses held-out phrasings, and those expose real gaps: `ACCUSED_IN` recall is 0.47 on "Police arrested X in connection with FIR …", which no rule or training template covers. Hybrid retrieval gains only about 2 points over LSA alone on this small corpus. On the 2-core benchmark machine, parallel parsing gives about 10%; the cache gives 400×. What the benchmark shows is *relative* behaviour against the prototype's methods, and that every safety property (span verification, source independence, contradiction surfacing, no false merges on conflicting identifiers) holds. Validating on real, legally reusable documents with annotation is the next research step.

## The investigation workspace

`drishti serve` opens a three-pane workspace:
* **Ask**: natural-language questions. Every sentence carries claim citations, and the structured query (IQR) can be inspected.
* **Graph**: coloured by Louvain community, node size by betweenness, edge width by belief, red edges where a claim is contradicted, dashed for co-occurrence. Filters: as-of date, minimum belief. Shift-click two nodes to get evidence paths.
* **Evidence panel**: the source passage with the claim span highlighted and verified against the immutable text, the document hash, the three time axes, origin groups, and confirm/reject/dispute buttons whose reasons go to the audit log.
* **Review / Leads / Audit**: the identity-match queue (same/different), patterns, contradictions, link hypotheses, brokers, resilience, and the audit chain status.

Cytoscape.js is vendored, so the workspace runs air-gapped.

## Design decisions that matter

1. **Claims, not edges, are the knowledge.** Several documents can support one relationship, conflicting statements coexist without corrupting the graph, and analyst decisions attach to claims.
2. **Facts are separated from interpretations.** `stated / extracted / inferred / confirmed / rejected` is the epistemic axis; `asserted / negated / hedged / reported` is what the source says. Link predictions are never stored as claims.
3. **Precision over recall where it can hurt someone.** Conflicting identifiers can never be merged, ambiguous identities go to a human, and DRISHTI does not produce suspect or risk labels.
4. **The LLM is optional and never in charge.** Answers are composed from verified claims. An LLM may only rephrase them, and its output is re-verified.
5. **Cheap first.** Native text before OCR, rules before models, deterministic graph algorithms, cached stages keyed by content hash and model version.

## Repository layout

```
drishti/
  core/        models (Pydantic), deterministic ids, domain packs (corporate | investigation | fusion)
  store/       SQLite source of truth, hash-chained audit log, RBAC + case ACL
  ingest/      adapters, hashing, dedup, versioning
  understand/  parser cascade + OCR, segmentation, quality, classification
  extract/     identifiers, temporal, NER, relations, modality, structured, calibration, document extractor
  resolve/     normalisation + Indic phonetics, similarity + MinHash LSH, resolution engine
  evidence/    origin groups, belief aggregation, contradictions
  graph/       projection, analytics, paths, link prediction, patterns, temporal
  retrieval/   BM25 + LSA hybrid retriever
  assistant/   IQR, parser, executor + verifier, optional LLM verbaliser
  pipeline/    orchestrator (cache, retries, parallelism, stage state)
  api/         FastAPI app, service layer, workspace UI (static)
  report/      HTML intelligence report
  bench/       synthetic world, baselines, metrics, training, evaluation, report
models/        trained relation classifier, ER pair model, calibrator (+ training report)
docs/          ARCHITECTURE.md (engineering spec) · BENCHMARK.md · PROTOTYPE_VS_DRISHTI2.md
tests/         30 unit and integration tests
```

## Research framing

*An Evidence-Preserving Framework for Multi-Source Entity Resolution and Explainable Knowledge-Graph Construction from Heterogeneous Documents.* The contributions it can support: (1) a claim-centric bitemporal data model with verified evidence spans; (2) constrained, reversible entity resolution with identifier cannot-links and anti-chaining average linkage; (3) source-independence-aware belief aggregation; (4) an evaluation protocol with held-out templates, hard negatives, planted structures and noise sweeps.

## Limitations and next steps

* Validate on real, legally reusable documents with human annotation (inter-annotator agreement), and report results per document type.
* Replace the relation LR with a fine-tuned transformer behind the same constrained-decoding interface. Add coreference resolution ("the Company", "the said accused").
* Use Sentence-Transformers + pgvector for semantic retrieval; Postgres + Neo4j projections at scale; Celery workers per stage.
* Hindi/Marathi OCR and transliteration-aware NER.
* Active learning from the review queue (analyst decisions become training pairs).

All names in the synthetic corpus are fictional. Nothing in DRISHTI constitutes a finding of wrongdoing: it ranks and explains, and a human confirms.
