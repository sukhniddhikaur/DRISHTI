# DRISHTI 2.0 — Engineering specification

This is the module-by-module spec that the blueprint asked for as its next step. It covers the data model, the algorithms and their complexity, API contracts and the guarantees each stage gives the next one.

```
            ┌────────────── immutable evidence ──────────────┐
 sources ─► │ Ingestor: SHA-256, dedup, versions, blob store │
            └───────────────┬────────────────────────────────┘
                            ▼  (process pool, cached by hash+version)
            ┌─────────────────────────────┐
            │ Parse cascade               │ native text → OCR (Tesseract+OpenCV) only where needed
            │ Canonical doc + passages    │ page / paragraph / table-row passages with char offsets
            │ Quality + classification    │ garbage ratio, OCR confidence, function-word ratio, dup sections
            └───────────────┬─────────────┘
                            ▼
            ┌─────────────────────────────┐
            │ NER (hybrid)                │ checksum-validated identifiers, grammar heads, case gazetteer
            │ Relations (closed schema)   │ rules + learned LR; constrained decoding; modality; valid time
            │ Structured (CDR/bank/regs)  │ deterministic STATED claims
            │ Span verifier               │ evidence_text == text[a:b] or the claim is rejected
            └───────────────┬─────────────┘
                            ▼  (case level)
            ┌─────────────────────────────┐
            │ Entity resolution           │ profiles → blocking+MinHash LSH → features → LR → constrained clustering
            │ Review queue / analyst ML/CL│ decisions logged, replayed as constraints
            └───────────────┬─────────────┘
                            ▼
   SQLite/Postgres (source of truth) ──► bitemporal graph projection (NetworkX; Neo4j-ready)
                            ▼
   Evidence engine: origin groups · noisy-OR belief · contradictions
   Graph intelligence: centrality · Louvain · k-shortest evidence paths · link prediction · resilience · patterns
   Retrieval: BM25 + LSA + metadata, tuned fusion, entity-alias expansion
   Assistant: NL → IQR (allowlisted) → executor (ACL, limits) → cited answer → verifier (→ optional LLM verbaliser → re-verify)
   Interfaces: REST API · investigation workspace · HTML report · CLI
```

## 1. Data model (`drishti/core/models.py`, `drishti/store/db.py`)

| Table | Purpose | Key invariants |
|---|---|---|
| `documents` | one row per *version* of a source document | `UNIQUE(case_id, file_hash)`; `parent_document_id` links versions; blob is read-only on disk |
| `passages` | structural units | `[char_start, char_end)` index the canonical text |
| `mentions` | NER spans | deterministic id = H(doc, start, end, label) → idempotent re-runs |
| `entities` | resolved identities | rebuilt per resolution run; `active` flag |
| `mention_entity` | mention→entity mapping | never destroyed; superseded (`active=0`) → reversible |
| `resolution_decisions` | every scored pair (auto_link / review / cannot_link / analyst_*) | analyst rows survive re-runs and become constraints |
| `claims` | the unit of knowledge | evidence span verified; modality, epistemic status, `valid_from/to`, `reported_at`, `ingested_at`, extractor+version, raw & calibrated score |
| `claim_history` | field-level change log for analyst actions | append-only |
| `events` | n-ary events (contract award, appointment, resignation, FIR registration, call, transaction) | roles → mention ids |
| `stage_runs`, `stage_cache` | pipeline state per (document, stage) | key = H(input hash, stage, version, config) |
| `users`, `case_acl` | RBAC + per-case grants | API keys stored hashed |
| `audit_log` | hash-chained, append-only (SQLite triggers) | `hash_i = SHA256(hash_{i-1} ‖ canonical_json(entry_i))` |

Three time axes are kept on every claim: **valid time** (when the relationship held, from the sentence), **report time** (publication date of the source) and **transaction time** (`ingested_at`). The graph can be projected `as_of` (valid time) and `known_at` (report/transaction time).

Epistemic status separates the kind of knowledge: `stated` (structured record), `extracted` (model output, unreviewed), `inferred` (analytics — never stored as a claim), `confirmed`/`rejected` (analyst). Modality records what the source *says*: `asserted`, `negated`, `hedged`, `reported`.

## 2. Ingestion (`drishti/ingest/ingestor.py`)

* Adapters (`DirectoryAdapter`, `MemoryAdapter`, protocol `SourceAdapter`) yield `RawItem`s with provenance (source, URL, access method, reliability class, publication date). Sidecar `<file>.meta.json` supplies provenance for folder ingestion.
* Dedup happens *before* processing: identical bytes in a case return `duplicate`. Changed bytes for the same (source, title) become `version+1` with a parent link.
* Blobs are content-addressed (`aa/bb/<sha256>`) and chmod 0444.

## 3. Document understanding (`drishti/understand/`)

* **Cascade**: PyMuPDF native text → OCR only when a page has < 40 chars or garbage ratio > 0.3 → Tesseract with optional OpenCV denoise, deskew (min-area-rect angle) and adaptive threshold; TSV output gives per-word confidence.
* **Structure**: paragraphs (de-hyphenated, soft line breaks joined), headings, tables (PyMuPDF table finder / python-docx / CSV / XLSX) emitted *row by row* as `Header: value; …` passages. Nothing is chunked into arbitrary token windows.
* **Sentence segmentation**: offset-preserving with an Indian legal/corporate abbreviation lexicon (`Pvt.`, `u/s.`, `w.e.f.`, initials); legal-form abbreviations end a sentence only when a new capitalised word follows ("… Ltd. Mr. X").
* **Quality**: garbage ratio, function-word ratio (OCR noise detector), duplicate passages, low OCR confidence → `needs_review`.
* **Classification**: cue-weighted log-odds (cue weight = 1/#classes sharing the cue) + table-header signals, softmax scores stored for explainability.

## 4. Extraction (`drishti/extract/`)

**Identifiers** (`identifiers.py`) — precision-first, every pattern has a validator:

| Type | Validation |
|---|---|
| GSTIN | structure + embedded PAN + official mod-36 check character |
| PAN | structure + holder-type 4th char (P/C/H/F/A/T/B/L/J/G → entity-type signal) |
| CIN | listing flag, state code, year range, ownership class |
| IFSC, vehicle registration, phone (+91 normalisation), DIN, account, FIR/court case numbers, tender refs, e-mail | structure; UTR-prefixed digits are not phones |
| Aadhaar-like | Verhoeff checksum, **detected only to redact** (stored as `XXXX-XXXX-1234`) |
| Money | ₹/Rs./INR with lakh/crore/million units → integer rupees |

**NER** (`ner.py`): validated identifiers; dates; organisation = legal-form suffix head + left scan over capitalised tokens (stops at role words, other legal-form words, sentence starters); government bodies (prefix and head grammars); persons (honorific, role cue, kinship/alias cue, given-name gazetteer); locations. Overlaps resolve by (priority, length, score) and multi-layer agreement raises the score. Then **case-level propagation**: high-confidence names found anywhere in the case are searched in every document, including suffix-less heads with prefix-truncated last tokens ("ABC Infra" after "ABC Infrastructure Pvt Ltd"). Identifiers attach to the nearest preceding owner in the same clause (skipping kinship references such as "s/o Mohan Yadav"); they become ER signals and `USES_PHONE`/`HOLDS_ACCOUNT` claims.

**Relations** (`relations.py`): candidates are ordered mention pairs in one sentence whose types fit *some* signature in the closed schema (`core/domain.py`).
* Rules: trigger lexicon with direction (active, passive, *by*-passive "awarded to A by B", right-context "X appointed Y as director"), intervening-argument stealing, coordination tolerance, distance penalty.
* Learned: multinomial logistic regression over sparse features (between-words with entity placeholders, bigrams, left/right context, distance bucket, inner-entity count, type pair); direction-aware labels (`DIRECTOR_OF>`); **constrained decoding** drops schema-violating predictions.
* Hybrid (default): rule hits are kept and re-scored; the model may add relations the lexicon missed when p ≥ 0.6.
* Modality (`modality.py`): scope-limited — negation must sit within 30 chars before the trigger or between the arguments; embedded relative clauses (", who was never charged,") are masked; "No." (number) and "no longer" are excluded; hedging/attribution looked up inside the clause.
* Valid time (`temporal.py`): DD/MM/YYYY-first parsing, textual dates, financial years (FY 2023-24 → 2023-04-01…2024-03-31), start/end cue words (w.e.f., since, resigned, ceased).

**Calibration** (`calibrate.py`): isotonic regression per extractor family fitted on a dev world; ECE reported before/after. Uncalibrated families fall back to conservative priors.

## 5. Entity resolution (`drishti/resolve/`)

1. **Profiles**: one per (document, label, normalised name) — one-sense-per-discourse — aggregating surface forms, identifiers, aliases, PAN holder type and context (labelled co-occurring names within 250 chars).
2. **Blocking** (sub-quadratic): identifier keys, org first token / 4-char prefix / sorted-token key, person surname phonetic key (Indic transliteration-aware), initials key, first-name-phonetic+surname-initial, and **MinHash LSH** over char 3-grams (30 permutations, 10 bands × 3 rows; P(candidate | J=0.7) ≈ 0.97). Oversized non-identifier blocks are dropped as stop-blocks.
3. **Hard constraints** (cannot-link): different PAN/DIN/CIN/Aadhaar, PAN holder-type mismatch, incompatible legal form (LLP vs Pvt Ltd), two different full names in one document.
4. **Features**: Jaro-Winkler, Monge-Elkan soft token overlap, token Jaccard, char-n-gram TF-IDF cosine, phonetic equality, initials compatibility, strong/soft identifier matches, context Jaccard, location overlap, length ratio, first/last token equality.
5. **Scorer**: logistic regression trained on labelled pairs from clean and OCR-noised training worlds; weighted formula `S = w1·name + w2·identifier + w3·address + w4·context` as fallback.
6. **Constrained average-linkage clustering**: pairs processed by descending score; two clusters merge only if no cross pair violates a constraint and the average cross-pair score ≥ 0.8·t_high. This blocks the A~B, B~C ⇒ A~C chaining failure. Mid-band pairs go to the review queue.
7. **Reversibility**: analyst must-link / cannot-link decisions are stored and replayed on every run; mention→entity rows are superseded, never deleted.

## 6. Evidence engine (`drishti/evidence/support.py`)

* **Origin groups**: claims for one triple are unioned when they share a document version chain, have near-duplicate evidence (MinHash over word 4-shingles, J ≥ 0.5 — syndicated copies), or a `reported` claim attributes itself to a record type present among the other claims ("according to the tender record").
* **Belief**: `support = 1 − Π_g (1 − w(modality)·max_{c∈g} p_c)`, the same for negating claims, `belief = support·(1 − contra)`; analyst confirmation overrides to 0.99; rejected claims are excluded.
* **Contradictions**: polarity (asserted vs negated, overlapping valid time, different documents), functional predicates (two parents at once), temporal staleness (ended at T, later source states it as current), attribute conflicts (same contract reference or same transaction triple with different amounts or dates).

## 7. Graph intelligence (`drishti/graph/`)

* **Projection**: `MultiDiGraph` with one edge per (s, p, o), carrying belief, support/contra, independent origins, document count, valid interval, modalities and claim ids. The undirected analytics view drops weak co-occurrence edges and LOCATION hubs by default.
* **Centrality**: betweenness on cost = −log(belief) (k-sampled above 300 nodes), PageRank, eigenvector, degree, k-core, articulation points.
* **Communities**: Louvain (weighted, seeded) with modularity; label propagation for comparison; per-community density, boundary edges and most central member.
* **Paths**: Yen's k-shortest simple paths on cost = −log(belief) + 0.15 per hop (+1.5 for a weak edge), bounded hops, each hop returned with its claims and evidence; labelled `direct`/`indirect`.
* **Link prediction**: Adamic-Adar, resource allocation, Jaccard, preferential attachment, weighted common neighbours, combined by logistic regression trained by edge hold-out and evaluated by ROC-AUC; output is a hypothesis list with supporting neighbours (never stored as claims).
* **Resilience**: adaptive targeted removal (recomputed betweenness), compared with degree and random removal; largest-component fraction and global efficiency.
* **Patterns** (explicit, analyst-tunable parameters, all return claim ids): temporal money cycles (DFS with non-decreasing timestamps, window, amount band [r, 1.05]), sliding-window pass-through accounts, structuring just below reporting thresholds, call bursts around case/event dates (z-score against the daily baseline), shared identifiers across resolved persons, director interlocks between suppliers of the same buyer.
* **Temporal**: snapshot diffs (new/ended edges between two dates), entity timelines (events + report dates).

## 8. Retrieval (`drishti/retrieval/hybrid.py`)

`Score = α·BM25 + β·LSA + γ·metadata`, each min-max normalised per query. BM25 (k1=1.5, b=0.75) runs over an inverted index; LSA uses word 1–2-gram plus char 3–5-gram TF-IDF reduced with truncated SVD (128 dims). Metadata (document type, reliability, heading penalty) is a tie-breaker (γ ≤ 0.3). Weights are tuned by grid search on dev queries. Reciprocal-rank fusion is available, and entity-alias expansion adds the resolved aliases of any linked entity to the query.

## 9. Assistant (`drishti/assistant/`)

* **Parser** → `IQR` (Pydantic, allowlisted intents `find_paths | find_related | relation_lookup | entity_profile | timeline | centrality | communities | contradictions | patterns | hypotheses | evidence_search`, `max_hops ≤ 4`, `limit ≤ 50`, allowlisted predicates and types). Entity linking does a longest match over canonical names and aliases, with a Jaro-Winkler n-gram fallback.
* **Executor**: re-checks the case ACL, enforces the time budget, works only with parameterised store and graph calls, and has no path to raw SQL.
* **Answer**: confidence-aware sentences ("according to 2 independent sources", "documents that trace back to a single origin", "alleged (hedged language)", "not yet reviewed"), caveats for indirect paths and structural prominence.
* **Verifier**: every finding must cite claims that exist, belong to the case and are not rejected. Failing sentences are dropped and counted.
* **Optional LLM verbaliser** (`llm.py`, off by default): receives only verified sentences and must return JSON with citations. Every sentence is re-verified (citations ⊆ provided, no new capitalised names), and the template answer is kept on any failure.

## 10. API contract (`drishti/api/app.py`) — header `X-API-Key`

| Method | Path | Notes |
|---|---|---|
| GET | `/api/cases` · POST `/api/cases` | list/create (creator gets owner grant) |
| POST | `/api/cases/{case}/documents` | multipart `files[]`, `reliability`, `publication_date` |
| POST | `/api/cases/{case}/process` | runs the pipeline, returns metrics |
| GET | `/api/cases/{case}/entities?type=&q=` · `/api/entities/{id}` | profile with relations, belief, origin groups, timeline |
| GET | `/api/claims/{id}` · POST `/api/claims/{id}/review` | evidence in context, span check, history; confirm/reject/dispute |
| GET | `/api/cases/{case}/review-queue` · POST `/api/cases/{case}/resolve` | ER queue; analyst link/split |
| GET | `/api/cases/{case}/graph?as_of=&include_weak=&min_belief=` | nodes (centrality, community) + edges (belief, independence) |
| GET | `/api/cases/{case}/paths?a=&b=&k=&max_hops=` | evidence paths |
| GET | `/api/cases/{case}/analytics` · `/patterns` · `/contradictions` · `/timeline` · `/diff?t1=&t2=` · `/search?q=` | |
| POST | `/api/cases/{case}/ask` | `{question, use_llm}` → IQR + cited answer + verification |
| GET | `/api/cases/{case}/audit` · `/report` | audit chain status; HTML report |

## 11. Reliability

* Idempotent stages through deterministic ids. Each stage records status, attempts, error class and duration.
* Transient errors (OS/timeout) are retried with exponential backoff; permanent errors fail only that document.
* A document that fails at `extract` never repeats `parse+ner`: that result is cached by file hash and version.
* Analyst-confirmed or rejected claims are never overwritten by re-extraction.

## 12. Scaling path (deliberately not built on day one)

| Today | Swap when needed |
|---|---|
| SQLite (WAL) | PostgreSQL — the schema uses only portable types; JSON columns map to JSONB |
| NetworkX projection | Neo4j/Memgraph projection fed from `claims` (the graph is a view, so it is rebuildable) |
| LSA vectors | Sentence-Transformers + pgvector behind the same `HybridRetriever` interface |
| process pool | Celery/RQ workers per stage; `stage_runs` already models durable task state |
| rules+LR relations | fine-tuned transformer relation classifier behind `RelationExtractor` (same constrained decoding) |
