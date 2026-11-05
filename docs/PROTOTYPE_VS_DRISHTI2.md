# Prototype → DRISHTI 2.0: what each "for-show" label became

Each row of the *Prototype vs Production* comparison is listed with its current implementation and where to find it.

| Capability | Prototype | DRISHTI 2.0 (this repo) | Where |
|---|---|---|---|
| Reading documents | Client-side PDF.js/Mammoth/SheetJS/Tesseract, no retries | Server-side parse cascade (native → OCR only where needed, with OpenCV preprocessing), table rows preserved, retries + per-stage state, cache by file hash | `understand/parser.py`, `pipeline/orchestrator.py` |
| Finding entities | Regex + stop-word filters | Checksum-validated identifiers (GSTIN mod-36, PAN holder type, CIN, Verhoeff-redacted Aadhaar), grammar NER, case-level gazetteer propagation, optional spaCy backend. Benchmark: F1 0.99 vs 0.79 for the regex baseline | `extract/identifiers.py`, `extract/ner.py` |
| Linking entities | Edge if two entities share a document | Typed relations from a closed schema with direction, negation/hedging/attribution and valid time. Benchmark: edge precision 0.99–1.00 vs 0.12 for co-occurrence | `extract/relations.py`, `extract/modality.py` |
| Role classification (suspect/victim/witness) | Keyword scoring | Roles are typed relations stated in the record (`ACCUSED_IN`, `COMPLAINANT_IN`) with their evidence. DRISHTI deliberately does **not** assign suspect/risk labels | `core/domain.py` |
| Duplicate entities | Pairwise string check capped at 60 entities | Blocking + MinHash LSH (~97% fewer comparisons), learned pair model, identifier cannot-links, constrained clustering, human review queue, reversible decisions | `resolve/engine.py` |
| Community detection | Nodes on a circle sorted by mentions | Evidence-weighted Louvain with modularity, label propagation comparison, per-community summaries. Benchmark ARI vs gold groups reported (prototype = 0) | `graph/analytics.py` |
| Network resilience | Edges ÷ possible edges | Adaptive targeted removal (recomputed betweenness) vs random/degree, largest component and global efficiency | `graph/analytics.py` |
| Anomaly / hypothesis | Fixed "suspicious keyword" list | Temporal money cycles, pass-through, structuring, call bursts, shared identifiers, director interlocks (explicit thresholds, claim-level evidence), and link prediction with hold-out AUC | `graph/patterns.py`, `graph/linkpred.py` |
| Data storage | Browser memory | Relational source of truth with versions, provenance, claim history (SQLite → Postgres-portable) | `store/db.py` |
| Authentication | Cosmetic login | Hashed API keys, roles (admin/analyst/viewer), per-case ACL enforced in the service and re-checked for assistant queries | `store/security.py`, `api/service.py` |
| Chain of custody | SHA-256 in browser, not stored | Content-addressed read-only blobs, append-only hash-chained audit log (DB triggers + verification that finds the first tampered entry) | `store/audit.py` |
| Multi-user | Single tab | Multi-user API; analyst actions audited with reasons; review queues | `api/app.py` |
| Scale | A handful of files | Sub-quadratic ER, cached incremental stages, process/thread pools, k-sampled betweenness; scaling path to Postgres/Neo4j/Celery documented | `docs/ARCHITECTURE.md §12` |

## Relation to the SIH26189 framing

The engine has one core and configurable domain packs (`core/domain.py`). `corporate` covers filings, tenders, judgments and annual reports. `investigation` covers FIRs, chargesheets, CDRs, bank statements and subscriber registers. `fusion` merges the two, so one case can connect a contractor's directors to accounts that appear in a bank statement. The SIH pitch's claims (every edge traceable to its source record, community detection, centrality, link prediction, human-in-the-loop) are now real implementations with measured results rather than labels.
