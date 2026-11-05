"""Self-contained HTML intelligence report: findings, network, evidence, method, limits."""
from __future__ import annotations

import html
import math
from collections import Counter
from typing import Any

import networkx as nx

from drishti.core.models import utcnow
from drishti.graph.projection import build

E = html.escape
PAL = ["#1f5fbf", "#c2410c", "#0f766e", "#7c3aed", "#b91c1c", "#4d7c0f", "#a16207", "#0e7490"]


def _svg(U: nx.Graph, comm: dict[str, int], names: dict[str, str], top: int = 60) -> str:
    if U.number_of_nodes() == 0:
        return "<p>No relationships to draw.</p>"
    bt = nx.betweenness_centrality(U, k=min(200, U.number_of_nodes()), seed=7)
    keep = sorted(U, key=lambda v: -(bt[v] + U.degree(v) * 1e-3))[:top]
    H = U.subgraph(keep)
    pos = nx.spring_layout(H, seed=7, k=1.6 / math.sqrt(max(1, H.number_of_nodes())), weight="weight")
    W, Hh, pad = 900, 560, 40
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    sx = lambda x: pad + (x - min(xs)) / ((max(xs) - min(xs)) or 1) * (W - 2 * pad)
    sy = lambda y: pad + (y - min(ys)) / ((max(ys) - min(ys)) or 1) * (Hh - 2 * pad)
    out = [f'<svg viewBox="0 0 {W} {Hh}" width="100%" role="img" aria-label="Evidence network">']
    for u, v, d in H.edges(data=True):
        out.append(f'<line x1="{sx(pos[u][0]):.1f}" y1="{sy(pos[u][1]):.1f}" x2="{sx(pos[v][0]):.1f}" y2="{sy(pos[v][1]):.1f}" '
                   f'stroke="#9aa0a8" stroke-opacity="{0.25 + 0.6 * d.get("weight", .5):.2f}" stroke-width="{0.6 + 2 * d.get("weight", .5):.1f}"/>')
    for v in H:
        r = 4 + 40 * math.sqrt(bt[v])
        c = PAL[comm.get(v, 0) % len(PAL)]
        out.append(f'<circle cx="{sx(pos[v][0]):.1f}" cy="{sy(pos[v][1]):.1f}" r="{r:.1f}" fill="{c}" fill-opacity="0.85"><title>{E(names.get(v, v))}</title></circle>')
        if U.nodes[v].get("type") in ("PERSON", "ORGANISATION", "GOVT_BODY") and (bt[v] > 0.02 or H.degree(v) > 4):
            out.append(f'<text x="{sx(pos[v][0]):.1f}" y="{sy(pos[v][1]) + r + 11:.1f}" font-size="10" text-anchor="middle" fill="currentColor">{E(names.get(v, v)[:28])}</text>')
    out.append("</svg>")
    return "".join(out)


def render_report(svc, user: str, case: str) -> str:
    st = svc.st
    c = st.get_case(case)
    docs = st.documents(case)
    ents = st.entities(case)
    names = {e["entity_id"]: e["canonical_name"] for e in ents}
    claims = st.claims(case)
    view = build(st, case, include_weak=False)
    U = view.undirected()
    an = svc.analytics(user, case)
    pats = svc.patterns(user, case)
    con = svc.contradictions(user, case)
    comm = {m: i for i, cm in enumerate(an["communities"]) for m in cm["members"]}
    chain = svc.audit.verify()
    cited: dict[str, dict] = {}

    def cite(cids: list[str]) -> str:
        out = []
        for cid in cids[:4]:
            cl = st.get_claim(cid)
            if cl:
                cited[cid] = cl
                out.append(f'<a href="#{cid}" class="cite">{cid[4:10]}</a>')
        return " ".join(out)

    strong = sorted([(k, e) for k, e in view.edges.items() if e["belief"] >= 0.75 and k[1] not in ("CALLED", "TRANSFERRED_TO", "LOCATED_IN")],
                    key=lambda x: (-x[1]["independent_sources"], -x[1]["belief"]))[:40]
    findings = "".join(
        f"<tr><td>{E(names.get(s, s))}</td><td><code>{E(p)}</code></td><td>{E(names.get(o, o))}</td><td>{e['belief']:.2f}</td>"
        f"<td>{e['independent_sources']} / {e['supporting_documents']}</td><td>{E(e['valid_from'] or '')}{' → ' + E(e['valid_to']) if e['valid_to'] else ''}</td>"
        f"<td>{E(', '.join(e['modalities']))}</td><td>{cite(e['claims'])}</td></tr>" for (s, p, o), e in strong)
    brokers = "".join(f"<tr><td>{E(b['name'] or '')}</td><td>{b['betweenness']:.3f}</td><td>{b['pagerank']:.4f}</td><td>{b['degree']:.3f}</td>"
                      f"<td>{'yes' if b['articulation_point'] else ''}</td></tr>" for b in an["top_brokers"][:12])
    comms = "".join(f"<li><b>Community {cm['id']}</b> — {cm['size']} members, density {cm['density']:.2f}; most central {E(cm.get('most_central_name') or '')}: "
                    f"{E(', '.join(n for n in cm['member_names'][:10] if n))}{' …' if cm['size'] > 10 else ''}</li>" for cm in an["communities"] if cm["size"] >= 3)
    leads = "".join(f"<li><b>{E(k.replace('_', ' '))}</b>: {E(x['description'])} {cite(x.get('claims', []))}</li>" for k, v in pats.items() for x in v[:4])
    contra = "".join(f"<li><b>{E(x['type'])}</b>: {E(x['explanation'])} {cite([y for y in x['claims'] if y])}</li>" for x in con)
    hyps = "".join(f"<li>{E(h['source_name'] or '')} ↔ {E(h['target_name'] or '')} (score {h['score']:.2f}) via {E(', '.join(n for n in h['shared_names'][:4] if n))}</li>"
                   for h in an["hypotheses"][:10])
    reg = "".join(f"<tr><td>{E(d['title'])}</td><td>{E(d['document_type'])}</td><td>{E(d['reliability'])}</td><td>{E(d['publication_date'] or '')}</td>"
                  f"<td><code>{d['file_hash'][:16]}…</code></td><td>{E(', '.join((d['quality'] or {}).get('flags', [])))}</td></tr>" for d in docs)
    appendix = "".join(f'<div class="ev" id="{cid}"><code>{cid}</code> · {E(cl["predicate"])} · {E((st.get_document(cl["document_id"]) or {}).get("title", ""))} '
                       f'p.{cl["page"]} chars {cl["char_start"]}–{cl["char_end"]} · {E(cl["modality"])} · {E(cl["epistemic"])} · p={cl["confidence"]:.2f}'
                       f'<blockquote>{E(cl["evidence_text"])}</blockquote></div>' for cid, cl in cited.items())
    by_type = Counter(e["type"] for e in ents)
    lp = an.get("link_prediction_eval") or {}
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DRISHTI Intelligence Report</title><style>
:root{{--ink:#1d1f22;--muted:#6b6f76;--line:#dedad2;--bg:#fbfaf7;--accent:#1f5fbf}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--ink:#e7e5e0;--muted:#9aa0a8;--line:#2c3036;--bg:#16181b;--accent:#7aa7ff}}}}
:root[data-theme="dark"]{{--ink:#e7e5e0;--muted:#9aa0a8;--line:#2c3036;--bg:#16181b;--accent:#7aa7ff}}
body{{font:15px/1.55 system-ui,sans-serif;color:var(--ink);background:var(--bg);max-width:1040px;margin:0 auto;padding:24px 16px}}
h1{{font-size:26px;margin-bottom:2px}} h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:4px;font-size:19px}}
table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow-x:auto}} td,th{{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left;vertical-align:top}}
.muted{{color:var(--muted)}} .cite{{font-family:monospace;font-size:11px;background:rgba(31,95,191,.12);color:var(--accent);padding:0 5px;border-radius:8px;text-decoration:none}}
.kpi{{display:flex;gap:22px;flex-wrap:wrap}} .kpi div{{min-width:120px}} .kpi b{{display:block;font-size:22px}}
.warn{{border-left:3px solid #a86400;padding:6px 12px;background:rgba(168,100,0,.08)}} blockquote{{margin:4px 0 10px 12px;color:var(--muted)}} .ev{{font-size:12.5px}}
</style></head><body>
<p class="muted">DRISHTI · evidence-first intelligence report · generated {E(utcnow())} by {E(user)}</p>
<h1>{E(c['name'])}</h1><p class="muted">Case <code>{E(case)}</code> · domain {E(c['domain'])}</p>
<div class="kpi"><div><b>{len(docs)}</b>documents</div><div><b>{len(ents)}</b>resolved entities</div><div><b>{len(claims)}</b>claims</div>
<div><b>{len(view.edges)}</b>evidence edges</div><div><b>{an['modularity'] or 0:.3f}</b>modularity</div><div><b>{len(con)}</b>contradictions</div></div>
<p class="warn">This report presents documented claims, their sources and computed structure. Belief scores are calibrated extraction
confidences aggregated over <em>independent</em> origins — they are not probabilities of guilt or truth. Network prominence and pattern leads
are prompts for review, not findings of wrongdoing. Every finding below links to its exact source passage.</p>
<h2>1. Network overview</h2>{_svg(U, comm, names)}
<p class="muted">Top nodes by betweenness; colour = Louvain community; edge opacity = aggregated belief. Entity types: {E(', '.join(f'{k.lower()} {v}' for k, v in by_type.most_common()))}.</p>
<h2>2. Well-evidenced relationships</h2><p class="muted">Belief ≥ 0.75, ordered by number of independent origins (independent / documents).</p>
<table><tr><th>Subject</th><th>Relation</th><th>Object</th><th>Belief</th><th>Indep./docs</th><th>Valid</th><th>Modality</th><th>Evidence</th></tr>{findings}</table>
<h2>3. Structural brokers</h2><table><tr><th>Entity</th><th>Betweenness</th><th>PageRank</th><th>Degree</th><th>Articulation point</th></tr>{brokers}</table>
<h2>4. Communities</h2><ul>{comms or '<li>None of size ≥ 3.</li>'}</ul>
<h2>5. Leads for review</h2><ul>{leads or '<li>No patterns detected.</li>'}</ul>
<h2>6. Contradictions between sources</h2><ul>{contra or '<li>None detected.</li>'}</ul>
<h2>7. Inferred link hypotheses</h2><p class="muted">Structural link prediction (hold-out ROC-AUC {lp.get('auc_combined')}). No source states these links.</p><ul>{hyps or '<li>None.</li>'}</ul>
<h2>8. Methodology</h2><ol>
<li>Documents are hashed (SHA-256), stored immutably and parsed with a cost cascade (native text → OCR only where needed); every passage keeps character offsets.</li>
<li>Entities are extracted by validated identifier patterns (PAN/GSTIN checksums, CIN, IFSC, vehicle, phone) and grammar rules with case-level propagation.</li>
<li>Relations are extracted against a closed schema with direction, negation/hedging/attribution detection and valid-time intervals; each claim stores its evidence span, which is verified against the source text.</li>
<li>Identities are resolved with blocking + MinHash LSH, a learned pair model and constrained average-linkage clustering (identifier conflicts are hard cannot-links); ambiguous pairs go to human review.</li>
<li>The graph is a bitemporal projection of claims; belief aggregates calibrated confidences by noisy-OR across independent origin groups, so copies of one source count once.</li>
<li>Analytics: evidence-weighted betweenness/PageRank, Louvain communities, k-shortest evidence paths, link prediction, resilience, and explicit-threshold pattern detectors.</li></ol>
<h2>9. Limitations</h2><ul><li>Extraction misses phrasings outside its lexicon/training data; absence of a claim is not evidence of absence.</li>
<li>Unreviewed claims are machine-extracted. Analyst-confirmed claims are marked as such.</li><li>Source independence is estimated (version chains, near-duplicate text, attribution cues) and may miss hidden common origins.</li>
<li>Entity resolution can still merge namesakes or split one person; see the review queue.</li></ul>
<h2>10. Source register</h2><table><tr><th>Document</th><th>Type</th><th>Reliability</th><th>Published</th><th>SHA-256</th><th>Quality flags</th></tr>{reg}</table>
<p class="muted">Audit chain: {'valid' if chain['valid'] else 'BROKEN'} · {chain['entries']} entries · head <code>{E(str(chain.get('head', ''))[:24])}…</code></p>
<h2>Appendix — cited evidence</h2>{appendix}
</body></html>"""
