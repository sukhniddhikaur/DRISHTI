"""Structural & temporal pattern detectors — investigative leads, not verdicts.

Each detector returns findings with the exact claim IDs that make up the
pattern, a plain-language description, and the parameters used. Thresholds are
explicit arguments so an analyst can see (and change) what was flagged and why.

Money flow (on TRANSFERRED_TO claims, one claim = one transaction):
  * temporal cycles      A→B→…→A with non-decreasing timestamps inside a window
                         and amount retention ≥ r per hop (layering signature)
  * pass-through         funds in from ≥k sources, ≥x% forwarded within h hours
  * structuring          ≥n transfers just below a reporting threshold in a window
Communication (CALLED claims):
  * burst around event   daily call volume z-score around an event date
Identity / corporate:
  * shared identifiers   one phone/account/vehicle tied to ≥2 resolved persons
  * director interlock   suppliers awarded by the same buyer sharing a director
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from statistics import mean, pstdev
from typing import Any

from drishti.store.db import Store


def _d(s: str | None) -> date | None:
    try:
        return date.fromisoformat(s[:10]) if s else None
    except ValueError:
        return None


def _txns(store: Store, case_id: str) -> list[dict[str, Any]]:
    out = []
    for c in store.claims(case_id, predicate="TRANSFERRED_TO"):
        if c["status"] == "rejected" or not (c["subject_entity_id"] and c["object_entity_id"]):
            continue
        out.append({"claim": c["claim_id"], "src": c["subject_entity_id"], "dst": c["object_entity_id"],
                    "date": _d(c["valid_from"]), "amount": (c["attributes"] or {}).get("amount_inr") or 0})
    return [t for t in out if t["date"]]


def money_cycles(store: Store, case_id: str, max_len: int = 5, window_days: int = 30, retention: float = 0.7,
                 limit: int = 50) -> list[dict[str, Any]]:
    tx = _txns(store, case_id)
    out_edges: dict[str, list[dict]] = defaultdict(list)
    for t in tx:
        out_edges[t["src"]].append(t)
    for v in out_edges.values():
        v.sort(key=lambda t: t["date"])
    found, seen = [], set()

    def dfs(start, node, path, t_prev, amt_prev, t0):
        if len(found) >= limit:
            return
        for t in out_edges.get(node, []):
            if t["date"] < t_prev or (t["date"] - t0).days > window_days:
                continue
            if amt_prev and not (retention * amt_prev <= t["amount"] <= amt_prev * 1.05):
                continue  # layering keeps amounts in a narrow band (minus fees)
            if t["dst"] == start and len(path) >= 2:
                cyc = path + [t]
                key = frozenset(x["claim"] for x in cyc)
                if key not in seen:
                    seen.add(key)
                    found.append(cyc)
                continue
            if t["dst"] in {x["src"] for x in path} or len(path) + 1 >= max_len:
                continue
            dfs(start, t["dst"], path + [t], t["date"], t["amount"], t0)

    for s in list(out_edges):
        for t in out_edges[s]:
            dfs(s, t["dst"], [t], t["date"], t["amount"], t["date"])
    res = []
    for cyc in found:
        accounts = [cyc[0]["src"]] + [t["dst"] for t in cyc]
        res.append({"pattern": "temporal_money_cycle", "accounts": accounts, "claims": [t["claim"] for t in cyc],
                    "start": cyc[0]["date"].isoformat(), "end": cyc[-1]["date"].isoformat(),
                    "amounts": [t["amount"] for t in cyc],
                    "retention": round(cyc[-1]["amount"] / max(1, cyc[0]["amount"]), 3),
                    "description": f"Funds left and returned to the same account through {len(cyc)} hops in "
                                   f"{(cyc[-1]['date'] - cyc[0]['date']).days} days, retaining "
                                   f"{cyc[-1]['amount'] / max(1, cyc[0]['amount']):.0%} of the amount.",
                    "params": {"max_len": max_len, "window_days": window_days, "retention": retention}})
    return sorted(res, key=lambda r: (-len(r["claims"]), r["start"]))


def pass_through(store: Store, case_id: str, min_sources: int = 3, forward_ratio: float = 0.7, hours: int = 72,
                 window_days: int = 3) -> list[dict[str, Any]]:
    """Sliding window: ≥min_sources distinct senders credit an account within `window_days`,
    and ≥forward_ratio of that inflow leaves within `hours` of the last credit in the window."""
    tx = _txns(store, case_id)
    inc, outg = defaultdict(list), defaultdict(list)
    for t in tx:
        inc[t["dst"]].append(t)
        outg[t["src"]].append(t)
    res = []
    for acct, ins in inc.items():
        ins.sort(key=lambda t: t["date"])
        best = None
        i = 0
        for j in range(len(ins)):
            while (ins[j]["date"] - ins[i]["date"]).days > window_days:
                i += 1
            win = ins[i:j + 1]
            srcs = {t["src"] for t in win}
            if len(srcs) < min_sources:
                continue
            total_in = sum(t["amount"] for t in win)
            last_in = win[-1]["date"]
            fwd = [t for t in outg.get(acct, []) if win[0]["date"] <= t["date"] <= last_in + timedelta(hours=hours)]
            total_out = sum(t["amount"] for t in fwd)
            if total_in and forward_ratio <= total_out / total_in <= 1.1 and (best is None or len(srcs) > len(best[1])):
                best = (win, srcs, fwd, total_in, total_out)
        if best:
            win, srcs, fwd, total_in, total_out = best
            res.append({"pattern": "pass_through", "account": acct, "sources": sorted(srcs), "in_amount": total_in,
                        "forwarded_amount": total_out, "ratio": round(total_out / total_in, 3),
                        "window": [win[0]["date"].isoformat(), win[-1]["date"].isoformat()],
                        "claims": [t["claim"] for t in win + fwd],
                        "description": f"Received from {len(srcs)} accounts within {window_days} days and forwarded "
                                       f"{total_out / total_in:.0%} within {hours}h of the last credit.",
                        "params": {"min_sources": min_sources, "forward_ratio": forward_ratio, "hours": hours, "window_days": window_days}})
    return res


def structuring(store: Store, case_id: str, thresholds: tuple[int, ...] = (50_000, 200_000, 1_000_000), band: float = 0.1,
                min_count: int = 3, window_days: int = 7) -> list[dict[str, Any]]:
    tx = _txns(store, case_id)
    res = []
    for thr in thresholds:
        near = [t for t in tx if thr * (1 - band) <= t["amount"] < thr]
        by_acct = defaultdict(list)
        for t in near:
            by_acct[t["src"]].append(t)
        for acct, ts in by_acct.items():
            ts.sort(key=lambda t: t["date"])
            i = 0
            for j in range(len(ts)):
                while (ts[j]["date"] - ts[i]["date"]).days > window_days:
                    i += 1
                if j - i + 1 >= min_count:
                    grp = ts[i:j + 1]
                    res.append({"pattern": "structuring", "account": acct, "threshold": thr, "count": len(grp),
                                "claims": [t["claim"] for t in grp], "total": sum(t["amount"] for t in grp),
                                "description": f"{len(grp)} transfers between ₹{int(thr * (1 - band)):,} and ₹{thr:,} "
                                               f"within {window_days} days (just under a reporting threshold).",
                                "params": {"threshold": thr, "band": band, "min_count": min_count, "window_days": window_days}})
                    break
    return res


def call_bursts(store: Store, case_id: str, event_dates: list[str] | None = None, radius: int = 2, z: float = 2.0) -> list[dict[str, Any]]:
    calls = [c for c in store.claims(case_id, predicate="CALLED") if c["valid_from"]]
    if not calls:
        return []
    per_day = defaultdict(list)
    for c in calls:
        per_day[c["valid_from"][:10]].append(c)
    days = sorted(per_day)
    d0, d1 = _d(days[0]), _d(days[-1])
    series = {(d0 + timedelta(i)).isoformat(): len(per_day.get((d0 + timedelta(i)).isoformat(), [])) for i in range((d1 - d0).days + 1)}
    mu, sd = mean(series.values()), pstdev(series.values()) or 1.0
    if event_dates is None:
        event_dates = [e["date"] for e in store.events(case_id) if e["date"] and e["event_type"] not in ("CALL", "TRANSACTION")]
    res = []
    for ed in sorted(set(event_dates)):
        e = _d(ed)
        if not e:
            continue
        win = [(e + timedelta(k)).isoformat() for k in range(-radius, radius + 1)]
        vol = sum(series.get(d, 0) for d in win)
        zz = (vol / len(win) - mu) / sd
        if zz >= z:
            cl = [c["claim_id"] for d in win for c in per_day.get(d, [])]
            res.append({"pattern": "call_burst_near_event", "event_date": ed, "window": [win[0], win[-1]], "calls": vol,
                        "z_score": round(zz, 2), "baseline_per_day": round(mu, 2), "claims": cl[:200],
                        "description": f"{vol} calls within ±{radius} days of {ed} (z={zz:.1f} vs daily baseline {mu:.1f}).",
                        "params": {"radius": radius, "z": z}})
    return res


def shared_identifiers(store: Store, case_id: str) -> list[dict[str, Any]]:
    res = []
    preds = ("USES_PHONE", "HOLDS_ACCOUNT", "OWNS_VEHICLE")
    by_obj: dict[str, list[dict]] = defaultdict(list)
    for c in store.claims(case_id, predicate=list(preds)):
        if c["status"] != "rejected" and c["subject_entity_id"] and c["object_entity_id"] and c["modality"] != "negated":
            by_obj[c["object_entity_id"]].append(c)
    names = {e["entity_id"]: e["canonical_name"] for e in store.entities(case_id)}
    for obj, cs in by_obj.items():
        subs = {c["subject_entity_id"] for c in cs}
        if len(subs) >= 2:
            res.append({"pattern": "shared_identifier", "identifier": obj, "identifier_value": names.get(obj),
                        "holders": sorted(subs), "holder_names": sorted(names.get(s, s) for s in subs),
                        "claims": [c["claim_id"] for c in cs],
                        "description": f"{names.get(obj)} is linked to {len(subs)} distinct resolved entities — either a "
                                       f"shared instrument or an unresolved duplicate identity; review both."})
    # also: same strong identifier attached to two distinct entities (ER review)
    owners = defaultdict(set)
    for e in store.entities(case_id):
        for t, vals in (e["identifiers"] or {}).items():
            for v in vals:
                owners[(t, v)].add(e["entity_id"])
    for (t, v), es in owners.items():
        if len(es) > 1 and t in ("PHONE", "EMAIL", "ACCOUNT"):
            res.append({"pattern": "shared_identifier_attribute", "identifier_type": t, "identifier_value": v,
                        "holders": sorted(es), "holder_names": sorted(names.get(x, x) for x in es), "claims": [],
                        "description": f"{t} {v} appears on {len(es)} distinct entity profiles."})
    return res


def director_interlocks(store: Store, case_id: str) -> list[dict[str, Any]]:
    awards = [c for c in store.claims(case_id, predicate="AWARDED_CONTRACT_TO")
              if c["status"] != "rejected" and c["modality"] != "negated" and c["subject_entity_id"] and c["object_entity_id"]]
    directors = defaultdict(set)
    dclaims = defaultdict(list)
    for c in store.claims(case_id, predicate="DIRECTOR_OF"):
        if c["status"] != "rejected" and c["modality"] != "negated" and c["subject_entity_id"] and c["object_entity_id"]:
            directors[c["object_entity_id"]].add(c["subject_entity_id"])
            dclaims[(c["subject_entity_id"], c["object_entity_id"])].append(c["claim_id"])
    by_buyer = defaultdict(list)
    for a in awards:
        by_buyer[a["subject_entity_id"]].append(a)
    names = {e["entity_id"]: e["canonical_name"] for e in store.entities(case_id)}
    res = []
    for buyer, aw in by_buyer.items():
        sup = sorted({a["object_entity_id"] for a in aw})
        for i, x in enumerate(sup):
            for y in sup[i + 1:]:
                common = directors[x] & directors[y]
                if common:
                    cl = [a["claim_id"] for a in aw if a["object_entity_id"] in (x, y)]
                    for p in common:
                        cl += dclaims[(p, x)] + dclaims[(p, y)]
                    res.append({"pattern": "director_interlock", "buyer": buyer, "suppliers": [x, y], "shared_directors": sorted(common),
                                "claims": cl,
                                "description": f"{names.get(x)} and {names.get(y)} both received contracts from {names.get(buyer)} "
                                               f"and share director(s) {', '.join(names.get(p, p) for p in common)}."})
    return res


def run_all(store: Store, case_id: str, thresholds: tuple[int, ...] = (50_000, 200_000, 1_000_000)) -> dict[str, list]:
    return {"money_cycles": money_cycles(store, case_id), "pass_through": pass_through(store, case_id),
            "structuring": structuring(store, case_id, thresholds), "call_bursts": call_bursts(store, case_id),
            "shared_identifiers": shared_identifiers(store, case_id), "director_interlocks": director_interlocks(store, case_id)}
