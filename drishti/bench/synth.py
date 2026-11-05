"""Synthetic intelligence corpus with exact ground truth.

Generates a self-consistent fictional world — companies, people, government
buyers, contracts, directorships, litigation, criminal cases, phones, bank
accounts, call and transaction records — then renders it into heterogeneous
documents (filings, tender notices, annual reports, judgments, news, FIRs,
CDR/bank CSVs, an XLSX register, a DOCX chargesheet, a scanned PDF) while
recording the exact character span and gold entity of every mention and every
gold relation (with modality and valid time).

Deliberate difficulty:
  * name variants (legal-form spellings, "Infra", ALL CAPS, initials,
    transliteration), missing suffixes, OCR-style noise
  * hard negatives: same brand different business, same name different person,
    same name different legal form
  * co-occurrence distractors, negated and hedged statements
  * syndicated news copies (source-independence test), stale statements after a
    resignation (temporal contradiction), conflicting contract amounts
  * planted money cycle, structuring, pass-through mule account, director
    interlock, call burst around an incident
  * template split: "test" templates never appear in the training world

Every name is fictional; no real person or company is represented.
"""
from __future__ import annotations

import csv
import io
import json
import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from drishti.extract.identifiers import gstin_checksum

FIRST = ("Aarav Abhishek Aditya Ajay Akash Alok Amit Anand Anil Ankit Arjun Arun Arvind Ashok Deepak Dinesh Farhan Gaurav "
         "Harish Imran Irfan Karan Kunal Mahesh Manish Manoj Mohan Mukesh Naveen Nikhil Pankaj Prakash Pradeep Rahul Rajesh "
         "Rakesh Ramesh Ravi Rohit Sachin Sameer Sandeep Sanjay Santosh Shankar Sunil Suresh Tarun Varun Vijay Vikas Vikram "
         "Vinod Vishal Yusuf Faisal Salim Javed Gurpreet Venkatesh Srinivas Lalit Hemant Naresh Girish Jitendra Bhavesh").split()
FIRST_F = "Anjali Anita Ayesha Deepika Divya Geeta Jaya Kavita Kiran Meena Neha Nisha Pooja Priya Rekha Ritu Shalini Shweta Sneha Sunita Swati Uma Usha".split()
MIDDLE = "Kumar Prasad Chandra Lal Nath Singh Mohan Raj Kant".split()
LAST = ("Sharma Verma Gupta Agarwal Mehta Shah Patel Desai Joshi Kulkarni Deshpande Patil Pawar Jadhav Yadav Mishra Tiwari "
        "Pandey Dubey Srivastava Saxena Malhotra Kapoor Khanna Bhatia Chopra Arora Sethi Reddy Rao Naidu Iyer Nair Menon Pillai "
        "Das Ghosh Banerjee Mukherjee Chatterjee Bose Sen Khan Qureshi Ansari Siddiqui Shaikh Sinha Thakur Chauhan Rathore").split()
TRANSLIT = {"Mohammad": ["Mohammed", "Muhammad"], "Srinivas": ["Shrinivas"], "Venkatesh": ["Venkatesh"], "Yusuf": ["Yousuf"],
            "Faisal": ["Faisel"], "Javed": ["Jawed"], "Shaikh": ["Sheikh"], "Qureshi": ["Kureshi"]}
BRANDS = ("Sunrise Zenith Orion Shakti Vardhan Kaveri Trident Nirmal Aakash Meridian Ganga Sapphire Kalinga Deccan Everest "
          "Lotus Indus Pinnacle Vertex Navkar Suvarna Tejas Sahyadri Narmada Amrit Prabhat Vishwa Aditi Keystone Bluewave").split()
SECTORS = "Infrastructure Constructions Builders Traders Enterprises Logistics Technologies Developers Exports Industries Solutions".split()
CITIES = [("Pune", "MH", "27"), ("Mumbai", "MH", "27"), ("Nagpur", "MH", "27"), ("Ahmedabad", "GJ", "24"), ("Jaipur", "RJ", "08"),
          ("Lucknow", "UP", "09"), ("Patna", "BR", "10"), ("Bhopal", "MP", "23"), ("Hyderabad", "TS", "36"), ("Chennai", "TN", "33")]
GOVT = ["Public Works Department", "Municipal Corporation of Pune", "National Highways Authority of India", "Pune Metropolitan Region Development Authority",
        "Maharashtra Jeevan Pradhikaran Board", "Department of Water Resources", "Nagpur Improvement Trust Board", "Rajasthan Housing Board"]
BANKS = ["State Bank of India", "Punjab National Bank", "Canara Bank", "Bank of Baroda", "Union Bank of India"]
IFSC = {"State Bank of India": "SBIN", "Punjab National Bank": "PUNB", "Canara Bank": "CNRB", "Bank of Baroda": "BARB", "Union Bank of India": "UBIN"}
POLICE = ["Kotwali Police Station", "Gandhi Maidan Police Station", "Shivajinagar Police Station", "Sadar Police Station"]


# ============================================================================ world model
@dataclass
class GEntity:
    gid: str
    type: str
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class GRel:
    subj: str
    pred: str
    obj: str
    valid_from: str | None = None
    valid_to: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class World:
    entities: dict[str, GEntity] = field(default_factory=dict)
    rels: list[GRel] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)       # gang id -> person gids
    planted: dict[str, Any] = field(default_factory=dict)

    def add(self, e: GEntity) -> GEntity:
        self.entities[e.gid] = e
        return e

    def of(self, t: str) -> list[GEntity]:
        return [e for e in self.entities.values() if e.type == t]


def _pan(rng: random.Random, holder: str, initial: str) -> str:
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(L) for _ in range(3)) + holder + initial.upper() + f"{rng.randint(0, 9999):04d}" + rng.choice(L)


def _cin(rng: random.Random, state: str, year: int) -> str:
    return f"U{rng.randint(10000, 99999)}{state}{year}PTC{rng.randint(0, 999999):06d}"


def _gstin(state_code: str, pan: str) -> str:
    base = f"{state_code}{pan}1Z"
    return base + gstin_checksum(base)


def _phone(rng: random.Random) -> str:
    return rng.choice("789") + "".join(str(rng.randint(0, 9)) for _ in range(9))


def _d(x: date) -> str:
    return x.isoformat()


def build_world(seed: int = 42, n_orgs: int = 22, n_people: int = 34, n_gangs: int = 3, gang_size: int = 6) -> World:
    rng = random.Random(seed)
    w = World()
    brands = rng.sample(BRANDS, k=min(len(BRANDS), n_orgs))
    used_names: set[str] = set()
    # ---- organisations (with hard negatives: same brand different sector; same name different legal form)
    for i in range(n_orgs):
        brand = brands[i % len(brands)]
        sector = rng.choice(SECTORS)
        city, st, sc = rng.choice(CITIES)
        name = f"{brand} {sector}"
        form = "LLP" if rng.random() < 0.08 else "Pvt Ltd"
        if name in used_names:
            sector = rng.choice([s for s in SECTORS if s != sector])
            name = f"{brand} {sector}"
        used_names.add(name)
        pan = _pan(rng, "C", brand[0])
        w.add(GEntity(f"O{i:02d}", "ORGANISATION", name, {"form": form, "city": city, "state": st, "cin": _cin(rng, st, rng.randint(1998, 2019)),
                                                          "pan": pan, "gstin": _gstin(sc, pan), "brand": brand, "sector": sector}))
    orgs = w.of("ORGANISATION")
    # hard negative: same brand, different sector
    o0 = orgs[0]
    alt = next(s for s in SECTORS if s != o0.attrs["sector"])
    city, st, sc = rng.choice(CITIES)
    pan = _pan(rng, "C", o0.attrs["brand"][0])
    w.add(GEntity(f"O{n_orgs:02d}", "ORGANISATION", f"{o0.attrs['brand']} {alt}",
                  {"form": "Pvt Ltd", "city": city, "state": st, "cin": _cin(rng, st, 2015), "pan": pan, "gstin": _gstin(sc, pan),
                   "brand": o0.attrs["brand"], "sector": alt, "hard_negative_of": o0.gid}))
    # hard negative: same name, different legal form
    o1 = orgs[1]
    pan = _pan(rng, "F", o1.attrs["brand"][0])
    w.add(GEntity(f"O{n_orgs + 1:02d}", "ORGANISATION", o1.name, {**o1.attrs, "form": "LLP", "cin": None, "pan": pan,
                                                                    "gstin": None, "hard_negative_of": o1.gid}))
    orgs = w.of("ORGANISATION")
    # ---- government buyers, banks, police stations, locations
    for i, g in enumerate(GOVT):
        w.add(GEntity(f"G{i:02d}", "GOVT_BODY", g))
    for i, b in enumerate(BANKS):
        w.add(GEntity(f"B{i:02d}", "ORGANISATION", b, {"bank": True}))
    w.add(GEntity("G-ROC", "GOVT_BODY", "Registrar of Companies"))
    w.add(GEntity("G-HCB", "GOVT_BODY", "High Court of Judicature at Bombay"))
    for i, p in enumerate(POLICE):
        w.add(GEntity(f"S{i:02d}", "GOVT_BODY", p))
    for c, _, _ in CITIES:
        w.add(GEntity(f"L-{c}", "LOCATION", c))
    # ---- people
    names_used: set[str] = set()
    total_people = n_people + n_gangs * gang_size
    for i in range(total_people):
        female = rng.random() < 0.25
        first = rng.choice(FIRST_F if female else FIRST)
        last = rng.choice(LAST)
        mid = rng.choice(MIDDLE) if rng.random() < 0.35 and not female else None
        full = " ".join(x for x in (first, mid, last) if x)
        if full in names_used:
            last = rng.choice(LAST)
            full = " ".join(x for x in (first, mid, last) if x)
        names_used.add(full)
        city = rng.choice(CITIES)[0]
        w.add(GEntity(f"P{i:02d}", "PERSON", full, {"first": first, "middle": mid, "last": last, "female": female,
                                                   "pan": _pan(rng, "P", last[0]), "din": f"0{rng.randint(1000000, 9999999)}",
                                                   "city": city, "alias": None}))
    people = w.of("PERSON")
    # hard negative: same full name, different person
    twin_src = people[2]
    w.add(GEntity(f"P{total_people:02d}", "PERSON", twin_src.name, {**twin_src.attrs, "pan": _pan(rng, "P", twin_src.attrs["last"][0]),
                                                                  "din": f"0{rng.randint(1000000, 9999999)}", "city": "Patna",
                                                                  "hard_negative_of": twin_src.gid}))
    people = w.of("PERSON")
    corporate_people = people[:n_people]
    gang_people = people[n_people:n_people + n_gangs * gang_size]
    companies = [o for o in orgs if not o.attrs.get("bank")]

    base = date(2019, 1, 1)
    # ---- directorships (2-3 per company), some resign
    for o in companies:
        for p in rng.sample(corporate_people, k=rng.randint(2, 3)):
            start = base + timedelta(days=rng.randint(0, 1500))
            end = start + timedelta(days=rng.randint(200, 900)) if rng.random() < 0.25 else None
            w.rels.append(GRel(p.gid, "DIRECTOR_OF", o.gid, _d(start), _d(end) if end else None))
    # director interlock: one person directs two suppliers that share a buyer (planted)
    sup_a, sup_b = companies[3], companies[4]
    inter = corporate_people[5]
    for o in (sup_a, sup_b):
        if not any(r.subj == inter.gid and r.obj == o.gid and r.pred == "DIRECTOR_OF" for r in w.rels):
            w.rels.append(GRel(inter.gid, "DIRECTOR_OF", o.gid, _d(base + timedelta(days=300)), None))
    # ---- subsidiaries
    for child, parent in [(companies[6], companies[7]), (companies[8], companies[7]), (companies[9], companies[10])]:
        w.rels.append(GRel(child.gid, "SUBSIDIARY_OF", parent.gid, _d(base + timedelta(days=rng.randint(0, 800))), None))
    # ---- contract awards
    buyers = w.of("GOVT_BODY")[:len(GOVT)]
    awards = []
    for k in range(12):
        buyer = buyers[k % 5]
        sup = rng.choice(companies[:16])
        dt = base + timedelta(days=rng.randint(400, 2000))
        amt = rng.choice([45, 82, 120, 260, 375, 520, 910]) * 100_000
        ref = f"{''.join(x[0] for x in buyer.name.split() if x[0].isupper())}/{dt.year}/{rng.randint(100, 999)}"
        awards.append(GRel(buyer.gid, "AWARDED_CONTRACT_TO", sup.gid, _d(dt), None, {"amount": amt, "ref": ref}))
    awards.append(GRel(buyers[0].gid, "AWARDED_CONTRACT_TO", sup_a.gid, "2023-02-14", None, {"amount": 31_000_000, "ref": "PWD/2023/417"}))
    awards.append(GRel(buyers[0].gid, "AWARDED_CONTRACT_TO", sup_b.gid, "2023-03-02", None, {"amount": 28_500_000, "ref": "PWD/2023/452"}))
    w.rels += awards
    w.planted["interlock"] = {"person": inter.gid, "suppliers": [sup_a.gid, sup_b.gid], "buyer": buyers[0].gid}
    # ---- acquisitions, litigation, shareholding, locations
    w.rels.append(GRel(companies[11].gid, "ACQUIRED", companies[12].gid, "2022-08-19", None))
    w.rels.append(GRel(companies[13].gid, "FILED_CASE_AGAINST", companies[14].gid, "2021-11-03", None, {"case": "W.P.(C) No. 4521 of 2021"}))
    w.rels.append(GRel(corporate_people[8].gid, "FILED_CASE_AGAINST", companies[2].gid, "2022-01-17", None, {"case": "CS(COMM) No. 88 of 2022"}))
    for o in companies:
        w.rels.append(GRel(o.gid, "LOCATED_IN", f"L-{o.attrs['city']}"))
    for p in rng.sample(corporate_people, 6):
        w.rels.append(GRel(p.gid, "SHAREHOLDER_OF", rng.choice(companies).gid))

    # ---- investigation world: gangs, phones, accounts, vehicles, FIRs
    for g in range(n_gangs):
        members = gang_people[g * gang_size:(g + 1) * gang_size]
        w.groups[f"gang{g}"] = [m.gid for m in members]
        for m in members:
            m.attrs["alias"] = rng.choice(["Pappu", "Guddu", "Bablu", "Chhotu", "Munna", "Raju", "Tinku", "Sonu", "Bunty", "Lucky"]) if rng.random() < 0.5 else None
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                if rng.random() < 0.45:
                    w.rels.append(GRel(members[a].gid, "ASSOCIATE_OF", members[b].gid))
    pid = 0
    for p in gang_people + rng.sample(corporate_people, 4):
        ph = w.add(GEntity(f"PH{pid:03d}", "PHONE", "+91" + _phone(rng)))
        w.rels.append(GRel(p.gid, "USES_PHONE", ph.gid))
        p.attrs["phone_gid"] = ph.gid
        pid += 1
    # shared phone between two gang members (shared-identifier pattern)
    shared_holder = gang_people[1]
    w.rels.append(GRel(gang_people[3].gid, "USES_PHONE", shared_holder.attrs["phone_gid"]))
    w.planted["shared_phone"] = {"phone": shared_holder.attrs["phone_gid"], "holders": [shared_holder.gid, gang_people[3].gid]}
    acc_owner: dict[str, str] = {}
    aid = 0
    holders = gang_people + companies[3:6] + rng.sample(corporate_people, 5)
    for h in holders:
        bank = rng.choice(BANKS)
        acc = w.add(GEntity(f"AC{aid:03d}", "ACCOUNT", "".join(str(rng.randint(0, 9)) for _ in range(12)), {"bank": bank}))
        w.rels.append(GRel(h.gid, "HOLDS_ACCOUNT", acc.gid))
        h.attrs["account_gid"] = acc.gid
        acc_owner[acc.gid] = h.gid
        aid += 1
    for k, p in enumerate(gang_people[::2]):
        v = w.add(GEntity(f"V{k:02d}", "VEHICLE", f"{rng.choice(['MH', 'BR', 'UP', 'RJ'])}{rng.randint(1, 40):02d}{rng.choice(['AB', 'CK', 'DX', 'PQ'])}{rng.randint(1000, 9999)}"))
        w.rels.append(GRel(p.gid, "OWNS_VEHICLE", v.gid))
    firs = []
    for g, members in w.groups.items():
        for k in range(2 if g != "gang2" else 1):
            dt = date(2024, 1, 1) + timedelta(days=rng.randint(0, 300))
            fir = w.add(GEntity(f"F-{g}-{k}", "CASE", f"FIR-{rng.randint(10, 999)}/{dt.year}",
                                {"date": _d(dt), "station": rng.choice(POLICE), "sections": rng.choice(["u/s 420, 406 IPC", "u/s 379, 411 IPC", "u/s 318(4) BNS"])}))
            firs.append(fir)
            acc = rng.sample(members, 3)
            for a in acc:
                w.rels.append(GRel(a, "ACCUSED_IN", fir.gid, _d(dt)))
            comp = people[-1] if not firs[:-1] else rng.choice(corporate_people)
            w.rels.append(GRel(comp.gid, "COMPLAINANT_IN", fir.gid, _d(dt)))
            fir.attrs["accused"], fir.attrs["complainant"] = acc, comp.gid
    w.planted["firs"] = [f.gid for f in firs]
    # ---- calls: dense within gangs, sparse across; burst near first FIR
    calls = []
    start = date(2024, 1, 1)
    all_ph = [p.attrs["phone_gid"] for p in gang_people]
    for _ in range(420):
        g = rng.choice(list(w.groups))
        mem = w.groups[g]
        a, b = rng.sample(mem, 2)
        if rng.random() < 0.12:
            b = rng.choice(gang_people).gid
            if b == a:
                continue
        dt = start + timedelta(days=rng.randint(0, 330), minutes=rng.randint(0, 1439))
        calls.append((w.entities[a].attrs["phone_gid"], w.entities[b].attrs["phone_gid"], dt, rng.randint(5, 600)))
    burst_day = date.fromisoformat(firs[0].attrs["date"])
    g0 = w.groups["gang0"]
    for _ in range(70):
        a, b = rng.sample(g0, 2)
        calls.append((w.entities[a].attrs["phone_gid"], w.entities[b].attrs["phone_gid"], burst_day + timedelta(days=rng.randint(-1, 1)), rng.randint(5, 120)))
    w.planted["call_burst"] = firs[0].attrs["date"]
    w.planted["calls"] = calls
    # ---- transactions: background + planted cycle, structuring, pass-through
    txns = []
    accts = [h.attrs["account_gid"] for h in holders]
    for _ in range(110):
        a, b = rng.sample(accts, 2)
        txns.append((a, b, date(2024, 1, 1) + timedelta(days=rng.randint(0, 330)), rng.choice([12000, 18500, 25000, 7300, 64000, 91000, 150000])))
    cyc = [gang_people[0].attrs["account_gid"], companies[3].attrs["account_gid"], gang_people[7].attrs["account_gid"],
           corporate_people[0].attrs.get("account_gid") or holders[-1].attrs["account_gid"]]
    d0, amt = date(2024, 5, 6), 1_850_000
    cycle_txn = []
    for i in range(len(cyc)):
        a, b = cyc[i], cyc[(i + 1) % len(cyc)]
        t = (a, b, d0 + timedelta(days=2 * i), amt)
        txns.append(t)
        cycle_txn.append(t)
        amt = int(amt * 0.96)
    w.planted["money_cycle"] = cyc
    smurf = gang_people[2].attrs["account_gid"]
    for k in range(4):
        txns.append((smurf, gang_people[8].attrs["account_gid"], date(2024, 7, 1) + timedelta(days=k), 46_000 + 900 * k))
    w.planted["structuring"] = smurf
    mule = gang_people[13].attrs["account_gid"]
    for k, src in enumerate([gang_people[12], gang_people[14], gang_people[15], companies[5]]):
        txns.append((src.attrs["account_gid"], mule, date(2024, 9, 10) + timedelta(days=k % 2), 210_000))
    txns.append((mule, gang_people[16].attrs["account_gid"], date(2024, 9, 12), 640_000))
    txns.append((mule, gang_people[17].attrs["account_gid"], date(2024, 9, 12), 150_000))
    w.planted["pass_through"] = mule
    w.planted["txns"] = txns
    for a, b, d, amt in txns:
        w.rels.append(GRel(a, "TRANSFERRED_TO", b, _d(d), _d(d), {"amount": amt}))
    for a, b, d, dur in calls:
        w.rels.append(GRel(a, "CALLED", b, _d(d if isinstance(d, date) else d), None, {"duration": dur}))
    return w


# ============================================================================ rendering
@dataclass
class GMention:
    start: int
    end: int
    label: str
    gid: str


@dataclass
class GDocRel:
    subj: str
    pred: str
    obj: str
    modality: str
    subj_span: tuple[int, int]
    obj_span: tuple[int, int]
    sentence: tuple[int, int]
    valid_from: str | None = None
    valid_to: str | None = None


@dataclass
class GDoc:
    name: str
    doc_type: str
    reliability: str
    publication_date: str | None
    text: str = ""
    mentions: list[GMention] = field(default_factory=list)
    rels: list[GDocRel] = field(default_factory=list)
    mime: str = "text/plain"
    data: bytes | None = None
    flags: dict[str, Any] = field(default_factory=dict)


class Writer:
    def __init__(self, doc: GDoc, world: World, rng: random.Random, split: str, noise: float = 0.0):
        self.d, self.w, self.rng, self.split, self.noise = doc, world, rng, split, noise
        self.buf: list[str] = []
        self.n = 0

    def t(self, s: str) -> None:
        self.buf.append(s)
        self.n += len(s)

    def para(self) -> None:
        self.t("\n\n")

    def m(self, gid: str, surface: str | None = None, label: str | None = None) -> tuple[int, int]:
        e = self.w.entities[gid]
        s = surface or self.surface(e)
        a = self.n
        self.t(s)
        self.d.mentions.append(GMention(a, a + len(s), label or e.type, gid))
        return a, a + len(s)

    # ------------------------------------------------------------------ surface variants
    def surface(self, e: GEntity) -> str:
        r = self.rng.random()
        if e.type == "ORGANISATION" and not e.attrs.get("bank"):
            name, form = e.name, e.attrs.get("form", "Pvt Ltd")
            if form == "LLP":
                return f"{name} LLP"
            if r < 0.3:
                return f"{name} Pvt Ltd"
            if r < 0.5:
                return f"{name} Private Limited"
            if r < 0.62:
                return f"{name} Pvt. Ltd."
            if r < 0.7:
                return f"{name} PRIVATE LIMITED".upper()
            if r < 0.8 and e.attrs.get("sector") == "Infrastructure":
                return f"{e.attrs['brand']} Infra Pvt Ltd"
            if r < 0.88 and e.attrs.get("sector") in ("Technologies", "Constructions", "Industries"):
                short = {"Technologies": "Tech", "Constructions": "Construction", "Industries": "Inds"}[e.attrs["sector"]]
                return f"{e.attrs['brand']} {short} Pvt Ltd"
            return f"{name} Pvt Ltd"
        if e.type == "PERSON":
            f, mid, last = e.attrs["first"], e.attrs["middle"], e.attrs["last"]
            if f in TRANSLIT and r < 0.4:
                f = self.rng.choice(TRANSLIT[f])
            if last in TRANSLIT and r > 0.6:
                last = self.rng.choice(TRANSLIT[last])
            r2 = self.rng.random()
            if r2 < 0.12 and mid:
                return f"{f[0]}. {mid[0]}. {last}"
            if r2 < 0.2:
                return f"{f[0]}. {last}" if not mid else f"{f} {mid[0]}. {last}"
            if r2 < 0.65 or not mid:
                return " ".join(x for x in (f, mid, last) if x) if r2 > 0.3 else f"{f} {last}"
            return f"{f} {mid} {last}"
        if e.type == "CASE":
            return e.attrs.get("surface", e.name)
        if e.type == "PHONE":
            x = e.name[3:]
            return self.rng.choice([x, x, f"+91-{x}", f"{x[:5]} {x[5:]}"])
        if e.type == "VEHICLE":
            m = re.fullmatch(r"([A-Z]{2})(\d{2})([A-Z]+)(\d{4})", e.name)
            return self.rng.choice([e.name, " ".join(m.groups()), "-".join(m.groups())]) if m else e.name
        return e.name

    def hon(self, gid: str) -> str:
        e = self.w.entities[gid]
        if e.type != "PERSON":
            return ""
        return self.rng.choice(["Ms. ", "Smt. "]) if e.attrs.get("female") else self.rng.choice(["Mr. ", "Shri ", "Mr. ", ""])

    # ------------------------------------------------------------------ templated sentences
    def sent(self, tpl: str, slots: dict[str, Any], rel: tuple[str, str, str, str] | None = None,
             rels: list[tuple[str, str, str, str]] | None = None, vf: str | None = None, vt: str | None = None) -> None:
        """tpl uses {NAME} placeholders. slots map NAME -> gid (entity) or str (literal).
        rel(s): (subj_slot, predicate, obj_slot, modality)."""
        s0 = self.n
        spans: dict[str, tuple[int, int]] = {}
        for piece in re.split(r"(\{\w+\})", tpl):
            m = re.fullmatch(r"\{(\w+)\}", piece)
            if not m:
                self.t(piece)
                continue
            k = m.group(1)
            v = slots[k]
            if isinstance(v, str) and v in self.w.entities:
                e = self.w.entities[v]
                if e.type == "PERSON" and not k.endswith("_bare"):
                    self.t(self.hon(v))
                spans[k] = self.m(v)
            else:
                self.t(str(v))
        s1 = self.n
        for r in ([rel] if rel else []) + (rels or []):
            sk, pred, ok, mod = r
            self.d.rels.append(GDocRel(slots[sk], pred, slots[ok], mod, spans[sk], spans[ok], (s0, s1), vf, vt))
        self.t(" ")

    def finish(self) -> GDoc:
        text = "".join(self.buf)
        if self.noise:
            text = ocr_noise(text, self.rng, self.noise, protect=[(m.start, m.end) for m in self.d.mentions])
        self.d.text = text
        return self.d


def ocr_noise(text: str, rng: random.Random, rate: float, protect: list[tuple[int, int]]) -> str:
    """Length-preserving character confusions outside entity spans (keeps gold offsets valid)."""
    conf = {"o": "0", "l": "1", "e": "c", "i": "l", "S": "5", "a": "o"}
    prot = bytearray(len(text))
    for a, b in protect:
        prot[a:b] = b"\x01" * (b - a)
    out = list(text)
    for i, ch in enumerate(out):
        if not prot[i] and ch in conf and rng.random() < rate:
            out[i] = conf[ch]
    return "".join(out)


def fmt_date(rng: random.Random, iso: str) -> str:
    d = date.fromisoformat(iso)
    st = rng.random()
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    if st < 0.4:
        return f"{d.day} {months[d.month - 1]} {d.year}"
    if st < 0.7:
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    if st < 0.85:
        return f"{months[d.month - 1]} {d.day}, {d.year}"
    return f"{d.day:02d}.{d.month:02d}.{d.year}"


def fmt_amt(rng: random.Random, amt: int) -> str:
    if amt >= 10_000_000 and rng.random() < 0.6:
        return f"Rs. {amt / 10_000_000:g} crore"
    if amt >= 100_000 and rng.random() < 0.6:
        return f"Rs. {amt / 100_000:g} lakh"
    s = str(amt)
    # Indian digit grouping
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
    return f"₹{head + ',' if head else ''}{tail}"


# Relation templates: (template, split). Slots: S, O, D, A, R (ref), etc.
T_DIRECTOR = [
    ("{S} (DIN: {DIN}) was appointed as a director of {O} with effect from {D}.", "train"),
    ("{S} is a director of {O}.", "train"),
    ("{S} has served on the board of {O} since {D}.", "train"),
    ("{O} appointed {S} as an additional director on {D}.", "test"),
    ("{S}, a whole-time director of {O}, signed the return.", "test"),
    ("{S} holds a directorship in {O} since {D}.", "test"),
]
T_RESIGN = [("{S} resigned as director of {O} on {D}.", "train"), ("{S} ceased to be a director of {O} on {D}.", "test")]
T_AWARD = [
    ("{B} awarded the contract to {S} on {D} for {A} under Tender No. {R}.", "train"),
    ("{S} received a contract from {B} on {D} for {A} (Tender No. {R}).", "train"),
    ("The contract under Tender No. {R} was awarded to {S} by {B} on {D} for {A}.", "test"),
    ("{B} issued a work order to {S} worth {A} dated {D} against Tender No. {R}.", "test"),
    ("{S} secured a work order from {B} valued at {A} on {D} (Tender No. {R}).", "test"),
]
T_SUBS = [("{C} is a wholly owned subsidiary of {P}.", "train"), ("{P} is the holding company of {C}.", "test"),
          ("{C}, a step-down subsidiary of {P}, reported higher revenue.", "test")]
T_ACQ = [("{A} completed the acquisition of {B} on {D}.", "train"), ("{B} was acquired by {A} on {D}.", "test")]
T_CASE = [("{A} filed a petition against {B} before the court in {K}.", "train"), ("{B} was sued by {A} in {K}.", "test")]
T_SHARE = [("{S} holds shares in {O}.", "train"), ("{S} is a promoter of {O}.", "test")]
T_LOC = [("{O} has its registered office in {L}.", "train"), ("{O} is headquartered in {L}.", "test")]
T_ACCUSED = [("{S} has been named as accused in {F}.", "train"), ("Police arrested {S} in connection with {F}.", "test")]
T_PHONE = [("{S} uses mobile number {PH}.", "train"), ("{S} was using mobile number {PH} at the time.", "test")]
T_ASSOC = [("{S} is a known associate of {S2}.", "train"), ("{S} was accompanied by {S2} on the day of the incident.", "test")]
T_VEH = [("{S} is the registered owner of vehicle {V}.", "train"), ("{S} was seen driving vehicle bearing registration {V}.", "test")]
T_ACCT = [("{S} holds account number {AC} with {BK}.", "train"), ("{S} operates bank account number {AC} at {BK}.", "test")]
DISTRACT = ["{X} and {Y} were listed in the cause list for the day.", "Representatives of {X} attended the hearing along with {Y}.",
            "The notice was also served on {X} and {Y}.", "{X} was seen at the same venue as {Y}, according to the attendance register."]


def pick(rng: random.Random, tpls: list[tuple[str, str]], split: str) -> str:
    opts = [t for t, s in tpls if split == "test" or s == "train"]
    return rng.choice(opts)


def render_corpus(w: World, seed: int = 42, split: str = "test") -> list[GDoc]:
    rng = random.Random(seed + 1000)
    docs: list[GDoc] = []
    ents = w.entities
    comps = [e for e in w.of("ORGANISATION") if not e.attrs.get("bank")]
    rels_by = lambda p: [r for r in w.rels if r.pred == p]
    dirs = rels_by("DIRECTOR_OF")

    # ---------------------------------------------------------------- corporate filings (2 companies each)
    for i in range(0, len(comps), 2):
        group = comps[i:i + 2]
        doc = GDoc(f"filing_{i // 2:02d}.txt", "corporate_filing", "official_record", "2024-03-31")
        wr = Writer(doc, w, rng, split)
        wr.t("FORM DIR-12 — PARTICULARS OF APPOINTMENT OF DIRECTORS\n\nFiled with the ")
        wr.m("G-ROC")
        wr.t("\n\n")
        for o in group:
            cin = f" (CIN {o.attrs['cin']})" if o.attrs.get("cin") else ""
            wr.sent("{O}" + cin + " filed particulars of its directors.", {"O": o.gid})
            wr.sent("{O} has its registered office in {L}.", {"O": o.gid, "L": f"L-{o.attrs['city']}"}, rel=("O", "LOCATED_IN", "L", "asserted"))
            wr.para()
            for r in [r for r in dirs if r.obj == o.gid]:
                p = ents[r.subj]
                tpl = pick(rng, T_DIRECTOR, split)
                wr.sent(tpl.replace("{S}", "{S} (PAN: " + p.attrs["pan"] + ")") if "DIN" not in tpl and rng.random() < 0.5 else tpl,
                        {"S": p.gid, "O": o.gid, "D": fmt_date(rng, r.valid_from), "DIN": p.attrs["din"]},
                        rel=("S", "DIRECTOR_OF", "O", "asserted"), vf=r.valid_from)
                if r.valid_to:
                    wr.sent(pick(rng, T_RESIGN, split), {"S": p.gid, "O": o.gid, "D": fmt_date(rng, r.valid_to)},
                            rel=("S", "DIRECTOR_OF", "O", "asserted"), vt=r.valid_to)
                wr.para()
        docs.append(wr.finish())

    # ---------------------------------------------------------------- tender award notices
    awards = rels_by("AWARDED_CONTRACT_TO")
    conflict_award = awards[0]
    for i in range(0, len(awards), 3):
        doc = GDoc(f"tender_awards_{i // 3:02d}.txt", "tender_award", "official_record", awards[i].valid_from)
        wr = Writer(doc, w, rng, split)
        wr.t("NOTICE OF AWARD — e-Procurement Portal\n\n")
        for r in awards[i:i + 3]:
            wr.sent(pick(rng, T_AWARD, split), {"B": r.subj, "S": r.obj, "D": fmt_date(rng, r.valid_from),
                                                 "A": fmt_amt(rng, r.attrs["amount"]), "R": r.attrs["ref"]},
                    rel=("B", "AWARDED_CONTRACT_TO", "S", "asserted"), vf=r.valid_from)
            if rng.random() < 0.5:
                x = rng.choice(comps)
                if x.gid != r.obj:
                    wr.sent("Bids were also received from {X}.", {"X": x.gid})
            wr.para()
        docs.append(wr.finish())

    # ---------------------------------------------------------------- annual reports
    parents = sorted({r.obj for r in rels_by("SUBSIDIARY_OF")})
    for k, par in enumerate(parents + [comps[11].gid]):
        doc = GDoc(f"annual_report_{k}.txt", "annual_report", "corporate_disclosure", f"202{3 + k % 2}-09-30")
        wr = Writer(doc, w, rng, split)
        wr.t("ANNUAL REPORT — BOARD'S REPORT FOR THE FINANCIAL YEAR 2023-24\n\n")
        wr.sent("{O} presents its annual report to the shareholders.", {"O": par})
        for r in [r for r in rels_by("SUBSIDIARY_OF") if r.obj == par]:
            wr.sent(pick(rng, T_SUBS, split), {"C": r.subj, "P": par}, rel=("C", "SUBSIDIARY_OF", "P", "asserted"))
        for r in [r for r in dirs if r.obj == par][:2]:
            wr.sent("{S} is a director of {O}.", {"S": r.subj, "O": par}, rel=("S", "DIRECTOR_OF", "O", "asserted"))
        for r in [r for r in rels_by("ACQUIRED") if r.subj == par]:
            wr.sent(pick(rng, T_ACQ, split), {"A": r.subj, "B": r.obj, "D": fmt_date(rng, r.valid_from)}, rel=("A", "ACQUIRED", "B", "asserted"))
        for r in [r for r in rels_by("SHAREHOLDER_OF")][k * 2:k * 2 + 2]:
            wr.sent(pick(rng, T_SHARE, split), {"S": r.subj, "O": r.obj}, rel=("S", "SHAREHOLDER_OF", "O", "asserted"))
        wr.para()
        docs.append(wr.finish())

    # ---------------------------------------------------------------- court judgments (with distractors + negation)
    for k, r in enumerate(rels_by("FILED_CASE_AGAINST")):
        doc = GDoc(f"judgment_{k}.txt", "court_judgment", "official_record", "2023-06-15")
        wr = Writer(doc, w, rng, split)
        wr.t("IN THE ")
        wr.m("G-HCB")
        wr.t("\n\nJUDGMENT\n\n")
        wr.sent(pick(rng, T_CASE, split), {"A": r.subj, "B": r.obj, "K": r.attrs["case"]}, rel=("A", "FILED_CASE_AGAINST", "B", "asserted"))
        for _ in range(3):
            x, y = rng.sample(comps + w.of("PERSON")[:20], 2)
            wr.sent(rng.choice(DISTRACT), {"X": x.gid, "Y": y.gid})
        # negated directorship (planted)
        nd = rng.choice(w.of("PERSON")[:30])
        no = rng.choice([c for c in comps if not any(d.subj == nd.gid and d.obj == c.gid for d in dirs)])
        wr.sent("It was submitted that {S} was never a director of {O}.", {"S": nd.gid, "O": no.gid},
                rel=("S", "DIRECTOR_OF", "O", "negated"))
        wr.para()
        docs.append(wr.finish())

    # ---------------------------------------------------------------- news (syndicated copy, hedged, stale)
    a0 = conflict_award
    story = GDoc("news_contract_a.txt", "news_report", "journalism", "2024-04-02")
    wr = Writer(story, w, rng, split)
    wr.t("Contract row: questions over award\n\n")
    wr.sent("According to the tender record, {B} awarded the contract to {S} for {A}.",
            {"B": a0.subj, "S": a0.obj, "A": fmt_amt(rng, a0.attrs["amount"] + 2_500_000)}, rel=("B", "AWARDED_CONTRACT_TO", "S", "reported"))
    hp, ho = rng.choice(w.of("PERSON")[:30]), rng.choice(comps)
    wr.sent("{S} is allegedly linked to {O} through a holding structure.", {"S": hp.gid, "O": ho.gid})
    wr.sent("{C} is reportedly a subsidiary of {P}.", {"C": comps[15].gid, "P": comps[16].gid}, rel=("C", "SUBSIDIARY_OF", "P", "hedged"))
    docs.append(wr.finish())
    # syndicated copy: same text, different outlet/date
    pre = "Wire copy - reproduced from a partner outlet\n\n"
    sh = len(pre)
    copy = GDoc("news_contract_a_syndicated.txt", "news_report", "journalism", "2024-04-03", text=pre + story.text,
                mentions=[GMention(m.start + sh, m.end + sh, m.label, m.gid) for m in story.mentions],
                rels=[GDocRel(r.subj, r.pred, r.obj, r.modality, (r.subj_span[0] + sh, r.subj_span[1] + sh),
                              (r.obj_span[0] + sh, r.obj_span[1] + sh), (r.sentence[0] + sh, r.sentence[1] + sh), r.valid_from, r.valid_to)
                      for r in story.rels], flags={"syndicated_of": story.name})
    docs.append(copy)
    # stale statement after resignation
    ended = [r for r in dirs if r.valid_to]
    if ended:
        r = ended[0]
        doc = GDoc("news_profile.txt", "news_report", "journalism", _d(date.fromisoformat(r.valid_to) + timedelta(days=120)))
        wr = Writer(doc, w, rng, split)
        wr.t("Business profile\n\n")
        wr.sent("{S} is a director of {O}, sources said.", {"S": r.subj, "O": r.obj}, rel=("S", "DIRECTOR_OF", "O", "reported"))
        doc.flags["stale_of"] = (r.subj, r.obj, r.valid_to)
        docs.append(wr.finish())

    # ---------------------------------------------------------------- FIRs
    for fir_gid in w.planted["firs"]:
        fir = ents[fir_gid]
        num, yr = fir.name[4:].split("/")
        doc = GDoc(f"fir_{fir_gid}.txt", "fir", "official_record", fir.attrs["date"])
        wr = Writer(doc, w, rng, split, noise=0.015 if fir_gid.endswith("1") else 0.0)
        wr.t("FIRST INFORMATION REPORT\n\n")
        fir_tok = f"FIR No. {num}/{yr}"
        _fir_slot(wr, fir_gid, fir_tok)
        wr.m(fir_gid)
        wr.t(f" dated {fmt_date(rng, fir.attrs['date'])} was registered at ")
        wr.m(next(g.gid for g in w.of("GOVT_BODY") if g.name == fir.attrs["station"]))
        wr.t(f" {fir.attrs['sections']}. ")
        wr.para()
        comp = fir.attrs["complainant"]
        wr.sent("The complainant {S} lodged the complaint at the police station.", {"S": comp})
        for a in fir.attrs["accused"]:
            p = ents[a]
            alias = f" @ {p.attrs['alias']}" if p.attrs.get("alias") else ""
            wr.sent(pick(rng, T_ACCUSED, split).replace("{F}", "{F}"), {"S": a, "F": _fir_slot(wr, fir_gid, fir_tok)},
                    rel=("S", "ACCUSED_IN", "F", "asserted"))
            if alias:
                wr.sent("{S_bare}" + alias + ", resident of {L}, is absconding.", {"S_bare": a, "L": f"L-{p.attrs['city']}"},
                        rel=("S_bare", "LOCATED_IN", "L", "asserted"))
            for r in [r for r in w.rels if r.subj == a and r.pred == "USES_PHONE"][:1]:
                ph = ents[r.obj].name[3:]
                wr.sent(pick(rng, T_PHONE, split), {"S": a, "PH": r.obj}, rel=("S", "USES_PHONE", "PH", "asserted"))
            for r in [r for r in w.rels if r.subj == a and r.pred == "OWNS_VEHICLE"][:1]:
                wr.sent(pick(rng, T_VEH, split), {"S": a, "V": r.obj}, rel=("S", "OWNS_VEHICLE", "V", "asserted"))
            for r in [r for r in w.rels if r.pred == "ASSOCIATE_OF" and a in (r.subj, r.obj)][:2]:
                wr.sent(pick(rng, T_ASSOC, split), {"S": r.subj, "S2": r.obj}, rel=("S", "ASSOCIATE_OF", "S2", "asserted"))
        wr.para()
        docs.append(wr.finish())

    # ---------------------------------------------------------------- chargesheet (DOCX) with accounts
    doc = GDoc("chargesheet_gang0.docx", "chargesheet", "official_record", "2024-11-20",
               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    wr = Writer(doc, w, rng, split)
    wr.t("CHARGESHEET — FINAL REPORT\n\n")
    for gid in w.groups["gang0"] + w.groups["gang1"][:2]:
        for r in [r for r in w.rels if r.subj == gid and r.pred == "HOLDS_ACCOUNT"]:
            bank = ents[r.obj].attrs["bank"]
            wr.sent(pick(rng, T_ACCT, split), {"S": gid, "AC": r.obj, "BK": next(b.gid for b in w.of("ORGANISATION") if b.name == bank)},
                    rel=("S", "HOLDS_ACCOUNT", "AC", "asserted"))
        wr.para()
    sp = w.planted["shared_phone"]
    wr.sent("{S} was also found using mobile number {PH}.", {"S": sp["holders"][1], "PH": sp["phone"]}, rel=("S", "USES_PHONE", "PH", "asserted"))
    docs.append(wr.finish())

    # ---------------------------------------------------------------- scanned PDF (OCR path): one award notice
    a1 = awards[1]
    doc = GDoc("scanned_award_notice.pdf", "tender_award", "official_record", a1.valid_from, mime="application/pdf")
    wr = Writer(doc, w, rng, split)
    wr.t("LETTER OF AWARD\n\n")
    wr.sent("{B} awarded the contract to {S} on {D} for {A}.", {"B": a1.subj, "S": a1.obj, "D": fmt_date(rng, a1.valid_from),
                                                              "A": fmt_amt(rng, a1.attrs["amount"])}, rel=("B", "AWARDED_CONTRACT_TO", "S", "asserted"))
    d = wr.finish()
    d.flags["scanned"] = True
    docs.append(d)

    # ---------------------------------------------------------------- structured records
    docs += _structured(w, rng)
    return docs


def _fir_slot(wr: Writer, fir_gid: str, surface: str) -> str:
    # register the FIR surface as the CASE entity's mention text
    wr.w.entities[fir_gid].attrs["surface"] = surface
    return fir_gid


def _structured(w: World, rng: random.Random) -> list[GDoc]:
    ents = w.entities
    out = []
    # CDR
    buf = io.StringIO()
    cw = csv.writer(buf)
    cw.writerow(["Calling Number", "Called Number", "Call Date", "Duration", "IMEI", "Cell ID"])
    for a, b, d, dur in w.planted["calls"]:
        cw.writerow([ents[a].name[3:], ents[b].name[3:], d.isoformat(), dur, f"35{rng.randint(10**12, 10**13 - 1)}", f"404-27-{rng.randint(1000, 9999)}"])
    out.append(GDoc("cdr_export.csv", "cdr", "structured_record", "2024-12-01", text=buf.getvalue(), mime="text/csv", flags={"structured": True}))
    # bank statement
    buf = io.StringIO()
    cw = csv.writer(buf)
    cw.writerow(["Txn Date", "From Account", "To Account", "Amount", "UTR", "Mode"])
    for a, b, d, amt in w.planted["txns"]:
        cw.writerow([d.isoformat(), ents[a].name, ents[b].name, amt, f"UTR{rng.randint(10**9, 10**10 - 1)}", rng.choice(["NEFT", "IMPS", "RTGS", "UPI"])])
    out.append(GDoc("bank_statements.csv", "bank_statement", "structured_record", "2024-12-01", text=buf.getvalue(), mime="text/csv", flags={"structured": True}))
    # subscriber register (XLSX)
    rows = [["Mobile Number", "Subscriber Name", "Address"]]
    for r in w.rels:
        if r.pred == "USES_PHONE":
            p = ents[r.subj]
            rows.append([ents[r.obj].name[3:], p.name, f"{p.attrs['city']}"])
    out.append(GDoc("subscriber_register.xlsx", "subscriber", "structured_record", "2024-12-01", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    flags={"structured": True, "rows": rows}))
    return out


# ============================================================================ materialise to disk
def write_corpus(docs: list[GDoc], out_dir: str | Path, world: World) -> Path:
    out = Path(out_dir)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    gold: dict[str, Any] = {"documents": {}, "entities": {g: {"type": e.type, "name": e.name} for g, e in world.entities.items()},
                            "relations": [r.__dict__ for r in world.rels],
                            "groups": world.groups,
                            "planted": {k: v for k, v in world.planted.items() if k not in ("calls", "txns")}}
    for d in docs:
        p = out / "docs" / d.name
        if d.mime == "application/pdf":
            _write_scanned_pdf(d.text, p)
        elif d.name.endswith(".docx"):
            _write_docx(d.text, p)
        elif d.name.endswith(".xlsx"):
            _write_xlsx(d.flags["rows"], p)
        else:
            p.write_text(d.text, encoding="utf-8")
        meta = {"title": d.name, "document_type": d.doc_type, "reliability": d.reliability, "publication_date": d.publication_date,
                "source_id": "news-wire" if d.doc_type == "news_report" else ("records" if d.flags.get("structured") else "official-portal")}
        (out / "docs" / (d.name + ".meta.json")).write_text(json.dumps(meta))
        gold["documents"][d.name] = {"text": d.text, "mentions": [m.__dict__ for m in d.mentions],
                                     "rels": [{**r.__dict__, "subj_span": list(r.subj_span), "obj_span": list(r.obj_span),
                                               "sentence": list(r.sentence)} for r in d.rels],
                                     "flags": {k: v for k, v in d.flags.items() if k != "rows"}, "doc_type": d.doc_type}
    (out / "gold.json").write_text(json.dumps(gold, indent=1, default=str))
    return out


def _write_docx(text: str, path: Path) -> None:
    import docx  # type: ignore
    d = docx.Document()
    for para in text.split("\n\n"):
        if para.strip():
            d.add_paragraph(para.strip())
    d.save(str(path))


def _write_xlsx(rows: list[list[str]], path: Path) -> None:
    import openpyxl  # type: ignore
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    wb.save(str(path))


def _write_scanned_pdf(text: str, path: Path) -> None:
    """Render text to an image and wrap it in an image-only PDF (forces the OCR path)."""
    import pymupdf  # type: ignore
    src = pymupdf.open()
    page = src.new_page(width=595, height=842)
    page.insert_textbox(pymupdf.Rect(50, 60, 545, 800), text, fontsize=12, fontname="helv")
    pix = page.get_pixmap(dpi=200)
    out = pymupdf.open()
    p2 = out.new_page(width=595, height=842)
    p2.insert_image(p2.rect, stream=pix.tobytes("png"))
    out.save(str(path))


def generate(out_dir: str | Path, seed: int = 42, split: str = "test", noise: float = 0.0) -> Path:
    """noise > 0 applies length-preserving OCR-style confusions to every text document,
    *including inside entity names* (robustness experiment; gold offsets stay valid)."""
    w = build_world(seed)
    docs = render_corpus(w, seed, split)
    if noise:
        rng = random.Random(seed + 99)
        for d in docs:
            if not d.flags.get("structured") and d.mime == "text/plain":
                d.text = ocr_noise(d.text, rng, noise, protect=[])
    return write_corpus(docs, out_dir, w)
