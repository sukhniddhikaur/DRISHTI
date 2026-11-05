"""Hybrid named-entity recognition.

Layers (each emits spans with an extractor tag and a raw score):
  1. validated identifiers        (identifiers.py)        score ~0.99
  2. dates                         (temporal.py)
  3. grammar rules: legal-form suffix heads for organisations, govt-body
     heads, honorific / role-cue / kinship-cue persons, gazetteers
  4. case-level propagation: names found with high confidence in one
     document are searched (as exact surface forms and suffix-less heads)
     across the case — recovers "ABC Infra" after "ABC Infra Pvt Ltd" was seen
  5. optional statistical backend (spaCy) merged by span voting

Overlaps resolve by (priority, length, score). Identifier mentions are attached
to the nearest owner mention in the same sentence: they become ER signals.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from drishti.extract.identifiers import PAN_HOLDER, find_identifiers
from drishti.extract.temporal import find_dates
from drishti.resolve.normalize import normalize
from drishti.understand.segment import sentences

CAP = r"(?:[A-Z][a-zA-Z0-9'’-]*\.?|[A-Z]\.|\d+[A-Za-z]+)"
ORG_SUFFIX = re.compile(
    r"\b(Private Limited|Pvt\.?\s*Ltd\.?|Pvt\.?\s+Limited|Private Ltd\.?|Limited|Ltd\.?|LLP|L\.L\.P\.|Inc\.|Corporation|Corp\.|"
    r"& Co\.?|and Company|Industries|Enterprises|Traders|Associates|Holdings|Ventures|Group|Co-operative Society|"
    r"Trust|Foundation|Infratech|Constructions|Builders|Exports|Logistics|Solutions|Technologies|Developers)(?![A-Za-z])", re.I)
GOV_HEAD = re.compile(
    r"\b(Department|Ministry|Authority|Commission|Municipal Corporation|Nagar Nigam|Police Station|Directorate|"
    r"Development Authority|Registrar of Companies|High Court|District Court|Sessions Court|Supreme Court|Tribunal|"
    r"Board|Council|Commissionerate|Zila Parishad|Panchayat)(?![A-Za-z])")
GOV_PREFIX = re.compile(r"\b(?:Ministry|Department|Directorate|Office|Commissionerate|Registrar) of (?:the )?(?:" + CAP + r"(?:\s+(?:and|&|of)?\s*" + CAP + r"){0,5})")
BANK = re.compile(r"\b(?:State Bank of India|(?:" + CAP + r"\s+){0,2}Bank(?: of " + CAP + r")?)")
HONORIFIC = re.compile(r"\b(Mr|Mrs|Ms|Smt|Shri|Sri|Dr|Kum|Km|Sh)\b\.?\s+((?:" + CAP + r"\s*){1,4})")
ROLE_CUE = re.compile(r"\b(?:accused(?: no\.? ?\d+)?|complainant|petitioner|respondent|deponent|informant|witness|"
                      r"director|promoter|proprietor|partner|suspect|subscriber|account holder)\s*[:,-]?\s+((?:" + CAP + r"\s*){2,4})", re.I)
KIN_CUE = re.compile(r"((?:" + CAP + r"\s+){1,3}" + CAP + r")\s*(?:,\s*)?(?:\(?\s*(?:s/o|d/o|w/o|son of|daughter of|wife of|aged|r/o)\b|\(DIN|\(PAN|@\s*[A-Z]|alias\b)", re.I)
ALIAS = re.compile(r"\s*(?:@|alias|a\.k\.a\.?|urf)\s+(" + CAP + r"(?:\s+" + CAP + r")?)")
DEFINED_TERM = re.compile(r"\s*\((?:hereinafter(?: referred to as)?\s*)?[\"“']([A-Z][A-Za-z0-9 &.]{1,30})[\"”']\)")

STOP = {"The", "This", "That", "These", "Those", "Mr", "Mrs", "Ms", "Smt", "Shri", "Director", "Directors", "Managing",
        "Chairman", "Accused", "Complainant", "Petitioner", "Respondent", "Hon'ble", "Honble", "Court", "Board", "Said",
        "Whole-time", "Independent", "Additional", "Company", "Contract", "Tender", "Order", "Notice", "Annexure", "Section",
        "Under", "As", "In", "On", "By", "For", "With", "From", "To", "And", "Of", "At", "According", "Sources", "It",
        "He", "She", "They", "Mob", "Mobile", "Phone", "Account", "Vehicle", "Police", "FIR", "Station", "Its", "Their",
        "His", "Her", "Messrs", "M/s", "M/s.", "We", "Our", "An", "A", "Report", "Annual", "Filing", "Form", "Subject",
        "Re", "Dated", "Date", "Present", "Representatives", "Bids", "Representative", "Police", "Smt", "Ms", "Appointment", "Resignation", "Award", "Letter", "Work", "Following", "After",
        "Before", "During", "Further", "Also", "However", "Thereafter", "Meanwhile", "Note", "Shareholding", "Details",
        "Name", "Address", "Registered", "Office", "Subsidiaries", "Subsidiary", "Holding", "Parent", "Promoter", "Every",
        "Whereas", "Upon", "Pursuant", "Having", "Being", "While", "Ld", "Learned", "Counsel", "Advocate", "Bench",
        "Justice", "Judgment", "Petition", "Suit", "Writ", "Case", "No", "Inspector", "Constable", "Officer", "SHO",
        "IO", "Sub-Inspector", "Investigating", "Statement", "Witnesses", "Cell", "Bank", "Branch", "Rs", "INR", "Total"}
CONNECT = {"of", "and", "&", "for"}

GIVEN = set("""Aarav Abhishek Aditya Ajay Akash Alok Amit Anand Anil Anjali Ankit Anita Arjun Arun Arvind Ashok Ayesha
Deepak Deepika Dinesh Divya Farhan Gaurav Geeta Harish Imran Irfan Jaya Kavita Karan Kiran Krishna Kunal Lakshmi Mahesh
Manish Manoj Meena Mohan Mohammad Mohammed Muhammad Mukesh Naveen Neha Nikhil Nisha Pankaj Pooja Prakash Pradeep Priya
Rahul Raj Rajesh Rajiv Rakesh Ramesh Ravi Rekha Ritu Rohit Sachin Sameer Sandeep Sanjay Sanjeev Santosh Sarita Shalini
Shankar Sharad Shweta Simran Sita Sneha Sonia Sunil Sunita Suresh Swati Tarun Uma Usha Varun Vijay Vikas Vikram Vinod
Vishal Yash Yusuf Zoya Faisal Salim Javed Gurpreet Harpreet Manpreet Baljeet Rupinder Venkatesh Srinivas Lakshman
Murali Kannan Senthil Lalit Hemant Naresh Girish Jitendra Dharmendra Bhavesh Hitesh Paresh Ketan Nilesh Jignesh""".split())

LOCATIONS = set("""Delhi Mumbai Kolkata Chennai Bengaluru Bangalore Hyderabad Pune Ahmedabad Surat Jaipur Lucknow Kanpur Nagpur
Indore Bhopal Patna Vadodara Ludhiana Agra Nashik Faridabad Meerut Rajkot Varanasi Srinagar Aurangabad Dhanbad Amritsar
Allahabad Prayagraj Ranchi Howrah Coimbatore Jabalpur Gwalior Vijayawada Jodhpur Madurai Raipur Kota Guwahati Chandigarh
Thiruvananthapuram Kochi Noida Gurugram Gurgaon Ghaziabad Thane Navi Visakhapatnam Bhubaneswar Dehradun Mysuru Mangaluru
Akola Rourkela Cuttack Udaipur Ajmer Bikaner Shimla Panaji Goa Maharashtra Karnataka Telangana Gujarat Rajasthan Kerala
Punjab Haryana Bihar Odisha Assam Jharkhand Uttarakhand Chhattisgarh Tamil Nadu Uttar Pradesh Madhya Pradesh Andhra
West Bengal Himachal Jammu Kashmir Ladakh Sikkim Tripura Manipur Meghalaya Mizoram Nagaland Arunachal India""".split())
MULTI_LOC = re.compile(r"\b(Tamil Nadu|Uttar Pradesh|Madhya Pradesh|Andhra Pradesh|West Bengal|Himachal Pradesh|Navi Mumbai|"
                       r"Jammu and Kashmir|New Delhi|Arunachal Pradesh)\b")

PRIO = {"IDENT": 9, "CASE": 8, "CONTRACT": 8, "GOVT_BODY": 7, "ORGANISATION": 6, "PERSON": 5, "LOCATION": 4, "DATE": 3}
ID_AS_ENTITY = {"PHONE": "PHONE", "ACCOUNT": "ACCOUNT", "VEHICLE": "VEHICLE", "CASE": "CASE", "CONTRACT": "CONTRACT",
                "MONEY": "MONEY"}
ID_AS_ATTR = {"PAN", "CIN", "DIN", "GSTIN", "EMAIL", "PHONE", "ACCOUNT", "AADHAAR", "IFSC"}


@dataclass
class Span:
    label: str
    start: int
    end: int
    text: str
    score: float
    extractor: str
    normalized: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)


LEGAL_WORDS = {"Ltd", "Limited", "LLP", "Pvt", "Private", "Inc", "Corp", "Co", "L.L.P", "LTD", "LIMITED", "PVT", "PRIVATE"}


def _left_scan(text: str, head_start: int, sent_start: int, max_tokens: int = 7, connect: frozenset = frozenset({"&"})) -> int:
    """Walk left from a head word over capitalised name tokens; returns name start."""
    toks = [(m.group(0), m.start()) for m in re.finditer(r"[A-Za-z0-9&'’./-]+", text[sent_start:head_start])]
    toks = [(t, s + sent_start) for t, s in toks]
    start, taken, i = head_start, 0, len(toks) - 1
    while i >= 0 and taken < max_tokens:
        t, s = toks[i]
        bare = t.rstrip(".,")
        if bare in LEGAL_WORDS:
            break  # an organisation name never contains another legal-form word to its left
        if bare in connect and i > 0 and toks[i - 1][0][:1].isupper() and toks[i - 1][0].rstrip(".,") not in STOP | LEGAL_WORDS:
            i -= 1
            continue
        if not (bare[:1].isupper() or bare[:1].isdigit()) or bare in STOP or t.endswith(","):
            break
        start, taken = s, taken + 1
        i -= 1
    # never start on a connector
    while True:
        m = re.match(r"(of|and|&|for)\s+", text[start:head_start])
        if not m:
            break
        start += m.end()
    return start


class NER:
    VERSION = "ner-2.1"

    def __init__(self, gazetteer: dict[str, str] | None = None, spacy_model: str | None = None):
        self.gazetteer = dict(gazetteer or {})  # surface form -> label, built at case level
        self._nlp = None
        if spacy_model:
            try:
                import spacy  # type: ignore
                self._nlp = spacy.load(spacy_model)
            except Exception:
                self._nlp = None

    # ------------------------------------------------------------------ public
    def extract(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for s0, s1 in sentences(text):
            spans += self._sentence(text, s0, s1)
        spans += self._gazetteer_pass(text, spans)
        if self._nlp is not None:
            spans += self._spacy(text)
        out = self._resolve_overlaps(spans)
        self._attach_identifiers(text, out)
        for sp in out:
            if not sp.normalized:
                sp.normalized = normalize(sp.label, sp.text)
        return out

    # ------------------------------------------------------------------ rules
    def _sentence(self, text: str, s0: int, s1: int) -> list[Span]:
        out: list[Span] = []
        seg = text[s0:s1]
        for im in find_identifiers(text, s0, s1):
            lab = ID_AS_ENTITY.get(im.label, "IDENT")
            out.append(Span(lab, im.start, im.end, im.text, 0.99 if im.label not in ("MONEY",) else 0.97, f"id:{im.label}",
                            im.normalized, {"id_type": im.label, **im.attrs}))
        for d in find_dates(text, s0, s1):
            out.append(Span("DATE", d.start, d.end, d.text, 0.97, "date", d.iso, {"end_iso": d.end_iso, "precision": d.precision}))
        # organisations: legal-form suffix head
        for m in ORG_SUFFIX.finditer(text, s0, s1):
            if not m.group(0)[0].isupper():
                continue  # "a limited number" is not a legal form
            st = _left_scan(text, m.start(), s0)
            if st == m.start():
                continue
            out.append(self._mk("ORGANISATION", text, st, m.end(), 0.9, "org:suffix"))
        for m in BANK.finditer(text, s0, s1):
            st = _left_scan(text, m.start(), s0, 2) if not m.group(0).startswith("State Bank") else m.start()
            out.append(self._mk("ORGANISATION", text, min(st, m.start()), m.end(), 0.85, "org:bank"))
        # government bodies
        for m in GOV_PREFIX.finditer(text, s0, s1):
            out.append(self._mk("GOVT_BODY", text, m.start(), m.end(), 0.9, "gov:prefix"))
        for m in GOV_HEAD.finditer(text, s0, s1):
            st = _left_scan(text, m.start(), s0, 5, frozenset(CONNECT))
            en = m.end()
            tail = re.match(r"\s+(?:of|at|,)\s+(" + CAP + r"(?:\s+" + CAP + r")?)", text[en:s1])
            if tail and tail.group(1).split()[0] not in STOP:
                en += tail.end()
            if st == m.start() and en == m.end():
                continue
            out.append(self._mk("GOVT_BODY", text, st, en, 0.88, "gov:head"))
        # persons
        for m in HONORIFIC.finditer(text, s0, s1):
            name = self._trim_name(m.group(2))
            if name:
                st = m.start(2)
                sp = self._mk("PERSON", text, st, st + len(name), 0.93, "per:honorific")
                sp.attrs["honorific"] = m.group(1)
                out.append(sp)
        for m in ROLE_CUE.finditer(text, s0, s1):
            name = self._trim_name(m.group(1))
            if name and len(name.split()) >= 2:
                out.append(self._mk("PERSON", text, m.start(1), m.start(1) + len(name), 0.8, "per:role"))
        for m in KIN_CUE.finditer(text, s0, s1):
            name = self._trim_name(m.group(1))
            if name and len(name.split()) >= 2:
                st = m.start(1) + (len(m.group(1)) - len(m.group(1).lstrip()))
                out.append(self._mk("PERSON", text, st, st + len(name), 0.85, "per:kin"))
        for m in re.finditer(r"\b([A-Z][a-z]+)\s+((?:[A-Z][a-z]+|[A-Z]\.)(?:\s+[A-Z][a-z]+)?)\b", seg):
            if m.group(1) in GIVEN:
                name = self._trim_name(m.group(0))
                if name and len(name.split()) >= 2:
                    out.append(self._mk("PERSON", text, s0 + m.start(), s0 + m.start() + len(name), 0.75, "per:given"))
        # locations
        for m in MULTI_LOC.finditer(text, s0, s1):
            out.append(self._mk("LOCATION", text, m.start(), m.end(), 0.85, "loc:gaz"))
        for m in re.finditer(r"\b([A-Z][a-z]+)(?:\s+District)?\b", seg):
            if m.group(1) in LOCATIONS:
                out.append(self._mk("LOCATION", text, s0 + m.start(), s0 + m.end(), 0.8, "loc:gaz"))
        return out

    @staticmethod
    def _trim_name(raw: str) -> str:
        toks = raw.strip().split()
        keep = []
        for t in toks:
            b = t.rstrip(".,;:")
            if re.fullmatch(r"[A-Z]\.", t):
                keep.append(t)
                continue
            if b in STOP or b.lower() in {"s/o", "d/o", "w/o", "r/o", "aged", "alias"} or not b[:1].isupper():
                break
            if (ORG_SUFFIX.fullmatch(b) and b[0].isupper()) or GOV_HEAD.fullmatch(b):
                return ""
            keep.append(t.rstrip(",;:") if not re.fullmatch(r"[A-Z]\.", t) else t)
        name = " ".join(keep).rstrip(".") if keep and not re.fullmatch(r"[A-Z]\.", keep[-1]) else " ".join(keep)
        return name

    def _mk(self, label: str, text: str, a: int, b: int, score: float, ex: str) -> Span:
        while b > a and text[b - 1] in " ,;:":
            b -= 1
        while a < b and text[a] in " ,;:(":
            a += 1
        if b - a > 2 and text[b - 1] == "." and text[b - 2].islower() and not text[a:b].endswith(("Ltd.", "Inc.", "Corp.", "Co.", "Pvt.")):
            b -= 1
        if label in ("ORGANISATION", "GOVT_BODY"):
            while True:
                m = re.match(r"(?:The|M/s\.?|Messrs\.?|the)\s+", text[a:b])
                if not m or a + m.end() >= b:
                    break
                a += m.end()
        sp = Span(label, a, b, text[a:b], score, ex)
        if label == "PERSON" and re.search(r"(?:s/o|d/o|w/o|son of|daughter of|wife of)\s*$", text[max(0, a - 16):a], re.I):
            sp.attrs["kin_reference"] = True
        # alias / defined-term capture right after the name
        tail = text[b:b + 60]
        am = ALIAS.match(tail)
        if am and label == "PERSON":
            sp.attrs["aliases"] = [am.group(1).strip()]
        dm = DEFINED_TERM.match(tail)
        if dm:
            sp.attrs.setdefault("aliases", []).append(dm.group(1).strip())
        return sp

    # ------------------------------------------------------------------ propagation
    def _gazetteer_pass(self, text: str, spans: list[Span]) -> list[Span]:
        forms = dict(self.gazetteer)
        for sp in spans:
            if sp.label in ("ORGANISATION", "PERSON", "GOVT_BODY") and sp.score >= 0.85:
                forms.setdefault(sp.text, sp.label)
        heads: dict[str, str] = {}
        for f, lab in forms.items():
            if lab == "ORGANISATION":
                m = ORG_SUFFIX.search(f)
                if m and m.start() > 0:
                    head = f[:m.start()].strip(" ,")
                    toks = head.split()
                    if len(toks) >= 2:
                        # head with last token prefix-truncated (Infrastructure ~ Infra)
                        rx = r"\b" + r"\s+".join(map(re.escape, toks[:-1])) + r"\s+" + re.escape(toks[-1][:4]) + r"[A-Za-z]*\b"
                        heads[rx] = lab
            elif lab == "PERSON":
                core = re.sub(r"^(?:Mr|Mrs|Ms|Smt|Shri|Sri|Dr|Kum|Km|Sh)\.?\s+", "", f)
                if len(core.split()) >= 2:
                    heads[r"\b" + re.escape(core) + r"\b"] = lab
        for f, lab in forms.items():
            if len(f) >= 4:
                heads.setdefault(r"(?<![A-Za-z])" + re.escape(f) + r"(?![A-Za-z])", lab)
        out = []
        for rx, lab in heads.items():
            for m in re.finditer(rx, text):
                out.append(Span(lab, m.start(), m.end(), m.group(0), 0.7, "gazetteer"))
        return out

    def _spacy(self, text: str) -> list[Span]:
        mp = {"PERSON": "PERSON", "ORG": "ORGANISATION", "GPE": "LOCATION", "LOC": "LOCATION"}
        return [Span(mp[e.label_], e.start_char, e.end_char, e.text, 0.72, "spacy")
                for e in self._nlp(text).ents if e.label_ in mp]

    # ------------------------------------------------------------------ overlaps
    @staticmethod
    def _resolve_overlaps(spans: list[Span]) -> list[Span]:
        def prio(s: Span) -> int:
            return PRIO["IDENT"] if s.extractor.startswith("id:") else PRIO.get(s.label, 1)

        # agreement bonus: same span proposed by several layers
        votes: dict[tuple[int, int, str], int] = {}
        for s in spans:
            votes[(s.start, s.end, s.label)] = votes.get((s.start, s.end, s.label), 0) + 1
        spans = sorted(spans, key=lambda s: (-prio(s), -(s.end - s.start), -s.score))
        taken: list[Span] = []
        for s in spans:
            if s.end - s.start < 2:
                continue
            if all(s.end <= t.start or s.start >= t.end for t in taken):
                v = votes[(s.start, s.end, s.label)]
                if v > 1:
                    s.score = min(0.99, s.score + 0.05 * (v - 1))
                taken.append(s)
        return sorted(taken, key=lambda s: s.start)

    # ------------------------------------------------------------------ id attachment
    @staticmethod
    def _attach_identifiers(text: str, spans: list[Span]) -> None:
        owners = [s for s in spans if s.label in ("PERSON", "ORGANISATION", "GOVT_BODY") and not s.attrs.get("kin_reference")]
        for s in spans:
            idt = s.attrs.get("id_type")
            if idt not in ID_AS_ATTR:
                continue
            prev = [o for o in owners if o.end <= s.start and s.start - o.end <= 70 and "\n" not in text[o.end:s.start]
                    and ";" not in text[o.end:s.start] and "." not in text[o.end:s.start].replace("No.", "").replace("Mob.", "")]
            if not prev:
                continue
            o = prev[-1]
            between = text[o.end:s.start]
            if re.search(r"\b(?:and|while|whereas)\b", between) and len(between) > 25:
                continue
            o.attrs.setdefault("identifiers", {}).setdefault(idt, [])
            if s.normalized not in o.attrs["identifiers"][idt]:
                o.attrs["identifiers"][idt].append(s.normalized)
            s.attrs["owner_span"] = (o.start, o.end)
            if idt in ("PAN", "GSTIN") and s.attrs.get("holder_type"):
                o.attrs["pan_holder_type"] = s.attrs["holder_type"]


def build_case_gazetteer(span_lists: Iterable[list[Span]], min_score: float = 0.85) -> dict[str, str]:
    g: dict[str, str] = {}
    for spans in span_lists:
        for s in spans:
            if s.label in ("ORGANISATION", "PERSON", "GOVT_BODY") and s.score >= min_score and s.extractor != "gazetteer":
                g.setdefault(s.text, s.label)
    return g


PAN_TYPES = PAN_HOLDER
