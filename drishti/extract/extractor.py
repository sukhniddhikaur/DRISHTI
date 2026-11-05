"""Per-document extraction: spans -> Mentions, relation hits -> Claims, Events.

Hard guarantee: a Claim is only emitted if `text[char_start:char_end] ==
evidence_text` for the document's canonical text. Anything else is rejected
and counted, so "unsupported claim" cannot enter the store.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from drishti.core.domain import DomainPack
from drishti.core.ids import stable_id
from drishti.core.models import Claim, Epistemic, Event, Mention, Modality, Passage
from drishti.extract.calibrate import Calibrator
from drishti.extract.ner import NER, Span
from drishti.extract.relations import RelationExtractor
from drishti.extract.structured import parse_rows
from drishti.extract.temporal import find_dates, interval_for
from drishti.resolve.normalize import normalize
from drishti.understand.parser import ParsedDocument
from drishti.understand.segment import sentences

EVENT_PREDICATES = {"AWARDED_CONTRACT_TO": "CONTRACT_AWARD", "ACQUIRED": "ACQUISITION", "FILED_CASE_AGAINST": "LITIGATION",
                    "ACCUSED_IN": "CASE_ACCUSATION", "TRANSFERRED_TO": "TRANSACTION", "CALLED": "CALL"}
ATTACH_PRED = {"PHONE": "USES_PHONE", "ACCOUNT": "HOLDS_ACCOUNT", "VEHICLE": "OWNS_VEHICLE"}


@dataclass
class ExtractionOutput:
    mentions: list[Mention]
    claims: list[Claim]
    events: list[Event]
    rejected: int = 0
    stats: dict[str, Any] = field(default_factory=dict)


class DocumentExtractor:
    def __init__(self, pack: DomainPack, ner: NER | None = None, rel: RelationExtractor | None = None,
                 calibrator: Calibrator | None = None):
        self.pack = pack
        self.ner = ner or NER()
        self.rel = rel or RelationExtractor(pack)
        self.cal = calibrator or Calibrator()

    @property
    def version(self) -> str:
        return f"{self.ner.VERSION}|{self.rel.VERSION}:{self.rel.mode}"

    def _pid(self, passages: list[Passage], pos: int) -> tuple[str | None, int]:
        for p in passages:
            if p.char_start <= pos < p.char_end:
                return p.passage_id, p.page
        return None, 1

    def run(self, doc: dict[str, Any], parsed: ParsedDocument, spans: list[Span] | None = None) -> ExtractionOutput:
        text, did, case = parsed.text, doc["document_id"], doc["case_id"]
        reported = doc.get("publication_date")
        spans = spans if spans is not None else self.ner.extract(text)
        sents = sentences(text)
        mentions: dict[tuple[int, int, str], Mention] = {}

        def mention(sp: Span) -> Mention:
            k = (sp.start, sp.end, sp.label)
            if k not in mentions:
                pid, _ = self._pid(parsed.passages, sp.start)
                mentions[k] = Mention(
                    mention_id=stable_id("MEN", did, sp.start, sp.end, sp.label), document_id=did, passage_id=pid,
                    label=sp.label, text=sp.text, char_start=sp.start, char_end=sp.end, normalized=sp.normalized,
                    attributes={k2: v for k2, v in sp.attrs.items() if k2 != "owner_span"}, extractor=sp.extractor,
                    extractor_version=self.ner.VERSION, confidence=round(min(1.0, sp.score), 4))
            return mentions[k]

        for sp in spans:
            if sp.label not in ("DATE", "IDENT", "MONEY"):
                mention(sp)

        claims: list[Claim] = []
        rejected = 0

        def add_claim(pred: str, subj: Span | Mention, obj: Span | Mention, a: int, b: int, raw: float, ex: str,
                      modality: Modality = Modality.ASSERTED, epistemic: Epistemic = Epistemic.EXTRACTED,
                      vf: str | None = None, vt: str | None = None, attrs: dict | None = None) -> Claim | None:
            nonlocal rejected
            sm = subj if isinstance(subj, Mention) else mention(subj)
            om = obj if isinstance(obj, Mention) else mention(obj)
            ev = text[a:b]
            pid, page = self._pid(parsed.passages, a)
            inside = all(a <= m.char_start and m.char_end <= b for m in (sm, om))
            if not ev or text[a:b] != ev or not (inside or epistemic == Epistemic.STATED):
                rejected += 1  # evidence span must contain both arguments
                return None
            c = Claim(claim_id=stable_id("CLM", did, pred, sm.mention_id, om.mention_id, a), case_id=case, predicate=pred,
                      subject_mention_id=sm.mention_id, object_mention_id=om.mention_id, subject_text=sm.text,
                      object_text=om.text, document_id=did, passage_id=pid, evidence_text=ev, char_start=a, char_end=b,
                      page=page, modality=modality, epistemic=epistemic, valid_from=vf, valid_to=vt, reported_at=reported,
                      extractor=ex, extractor_version=self.rel.VERSION, raw_score=round(raw, 4),
                      confidence=round(self.cal(ex, raw), 4) if epistemic != Epistemic.STATED else 0.99,
                      attributes=attrs or {})
            claims.append(c)
            return c

        # ---------------------------------------------------------------- text relations
        by_start = sorted(spans, key=lambda s: s.start)
        for h in self.rel.extract(text, spans, sents):
            sent_dates = find_dates(text, h.s0, h.s1)
            vf, vt, point = interval_for(text[h.s0:h.s1], sent_dates, h.s0)
            if h.ended and not vt:
                vt = point or reported
            attrs: dict[str, Any] = {"trigger": h.trigger, "cues": h.cues}
            if point:
                attrs["event_date"] = point
            money = [s for s in by_start if s.label == "MONEY" and h.s0 <= s.start < h.s1]
            if money and h.predicate in ("AWARDED_CONTRACT_TO", "ACQUIRED", "TRANSFERRED_TO"):
                attrs["amount_inr"] = money[0].attrs.get("amount_inr")
            contracts = [s for s in by_start if s.label == "CONTRACT" and h.s0 <= s.start < h.s1]
            if contracts:
                attrs["contract_ref"] = contracts[0].normalized
            if h.predicate == "AWARDED_CONTRACT_TO" and not vf:
                vf = point
            add_claim(h.predicate, h.subj, h.obj, h.s0, h.s1, h.raw_score, h.extractor, h.modality, vf=vf, vt=vt, attrs=attrs)

        # ---------------------------------------------------------------- identifier attachments
        owners = {(s.start, s.end): s for s in spans}
        for sp in spans:
            own = sp.attrs.get("owner_span")
            if not own or sp.label not in ATTACH_PRED:
                continue
            o = owners.get(tuple(own))
            if o is None:
                continue
            spec = self.pack.relations.get(ATTACH_PRED[sp.label])
            if not spec or o.label not in spec.subject_types:
                continue
            s0, s1 = next(((a, b) for a, b in sents if a <= o.start < b), (o.start, sp.end))
            if not any(c.predicate == spec.name and c.subject_mention_id == mention(o).mention_id
                       and c.object_mention_id == mention(sp).mention_id for c in claims):
                add_claim(spec.name, o, sp, s0, s1, 0.9, "attachment", attrs={"via": "adjacent identifier"})

        # ---------------------------------------------------------------- structured rows
        pmap = {p.passage_id: p for p in parsed.passages}
        for row in parse_rows(parsed.tables):
            p = pmap[row.passage_id]
            f = row.fields

            def node(label: str, value: str, display: str | None = None) -> Mention:
                k = (p.char_start, p.char_end, label + ":" + value)
                if k not in mentions:
                    rel = p.text.find(display or value)
                    a = p.char_start + (rel if rel >= 0 else 0)
                    b = a + (len(display or value) if rel >= 0 else len(p.text))
                    mentions[k] = Mention(mention_id=stable_id("MEN", did, a, b, label, value), document_id=did,
                                          passage_id=p.passage_id, label=label, text=text[a:b], char_start=a, char_end=b,
                                          normalized=value, extractor="structured", extractor_version="1", confidence=0.99)
                return mentions[k]

            if row.kind == "cdr":
                a = node("PHONE", f["a"], f["raw"].get("a_party"))
                b = node("PHONE", f["b"], f["raw"].get("b_party"))
                add_claim("CALLED", a, b, p.char_start, p.char_end, 1.0, "structured", epistemic=Epistemic.STATED,
                          vf=f["date"], vt=f["date"], attrs={"duration_s": f["duration"], "time": f["time"], "imei": f["imei"], "cell": f["cell"]})
            elif row.kind == "bank":
                a = node("ACCOUNT", f["from"], f["raw"].get("from_acct"))
                b = node("ACCOUNT", f["to"], f["raw"].get("to_acct"))
                add_claim("TRANSFERRED_TO", a, b, p.char_start, p.char_end, 1.0, "structured", epistemic=Epistemic.STATED,
                          vf=f["date"], vt=f["date"], attrs={"amount_inr": f["amount"], "utr": f["utr"], "mode": f["mode"]})
                for nm, acct in ((f.get("from_name"), a), (f.get("to_name"), b)):
                    if nm:
                        lab = "ORGANISATION" if any(x in nm for x in ("Ltd", "Limited", "Traders", "Enterprises", "LLP", "Pvt")) else "PERSON"
                        holder = node(lab, nm.strip(), nm.strip())
                        holder.normalized = normalize(lab, nm)
                        add_claim("HOLDS_ACCOUNT", holder, acct, p.char_start, p.char_end, 1.0, "structured",
                                  epistemic=Epistemic.STATED)
            elif row.kind == "subscriber" and f.get("name"):
                ph = node("PHONE", f["phone"], f["raw"].get("phone"))
                per = node("PERSON", f["name"].strip(), f["name"].strip())
                per.normalized = normalize("PERSON", f["name"])
                add_claim("USES_PHONE", per, ph, p.char_start, p.char_end, 1.0, "structured", epistemic=Epistemic.STATED,
                          attrs={"address": f.get("address")})

        # ---------------------------------------------------------------- events
        events: list[Event] = []
        for c in claims:
            et = EVENT_PREDICATES.get(c.predicate)
            if not et or c.modality == Modality.NEGATED:
                continue
            events.append(Event(event_id=stable_id("EVT", c.claim_id), case_id=case, event_type=et,
                                date=c.attributes.get("event_date") or c.valid_from, document_id=did, passage_id=c.passage_id,
                                evidence_text=c.evidence_text, char_start=c.char_start, char_end=c.char_end,
                                roles={"subject": c.subject_mention_id, "object": c.object_mention_id},
                                attributes={k: v for k, v in c.attributes.items() if k in ("amount_inr", "contract_ref", "duration_s", "utr")}
                                | {"claim_id": c.claim_id}))
        for c in claims:
            if c.predicate == "DIRECTOR_OF" and c.modality != Modality.NEGATED:
                for when, et in ((c.valid_from, "APPOINTMENT"), (c.valid_to, "RESIGNATION")):
                    if when:
                        events.append(Event(event_id=stable_id("EVT", c.claim_id, et), case_id=case, event_type=et, date=when,
                                            document_id=did, passage_id=c.passage_id, evidence_text=c.evidence_text,
                                            char_start=c.char_start, char_end=c.char_end,
                                            roles={"person": c.subject_mention_id, "organisation": c.object_mention_id},
                                            attributes={"claim_id": c.claim_id}))
        # case registration events: a CASE mention with a date in the same sentence ("FIR No. 12/2024 dated ...")
        for sp in spans:
            if sp.label != "CASE":
                continue
            s0, s1 = next(((a, b) for a, b in sents if a <= sp.start < b), (sp.start, sp.end))
            ds = [d for d in spans if d.label == "DATE" and s0 <= d.start < s1]
            if ds and re.search(r"\b(?:dated|registered|lodged|filed|instituted)\b", text[s0:s1], re.I):
                m = mention(sp)
                events.append(Event(event_id=stable_id("EVT", did, m.mention_id, "REG"), case_id=case, event_type="CASE_REGISTERED",
                                    date=ds[0].normalized, document_id=did, passage_id=m.passage_id, evidence_text=text[s0:s1],
                                    char_start=s0, char_end=s1, roles={"case": m.mention_id}, attributes={"case_ref": sp.normalized}))
        # dedupe claims by id (same span proposed twice)
        uniq = {c.claim_id: c for c in claims}
        return ExtractionOutput(list(mentions.values()), list(uniq.values()), events, rejected,
                                {"spans": len(spans), "claims": len(uniq), "rejected": rejected})
