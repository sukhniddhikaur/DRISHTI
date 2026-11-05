"""Domain packs: the only thing that changes between application domains.

One engine; configurable schemas. A pack declares entity types, the closed
relation schema (argument types, functional/symmetric flags, trigger lexicon)
and document classes. The extractor may only emit predicates in the schema —
models cannot invent labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RelationSpec:
    name: str
    subject_types: tuple[str, ...]
    object_types: tuple[str, ...]
    triggers: tuple[str, ...]              # lexical cues (lemmatised-ish, lowercase)
    passive_triggers: tuple[str, ...] = ()  # cues where surface order is object..subject
    symmetric: bool = False
    functional: bool = False               # subject can hold it with one object at a time
    description: str = ""


@dataclass(frozen=True)
class DomainPack:
    name: str
    entity_types: tuple[str, ...]
    relations: dict[str, RelationSpec]
    document_classes: dict[str, tuple[str, ...]]  # class -> cue words
    structured_relations: tuple[str, ...] = ()
    reporting_thresholds_inr: tuple[int, ...] = ()
    notes: str = ""

    def spec(self, predicate: str) -> RelationSpec:
        return self.relations[predicate]


ORG = ("ORGANISATION", "GOVT_BODY")
PER = ("PERSON",)
ANY_ACTOR = PER + ORG

_COMMON = {
    "LOCATED_IN": RelationSpec(
        "LOCATED_IN", ANY_ACTOR, ("LOCATION",),
        ("located in", "registered office at", "registered office in", "based in", "resident of", "residing at", "headquartered in"),
        description="Entity is associated with a location"),
    "MENTIONED_WITH": RelationSpec(
        "MENTIONED_WITH", ANY_ACTOR, ANY_ACTOR, (), symmetric=True,
        description="Weak co-occurrence in a single passage; never shown as a relationship"),
}

CORPORATE = DomainPack(
    name="corporate",
    entity_types=("PERSON", "ORGANISATION", "GOVT_BODY", "LOCATION", "CONTRACT", "CASE", "IDENTIFIER", "MONEY", "DATE"),
    relations={
        "DIRECTOR_OF": RelationSpec(
            "DIRECTOR_OF", PER, ("ORGANISATION",),
            ("director of", "directors of", "appointed as director", "appointed as a director", "on the board of", "board of directors of",
             "whole-time director of", "managing director of", "independent director of", "directorship in", "director in"),
            passive_triggers=("appointed to its board", "inducted", "board comprises"),
            description="Person holds a disclosed directorship"),
        "SUBSIDIARY_OF": RelationSpec(
            "SUBSIDIARY_OF", ("ORGANISATION",), ("ORGANISATION",),
            ("subsidiary of", "wholly owned subsidiary of", "wholly-owned subsidiary of", "step-down subsidiary of"),
            passive_triggers=("holding company of", "parent company of", "parent of"),
            functional=True, description="Organisation is a subsidiary"),
        "AWARDED_CONTRACT_TO": RelationSpec(
            "AWARDED_CONTRACT_TO", ORG, ("ORGANISATION",),
            ("awarded the contract to", "awarded a contract to", "awarded contract to", "awarded to", "issued the work order to",
             "issued a work order to", "letter of award to", "selected as the l1 bidder", "contract was awarded to"),
            passive_triggers=("received a contract from", "was awarded a contract by", "won the tender floated by",
                              "secured a work order from", "bagged the contract from"),
            description="Buyer awarded a contract to a supplier"),
        "FILED_CASE_AGAINST": RelationSpec(
            "FILED_CASE_AGAINST", ANY_ACTOR, ANY_ACTOR,
            ("filed a petition against", "filed a suit against", "filed a case against", "instituted proceedings against",
             "petitioner versus", "complaint against"),
            passive_triggers=("was sued by", "faces a suit filed by", "respondent in a petition filed by"),
            description="A party filed a case against another"),
        "ACQUIRED": RelationSpec(
            "ACQUIRED", ("ORGANISATION",), ("ORGANISATION",),
            ("acquired", "completed the acquisition of", "took over", "bought a controlling stake in"),
            passive_triggers=("was acquired by", "was taken over by"),
            description="One organisation acquired another"),
        "SHAREHOLDER_OF": RelationSpec(
            "SHAREHOLDER_OF", ANY_ACTOR, ("ORGANISATION",),
            ("holds shares in", "shareholder of", "holds a stake in", "holds equity in", "promoter of"),
            description="Disclosed shareholding"),
        **_COMMON,
    },
    document_classes={
        "corporate_filing": ("form dir-12", "registrar of companies", "cin", "din", "appointment of director", "mca"),
        "tender_award": ("tender", "bid", "work order", "letter of award", "l1", "e-procurement"),
        "court_judgment": ("petitioner", "respondent", "high court", "judgment", "hon'ble", "bench", "order"),
        "annual_report": ("annual report", "board's report", "financial year", "shareholders", "subsidiaries"),
        "news_report": ("reported", "sources said", "according to", "correspondent", "news"),
    },
)

INVESTIGATION = DomainPack(
    name="investigation",
    entity_types=("PERSON", "ORGANISATION", "GOVT_BODY", "LOCATION", "PHONE", "ACCOUNT", "VEHICLE", "CASE", "MONEY", "DATE"),
    relations={
        "ACCUSED_IN": RelationSpec(
            "ACCUSED_IN", PER, ("CASE",), ("named as accused in", "accused in", "arrested in connection with", "booked in"),
            description="Person is named as accused in a registered case (not a finding of guilt)"),
        "COMPLAINANT_IN": RelationSpec(
            "COMPLAINANT_IN", PER, ("CASE",), ("lodged", "complainant in", "filed the complaint"),
            description="Person is complainant"),
        "ASSOCIATE_OF": RelationSpec(
            "ASSOCIATE_OF", PER, PER, ("associate of", "accompanied by", "along with his associate", "known associate of", "worked with"),
            symmetric=True, description="Stated association in an official record"),
        "USES_PHONE": RelationSpec(
            "USES_PHONE", PER, ("PHONE",), ("uses mobile number", "using mobile number", "mobile number", "phone number", "contact number"),
            description="Subscriber/user of a phone number"),
        "HOLDS_ACCOUNT": RelationSpec(
            "HOLDS_ACCOUNT", ANY_ACTOR, ("ACCOUNT",), ("holds account", "account number", "a/c no", "bank account"),
            description="Account holder"),
        "OWNS_VEHICLE": RelationSpec(
            "OWNS_VEHICLE", PER, ("VEHICLE",), ("registered owner of", "owns vehicle", "vehicle bearing registration", "driving"),
            description="Vehicle ownership/use"),
        "CALLED": RelationSpec("CALLED", ("PHONE",), ("PHONE",), (), description="CDR call record"),
        "TRANSFERRED_TO": RelationSpec("TRANSFERRED_TO", ("ACCOUNT",), ("ACCOUNT",), (), description="Bank transfer"),
        **_COMMON,
    },
    document_classes={
        "fir": ("first information report", "fir no", "police station", "u/s", "complainant", "ipc", "bns"),
        "chargesheet": ("chargesheet", "final report", "investigating officer", "accused no"),
        "cdr": ("calling number", "called number", "imei", "cell id", "duration"),
        "bank_statement": ("ifsc", "utr", "debit", "credit", "balance", "narration"),
        "news_report": ("reported", "sources said", "according to"),
    },
    structured_relations=("CALLED", "TRANSFERRED_TO"),
    reporting_thresholds_inr=(50_000, 200_000, 1_000_000),
    notes="Outputs are investigative leads for human confirmation, never findings of guilt.",
)

PACKS = {p.name: p for p in (CORPORATE, INVESTIGATION)}


def get_pack(name: str) -> DomainPack:
    return PACKS[name]


def merged_pack() -> DomainPack:
    """Union pack so one case can fuse corporate + investigation sources."""
    rel = {**CORPORATE.relations, **INVESTIGATION.relations}
    types = tuple(dict.fromkeys(CORPORATE.entity_types + INVESTIGATION.entity_types))
    classes = {**CORPORATE.document_classes, **INVESTIGATION.document_classes}
    return DomainPack("fusion", types, rel, classes, INVESTIGATION.structured_relations,
                      INVESTIGATION.reporting_thresholds_inr)
