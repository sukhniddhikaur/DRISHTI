"""Canonical, schema-validated records shared by every stage.

Design rules
------------
* Documents are immutable evidence. Re-ingesting changed bytes yields a new
  version linked to its parent; it never overwrites.
* A Mention is a span in a document. An Entity is a *resolution* of mentions;
  the mention->entity mapping is a separate, reversible table.
* A Claim is the unit of knowledge: (subject, predicate, object) + the exact
  evidence span + modality + bitemporal times + provenance. Graph edges are
  aggregations of claims, computed on demand.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReliabilityClass(str, Enum):
    """Evidentiary role of a source. Not a truth score — a provenance category."""
    OFFICIAL_RECORD = "official_record"        # court judgment, registry filing, FIR
    STRUCTURED_RECORD = "structured_record"    # CDR, bank statement, tabular export
    CORPORATE_DISCLOSURE = "corporate_disclosure"  # annual report (self-reported)
    JOURNALISM = "journalism"
    USER_UPLOAD = "user_upload"


class Modality(str, Enum):
    """What the source *says* about the relation."""
    ASSERTED = "asserted"     # "X is a director of Y"
    NEGATED = "negated"       # "X was never a director of Y"
    HEDGED = "hedged"         # "X is allegedly linked to Y"
    REPORTED = "reported"     # "According to the filing, X ..." (second-hand)


class Epistemic(str, Enum):
    """How DRISHTI knows it. Prevents interpretations masquerading as facts."""
    STATED = "stated"          # deterministic parse of a structured record
    EXTRACTED = "extracted"    # model-extracted from text, unreviewed
    INFERRED = "inferred"      # produced by analytics (link prediction etc.)
    CONFIRMED = "confirmed"    # analyst approved
    REJECTED = "rejected"      # analyst rejected


class DocumentRecord(BaseModel):
    document_id: str
    case_id: str
    source_id: str
    title: str
    document_type: str = "unknown"
    mime: str = "text/plain"
    publication_date: str | None = None
    retrieval_date: str = Field(default_factory=utcnow)
    file_hash: str
    storage_uri: str
    source_url: str | None = None
    access_method: str = "upload"
    reliability: ReliabilityClass = ReliabilityClass.USER_UPLOAD
    version: int = 1
    parent_document_id: str | None = None
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    text: str = ""
    quality: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Passage(BaseModel):
    """A structural unit (paragraph, table row, heading) with document offsets."""
    passage_id: str
    document_id: str
    page: int = 1
    kind: str = "paragraph"
    char_start: int
    char_end: int
    text: str

    @model_validator(mode="after")
    def _span(self) -> "Passage":
        if self.char_end < self.char_start:
            raise ValueError("passage span inverted")
        return self


class Mention(BaseModel):
    mention_id: str
    document_id: str
    passage_id: str | None = None
    label: str
    text: str
    char_start: int
    char_end: int
    normalized: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)
    extractor: str = "rules"
    extractor_version: str = "1"
    confidence: float = 1.0

    @field_validator("confidence")
    @classmethod
    def _p(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be a probability")
        return v


class Entity(BaseModel):
    entity_id: str
    case_id: str
    type: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    identifiers: dict[str, list[str]] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
    mention_ids: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    claim_id: str
    case_id: str
    predicate: str
    subject_mention_id: str | None = None
    object_mention_id: str | None = None
    subject_entity_id: str | None = None
    object_entity_id: str | None = None
    subject_text: str = ""
    object_text: str = ""
    document_id: str
    passage_id: str | None = None
    evidence_text: str
    char_start: int
    char_end: int
    page: int = 1
    modality: Modality = Modality.ASSERTED
    epistemic: Epistemic = Epistemic.EXTRACTED
    valid_from: str | None = None     # when the relationship held (world time)
    valid_to: str | None = None
    reported_at: str | None = None    # when the source said so (publication)
    ingested_at: str = Field(default_factory=utcnow)  # when DRISHTI learned it
    extractor: str = "rules"
    extractor_version: str = "1"
    raw_score: float = 1.0
    confidence: float = 1.0           # calibrated probability
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: str = "unreviewed"        # unreviewed | confirmed | rejected | disputed

    @field_validator("valid_from", "valid_to", "reported_at")
    @classmethod
    def _iso(cls, v: str | None) -> str | None:
        if v is None:
            return v
        date.fromisoformat(v[:10])
        return v


class Event(BaseModel):
    event_id: str
    case_id: str
    event_type: str
    date: str | None = None
    document_id: str
    passage_id: str | None = None
    evidence_text: str = ""
    char_start: int = 0
    char_end: int = 0
    roles: dict[str, str] = Field(default_factory=dict)  # role -> mention_id
    attributes: dict[str, Any] = Field(default_factory=dict)
