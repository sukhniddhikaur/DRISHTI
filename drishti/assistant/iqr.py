"""Intermediate Query Representation (IQR): the only thing the NL layer may emit.

Allowlisted intents, bounded parameters. The executor turns an IQR into
parameterised store/graph calls — there is no path from user text (or an LLM)
to raw SQL/Cypher, and the case scope is re-checked against the caller's ACL.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Intent = Literal["find_paths", "find_related", "relation_lookup", "entity_profile", "timeline", "centrality", "communities",
                 "contradictions", "patterns", "hypotheses", "evidence_search"]
ALLOWED_PREDICATES = {"DIRECTOR_OF", "SUBSIDIARY_OF", "AWARDED_CONTRACT_TO", "FILED_CASE_AGAINST", "ACQUIRED", "SHAREHOLDER_OF",
                      "LOCATED_IN", "ACCUSED_IN", "COMPLAINANT_IN", "ASSOCIATE_OF", "USES_PHONE", "HOLDS_ACCOUNT", "OWNS_VEHICLE",
                      "CALLED", "TRANSFERRED_TO"}
ALLOWED_TYPES = {"PERSON", "ORGANISATION", "GOVT_BODY", "LOCATION", "PHONE", "ACCOUNT", "VEHICLE", "CASE", "CONTRACT"}


class IQR(BaseModel):
    intent: Intent
    case_id: str
    entity_ids: list[str] = Field(default_factory=list, max_length=2)
    entity_types: list[str] = Field(default_factory=list)
    predicates: list[str] = Field(default_factory=list)
    direction: Literal["out", "in", "any"] = "any"
    max_hops: int = Field(default=2, ge=1, le=4)
    as_of: str | None = None
    include_weak: bool = False
    include_evidence: bool = True
    limit: int = Field(default=10, ge=1, le=50)
    text: str = ""

    @field_validator("entity_types")
    @classmethod
    def _types(cls, v: list[str]) -> list[str]:
        bad = set(v) - ALLOWED_TYPES
        if bad:
            raise ValueError(f"entity types not allowed: {bad}")
        return v

    @field_validator("predicates")
    @classmethod
    def _preds(cls, v: list[str]) -> list[str]:
        bad = set(v) - ALLOWED_PREDICATES
        if bad:
            raise ValueError(f"predicates not allowed: {bad}")
        return v
