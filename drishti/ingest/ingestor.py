"""Source-adapter ingestion with content addressing, dedup and versioning.

All adapters yield `RawItem`s; the Ingestor turns them into immutable,
content-addressed evidence plus a `DocumentRecord`. Duplicate bytes inside a
case are detected *before* any expensive processing. Changed bytes for the same
(source, title) become version n+1 linked to the previous version.
"""
from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from drishti.core.ids import sha256_bytes, stable_id
from drishti.core.models import DocumentRecord, ReliabilityClass
from drishti.store.audit import AuditLog
from drishti.store.db import Store


@dataclass
class RawItem:
    data: bytes
    title: str
    source_id: str
    mime: str | None = None
    source_url: str | None = None
    access_method: str = "upload"
    reliability: ReliabilityClass = ReliabilityClass.USER_UPLOAD
    publication_date: str | None = None
    document_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceAdapter(Protocol):
    source_id: str

    def items(self) -> Iterator[RawItem]: ...


class DirectoryAdapter:
    """Reads a folder; optional sidecar `<file>.meta.json` supplies provenance."""

    def __init__(self, root: str | Path, source_id: str = "local-dir",
                 reliability: ReliabilityClass = ReliabilityClass.USER_UPLOAD):
        self.root, self.source_id, self.reliability = Path(root), source_id, reliability

    def items(self) -> Iterator[RawItem]:
        for p in sorted(self.root.rglob("*")):
            if not p.is_file() or p.name.endswith(".meta.json") or p.name.startswith("."):
                continue
            meta: dict[str, Any] = {}
            side = p.with_name(p.name + ".meta.json")
            if side.exists():
                meta = json.loads(side.read_text())
            yield RawItem(
                data=p.read_bytes(), title=meta.pop("title", p.name), source_id=meta.pop("source_id", self.source_id),
                mime=meta.pop("mime", None) or mimetypes.guess_type(p.name)[0],
                source_url=meta.pop("source_url", None), access_method=meta.pop("access_method", "upload"),
                reliability=ReliabilityClass(meta.pop("reliability", self.reliability.value)),
                publication_date=meta.pop("publication_date", None), document_type=meta.pop("document_type", None),
                metadata=meta)


class MemoryAdapter:
    def __init__(self, items: Iterable[RawItem], source_id: str = "memory"):
        self._items, self.source_id = list(items), source_id

    def items(self) -> Iterator[RawItem]:
        yield from self._items


@dataclass
class IngestResult:
    document_id: str
    status: str  # new | duplicate | new_version
    version: int


class Ingestor:
    def __init__(self, store: Store, blob_root: str | Path, audit: AuditLog | None = None):
        self.store, self.blob_root = store, Path(blob_root)
        self.audit = audit or AuditLog(store)

    def _blob(self, digest: str, data: bytes) -> str:
        p = self.blob_root / digest[:2] / digest[2:4] / digest
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            p.chmod(0o444)  # immutable evidence copy
        return p.as_uri()

    def ingest(self, case_id: str, item: RawItem, actor: str = "system") -> IngestResult:
        digest = sha256_bytes(item.data)
        dup = self.store.find_document_by_hash(case_id, digest)
        if dup:
            self.audit.append(actor, "ingest.duplicate", "document", dup["document_id"], case_id, {"hash": digest})
            return IngestResult(dup["document_id"], "duplicate", dup["version"])
        prev = self.store.latest_version(case_id, item.source_id, item.title)
        version = prev["version"] + 1 if prev else 1
        doc = DocumentRecord(
            document_id=stable_id("DOC", case_id, digest), case_id=case_id, source_id=item.source_id,
            title=item.title, document_type=item.document_type or "unknown",
            mime=item.mime or mimetypes.guess_type(item.title)[0] or "application/octet-stream",
            publication_date=item.publication_date, file_hash=digest, storage_uri=self._blob(digest, item.data),
            source_url=item.source_url, access_method=item.access_method, reliability=item.reliability,
            version=version, parent_document_id=prev["document_id"] if prev else None, metadata=item.metadata)
        self.store.save_document(doc)
        self.audit.append(actor, "ingest.new" if version == 1 else "ingest.new_version", "document", doc.document_id,
                          case_id, {"hash": digest, "title": item.title, "version": version, "source": item.source_id})
        return IngestResult(doc.document_id, "new" if version == 1 else "new_version", version)

    def ingest_all(self, case_id: str, adapter: SourceAdapter, actor: str = "system") -> list[IngestResult]:
        return [self.ingest(case_id, it, actor) for it in adapter.items()]
