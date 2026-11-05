"""Document understanding: bytes -> canonical text + structural passages.

Model cascade (cheapest first):
  1. native text layer (PyMuPDF / python-docx / csv)      ~ms per page
  2. OCR only for pages whose text layer is missing/garbled (Tesseract,
     with OpenCV deskew + adaptive threshold when available)
  3. quality flags route poor pages to human review instead of silently
     feeding noise to extraction.

Every passage carries [char_start, char_end) into the canonical text and its
page number; tables are kept row-by-row (header-qualified) instead of being
flattened into prose.
"""
from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from drishti.core.ids import stable_id
from drishti.core.models import Passage


@dataclass
class ParsedDocument:
    text: str
    passages: list[Passage]
    pages: int = 1
    tables: list[dict[str, Any]] = field(default_factory=list)  # {"header": [...], "rows": [[...]], "passage_ids": [...]}
    page_methods: dict[int, str] = field(default_factory=dict)  # page -> native|ocr
    ocr_confidence: dict[int, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class _Builder:
    def __init__(self, document_id: str):
        self.doc_id, self.buf, self.passages = document_id, [], []
        self.pos = 0

    def add(self, text: str, page: int, kind: str = "paragraph") -> str | None:
        text = text.strip("\n")
        if not text.strip():
            return None
        if self.buf:
            self.buf.append("\n\n")
            self.pos += 2
        start = self.pos
        self.buf.append(text)
        self.pos += len(text)
        pid = stable_id("PSG", self.doc_id, start)
        self.passages.append(Passage(passage_id=pid, document_id=self.doc_id, page=page, kind=kind,
                                     char_start=start, char_end=self.pos, text=text))
        return pid

    def text(self) -> str:
        return "".join(self.buf)


_HEADING = re.compile(r"^(?:[A-Z][A-Z0-9 &.,'()/-]{3,80}|\d+(?:\.\d+)*\s+[A-Z].{0,80})$")


def _split_paragraphs(page_text: str) -> list[str]:
    page_text = page_text.replace("\r", "")
    blocks = re.split(r"\n\s*\n", page_text)
    out = []
    for b in blocks:
        # de-hyphenate line wraps and join soft line breaks within a paragraph
        b = re.sub(r"(\w)-\n(\w)", r"\1\2", b)
        b = re.sub(r"[ \t]*\n[ \t]*", " ", b).strip()
        if b:
            out.append(b)
    return out


def _garbage_ratio(s: str) -> float:
    if not s:
        return 1.0
    bad = sum(1 for ch in s if not (ch.isalnum() or ch.isspace() or ch in ".,;:'\"()/-&@₹%[]"))
    return bad / len(s)


# ----------------------------------------------------------------------------- OCR
def _ocr_image_bytes(png: bytes) -> tuple[str, float]:
    """Tesseract OCR with optional OpenCV preprocessing; returns (text, mean word confidence)."""
    if not shutil.which("tesseract"):
        return "", 0.0
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "page.png"
        src.write_bytes(png)
        try:
            import cv2  # type: ignore
            import numpy as np
            img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
            img = cv2.fastNlMeansDenoising(img, h=10)
            coords = np.column_stack(np.where(img < 128))
            if len(coords) > 50:
                angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
                angle = -(90 + angle) if angle < -45 else -angle
                if abs(angle) > 0.3:
                    h, w = img.shape
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
            cv2.imwrite(str(src), img)
        except Exception:
            pass  # OpenCV optional: fall back to raw image
        tsv = subprocess.run(["tesseract", str(src), "stdout", "--psm", "4", "tsv"], capture_output=True, text=True, timeout=120).stdout
    words, confs, lines, last = [], [], [], None
    for row in tsv.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or not cols[11].strip():
            continue
        key = (cols[2], cols[3], cols[4])  # block, par, line
        if last is not None and key[:2] != last[:2]:
            lines.append("\n\n")
        elif last is not None and key != last:
            lines.append("\n")
        elif last is not None:
            lines.append(" ")
        lines.append(cols[11])
        last = key
        try:
            c = float(cols[10])
            if c >= 0:
                confs.append(c)
        except ValueError:
            pass
        words.append(cols[11])
    return "".join(lines), (sum(confs) / len(confs) / 100.0) if confs else 0.0


# ----------------------------------------------------------------------------- parsers
def parse_pdf(data: bytes, document_id: str, min_chars: int = 40) -> ParsedDocument:
    import pymupdf  # type: ignore

    b = _Builder(document_id)
    pd = ParsedDocument("", [])
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        pd.pages = pdf.page_count
        for i, page in enumerate(pdf, start=1):
            native = page.get_text("text") or ""
            method = "native"
            if len(native.strip()) < min_chars or _garbage_ratio(native) > 0.3:
                png = page.get_pixmap(dpi=300).tobytes("png")
                text, conf = _ocr_image_bytes(png)
                if text.strip():
                    native, method = text, "ocr"
                    pd.ocr_confidence[i] = round(conf, 3)
                else:
                    pd.warnings.append(f"page {i}: no extractable text")
            pd.page_methods[i] = method
            # tables: PyMuPDF table finder on native pages
            if method == "native":
                try:
                    for t in page.find_tables().tables:
                        rows = [[(c or "").strip() for c in r] for r in t.extract()]
                        if len(rows) >= 2:
                            _emit_table(b, pd, rows[0], rows[1:], i)
                except Exception:
                    pass
            for para in _split_paragraphs(native):
                b.add(para, i, "heading" if _HEADING.match(para) else "paragraph")
    pd.text, pd.passages = b.text(), b.passages
    return pd


def parse_docx(data: bytes, document_id: str) -> ParsedDocument:
    import docx  # type: ignore

    d = docx.Document(io.BytesIO(data))
    b = _Builder(document_id)
    pd = ParsedDocument("", [])
    for p in d.paragraphs:
        style = (p.style.name or "").lower() if p.style is not None else ""
        b.add(p.text, 1, "heading" if style.startswith("heading") or style == "title" else "paragraph")
    for t in d.tables:
        rows = [[c.text.strip() for c in r.cells] for r in t.rows]
        if len(rows) >= 2:
            _emit_table(b, pd, rows[0], rows[1:], 1)
    pd.text, pd.passages = b.text(), b.passages
    return pd


def _emit_table(b: _Builder, pd: ParsedDocument, header: list[str], rows: list[list[str]], page: int) -> None:
    header = [h.strip() or f"col{j}" for j, h in enumerate(header)]
    pids = []
    for r in rows:
        line = "; ".join(f"{h}: {v}" for h, v in zip(header, r) if str(v).strip())
        pid = b.add(line, page, "table_row")
        pids.append(pid)
    pd.tables.append({"header": header, "rows": rows, "passage_ids": pids})


def parse_tabular(data: bytes, document_id: str, mime: str) -> ParsedDocument:
    b = _Builder(document_id)
    pd = ParsedDocument("", [])
    if "csv" in mime or mime.startswith("text/"):
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig", errors="replace"))))
        sheets = [("sheet1", rows)]
    else:
        import openpyxl  # type: ignore
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sheets = [(ws.title, [["" if c is None else str(c) for c in r] for r in ws.iter_rows(values_only=True)]) for ws in wb]
    for _, rows in sheets:
        rows = [r for r in rows if any(str(c).strip() for c in r)]
        if len(rows) >= 2:
            _emit_table(b, pd, rows[0], rows[1:], 1)
    pd.text, pd.passages = b.text(), b.passages
    return pd


def parse_text(data: bytes, document_id: str) -> ParsedDocument:
    raw = data.decode("utf-8", errors="replace")
    if raw.lstrip().startswith(("<html", "<!doctype", "<HTML", "<!DOCTYPE")):
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        raw = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>|</li>", "\n\n", raw, flags=re.I)
        raw = re.sub(r"<[^>]+>", " ", raw)
    b = _Builder(document_id)
    pages = raw.split("\f")
    for i, pg in enumerate(pages, start=1):
        for para in _split_paragraphs(pg):
            b.add(para, i, "heading" if _HEADING.match(para) else "paragraph")
    return ParsedDocument(b.text(), b.passages, pages=len(pages))


def parse_json(data: bytes, document_id: str) -> ParsedDocument:
    obj = json.loads(data)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        header = list(dict.fromkeys(k for o in obj for k in o))
        b = _Builder(document_id)
        pd = ParsedDocument("", [])
        _emit_table(b, pd, header, [[str(o.get(h, "")) for h in header] for o in obj], 1)
        pd.text, pd.passages = b.text(), b.passages
        return pd
    return parse_text(json.dumps(obj, indent=1, ensure_ascii=False).encode(), document_id)


def parse_image(data: bytes, document_id: str) -> ParsedDocument:
    text, conf = _ocr_image_bytes(data)
    pd = parse_text(text.encode(), document_id)
    pd.page_methods, pd.ocr_confidence = {1: "ocr"}, {1: round(conf, 3)}
    return pd


def parse(data: bytes, document_id: str, mime: str, title: str = "") -> ParsedDocument:
    t = title.lower()
    if mime == "application/pdf" or t.endswith(".pdf"):
        return parse_pdf(data, document_id)
    if "wordprocessingml" in mime or t.endswith(".docx"):
        return parse_docx(data, document_id)
    if "spreadsheetml" in mime or t.endswith((".xlsx", ".xlsm")) or "csv" in mime or t.endswith(".csv"):
        return parse_tabular(data, document_id, "text/csv" if t.endswith(".csv") or "csv" in mime else mime)
    if mime.startswith("image/"):
        return parse_image(data, document_id)
    if "json" in mime or t.endswith(".json"):
        return parse_json(data, document_id)
    return parse_text(data, document_id)
