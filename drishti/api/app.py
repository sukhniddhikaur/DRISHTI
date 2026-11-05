"""FastAPI application.

    uvicorn drishti.api.app:app            (env: DRISHTI_DB, DRISHTI_MODELS)
    or:  python -m drishti.cli serve

Auth: `X-API-Key` header -> user; the dev admin key is printed at startup.
All routes go through `Service`, which enforces role + per-case ACL.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from drishti.api.service import Service
from drishti.store.db import Store
from drishti.store.security import AccessDenied

STATIC = Path(__file__).parent / "static"


class AskBody(BaseModel):
    question: str
    use_llm: bool = False


class ReviewBody(BaseModel):
    decision: str
    reason: str = ""


class ResolveBody(BaseModel):
    left: str
    right: str
    link: bool
    reason: str = ""


class CaseBody(BaseModel):
    case_id: str
    name: str
    domain: str = "fusion"
    description: str = ""


class UserBody(BaseModel):
    user_id: str
    name: str
    role: str
    cases: list[str] = []


def create_app(db_path: str | None = None, model_dir: str | None = None) -> FastAPI:
    db_path = db_path or os.environ.get("DRISHTI_DB", "drishti_data/drishti.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    svc = Service(Store(db_path), model_dir=model_dir or os.environ.get("DRISHTI_MODELS", "models"),
                  rel_mode=os.environ.get("DRISHTI_REL_MODE", "hybrid"))
    app = FastAPI(title="DRISHTI", version="2.0", description="Evidence-first intelligence & knowledge-graph platform")
    app.state.svc = svc
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(AccessDenied)
    async def _denied(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(KeyError)
    async def _nf(_, exc):
        return JSONResponse({"detail": f"not found: {exc}"}, status_code=404)

    @app.exception_handler(ValueError)
    async def _bad(_, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    def user(x_api_key: str = Header(default="")) -> str:
        u = svc.sec.authenticate(x_api_key) if x_api_key else None
        if not u:
            raise HTTPException(401, "invalid or missing X-API-Key")
        return u["user_id"]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "audit": svc.audit.verify()}

    @app.get("/api/me")
    def me(u: str = Depends(user)):
        return svc.st.one("SELECT user_id, name, role FROM users WHERE user_id=?", (u,))

    @app.post("/api/users")
    def create_user(b: UserBody, u: str = Depends(user)):
        if svc.st.one("SELECT role FROM users WHERE user_id=?", (u,))["role"] != "admin":
            raise AccessDenied("admin only")
        key = svc.sec.create_user(b.user_id, b.name, b.role)
        for c in b.cases:
            svc.sec.grant(c, b.user_id)
        svc.audit.append(u, "user.create", "user", b.user_id, None, {"role": b.role, "cases": b.cases})
        return {"user_id": b.user_id, "api_key": key}

    @app.get("/api/cases")
    def cases(u: str = Depends(user)):
        return svc.cases(u)

    @app.post("/api/cases")
    def create_case(b: CaseBody, u: str = Depends(user)):
        return svc.create_case(u, b.case_id, b.name, b.domain, b.description)

    @app.post("/api/cases/{case}/documents")
    async def upload(case: str, files: list[UploadFile] = File(...), reliability: str = Form("user_upload"),
                     publication_date: str | None = Form(None), u: str = Depends(user)):
        return [svc.ingest(u, case, f.filename or "upload.bin", await f.read(), reliability, publication_date) for f in files]

    @app.post("/api/cases/{case}/process")
    def process(case: str, u: str = Depends(user)):
        return svc.process(u, case)

    @app.get("/api/cases/{case}/documents")
    def documents(case: str, u: str = Depends(user)):
        return svc.documents(u, case)

    @app.get("/api/documents/{doc}")
    def document(doc: str, u: str = Depends(user)):
        return svc.document(u, doc)

    @app.get("/api/cases/{case}/entities")
    def entities(case: str, type: str | None = None, q: str | None = None, u: str = Depends(user)):
        return svc.entities(u, case, type, q)

    @app.get("/api/entities/{eid}")
    def entity(eid: str, u: str = Depends(user)):
        return svc.entity(u, eid)

    @app.get("/api/claims/{cid}")
    def claim(cid: str, u: str = Depends(user)):
        return svc.claim(u, cid)

    @app.post("/api/claims/{cid}/review")
    def review(cid: str, b: ReviewBody, u: str = Depends(user)):
        return svc.review_claim(u, cid, b.decision, b.reason)

    @app.get("/api/cases/{case}/review-queue")
    def rq(case: str, u: str = Depends(user)):
        return svc.review_queue(u, case)

    @app.post("/api/cases/{case}/resolve")
    def resolve(case: str, b: ResolveBody, u: str = Depends(user)):
        return svc.resolve_pair(u, case, b.left, b.right, b.link, b.reason)

    @app.get("/api/cases/{case}/graph")
    def graph(case: str, as_of: str | None = None, include_weak: bool = False, min_belief: float = 0.0,
              types: list[str] | None = Query(None), u: str = Depends(user)):
        return svc.graph(u, case, as_of, include_weak, min_belief, types)

    @app.get("/api/cases/{case}/paths")
    def paths(case: str, a: str, b: str, k: int = 5, max_hops: int = 4, include_weak: bool = False, as_of: str | None = None,
              u: str = Depends(user)):
        return svc.paths(u, case, a, b, k, max_hops, include_weak, as_of)

    @app.get("/api/cases/{case}/analytics")
    def an(case: str, u: str = Depends(user)):
        return svc.analytics(u, case)

    @app.get("/api/cases/{case}/patterns")
    def pats(case: str, u: str = Depends(user)):
        return svc.patterns(u, case)

    @app.get("/api/cases/{case}/contradictions")
    def contra(case: str, u: str = Depends(user)):
        return svc.contradictions(u, case)

    @app.get("/api/cases/{case}/timeline")
    def tl(case: str, entity: str | None = None, u: str = Depends(user)):
        return svc.timeline(u, case, entity)

    @app.get("/api/cases/{case}/diff")
    def df(case: str, t1: str, t2: str, u: str = Depends(user)):
        return svc.diff(u, case, t1, t2)

    @app.get("/api/cases/{case}/search")
    def search(case: str, q: str, k: int = 10, mode: str = "hybrid", u: str = Depends(user)):
        return svc.search(u, case, q, k, mode)

    @app.post("/api/cases/{case}/ask")
    def ask(case: str, b: AskBody, u: str = Depends(user)):
        return svc.ask(u, case, b.question, b.use_llm)

    @app.get("/api/cases/{case}/audit")
    def audit(case: str, u: str = Depends(user)):
        return svc.audit_log(u, case)

    @app.get("/api/cases/{case}/report", response_class=HTMLResponse)
    def report(case: str, u: str = Depends(user)):
        return HTMLResponse(svc.report(u, case))

    return app


_APP: FastAPI | None = None


def __getattr__(name: str) -> Any:  # lazy `app` for `uvicorn drishti.api.app:app` (PEP 562)
    global _APP
    if name == "app":
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(name)
