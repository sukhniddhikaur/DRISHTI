"""Optional LLM verbaliser — rewrites verified findings into fluent prose.

Off by default. When enabled (DRISHTI_LLM=1 and ANTHROPIC_API_KEY set) the
model receives ONLY the already-verified sentences and their claim IDs, must
return JSON `[{"text": ..., "citations": [...]}]`, and every returned sentence
is re-verified: citations must be a subset of those provided, and every
capitalised name it contains must already occur in the provided sentences.
Anything failing is discarded and the deterministic template answer is kept.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

from drishti.assistant.engine import Answer, Sentence

PROMPT = """You rewrite investigative findings for an analyst. Rules:
- Use ONLY the facts in FINDINGS. Do not add entities, dates, numbers or relationships.
- Keep every uncertainty qualifier (alleged, reported, not independently corroborated, belief values).
- Each output sentence must cite the claim IDs of the findings it uses.
Return JSON only: [{"text": "...", "citations": ["C-..."]}]

FINDINGS:
"""


def enabled() -> bool:
    return os.environ.get("DRISHTI_LLM") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))


def _call(prompt: str) -> str:
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({"model": os.environ.get("DRISHTI_LLM_MODEL", "claude-sonnet-5-5"), "max_tokens": 1200,
                         "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read())
    return "".join(b.get("text", "") for b in body.get("content", []))


def verbalise(ans: Answer, call=_call) -> tuple[Answer, dict[str, Any]]:
    findings = [s for s in ans.sentences if s.citations]
    if not findings:
        return ans, {"used": False, "reason": "no findings"}
    allowed = {c for s in findings for c in s.citations}
    corpus = " ".join(s.text for s in ans.sentences)
    try:
        raw = call(PROMPT + "\n".join(f"- {s.text} {s.citations}" for s in findings))
        out = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
    except Exception as e:  # network, parsing, refusal: keep template answer
        return ans, {"used": False, "reason": type(e).__name__}
    kept, rejected = [], 0
    for o in out:
        cites = [c for c in o.get("citations", []) if isinstance(c, str)]
        names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", o.get("text", ""))
        if not cites or not set(cites) <= allowed or any(n not in corpus for n in names):
            rejected += 1
            continue
        kept.append(Sentence(o["text"], cites))
    if not kept:
        return ans, {"used": False, "reason": "all rewritten sentences failed verification", "rejected": rejected}
    meta = [s for s in ans.sentences if not s.citations]
    ans.sentences = meta[:1] + kept + meta[1:]
    return ans, {"used": True, "rejected": rejected}
