"""Hybrid evidence retrieval over passages.

    Score(d, q) = α·S_BM25 + β·S_semantic + γ·S_metadata     (min-max normalised per query)

* BM25 (Okapi, k1=1.5, b=0.75) implemented from scratch over an inverted index.
* Semantic: LSA — TF-IDF (word 1-2 grams + char 3-5 grams) projected with
  truncated SVD; cosine in the latent space. Swappable for Sentence-Transformers
  through `SemanticBackend` when model weights are available.
* Metadata: document type / reliability / date / case filters as soft boosts;
  hard filters are applied before scoring.
* Fusion: weighted sum (weights tuned on labelled dev queries by grid search)
  or Reciprocal Rank Fusion (k=60) when no labels exist.
* Entity-aware expansion: query mentions are linked to resolved entities and
  expanded with their aliases, so "ABC Infra" also retrieves
  "ABC Infrastructure Private Limited".
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an of to in on for by and or is was were be been are with as at from that this which who whom what when where "
           "how did does do has have had its it their his her into under over".split())


def tokenize(s: str) -> list[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        self.N = len(docs)
        df = Counter()
        self.index: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.tf):
            for w in t:
                df[w] += 1
                self.index[w].append(i)
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.N)
        for w in set(tokenize(query)):
            if w not in self.index:
                continue
            idf = self.idf[w]
            for i in self.index[w]:
                f = self.tf[i][w]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1)))
        return s


class LSA:
    def __init__(self, docs: list[str], dim: int = 128, seed: int = 7):
        from scipy.sparse import hstack
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words=list(STOP))
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1, max_features=60000)
        X = hstack([self.w.fit_transform(docs), self.c.fit_transform(docs)]).tocsr()
        k = max(2, min(dim, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=seed)
        Z = self.svd.fit_transform(X)
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        self._hstack = hstack

    def scores(self, query: str) -> np.ndarray:
        q = self._hstack([self.w.transform([query]), self.c.transform([query])]).tocsr()
        z = self.svd.transform(q)[0]
        z = z / (np.linalg.norm(z) + 1e-9)
        return self.Z @ z


def _mm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


@dataclass
class Hit:
    passage_id: str
    document_id: str
    score: float
    text: str
    components: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, passages: list[dict[str, Any]], docs: dict[str, dict[str, Any]] | None = None,
                 alpha: float = 0.5, beta: float = 0.4, gamma: float = 0.1, fusion: str = "weighted",
                 alias_index: dict[str, list[str]] | None = None):
        self.p = passages
        self.docs = docs or {}
        self.alpha, self.beta, self.gamma, self.fusion = alpha, beta, gamma, fusion
        texts = [x["text"] for x in passages]
        self.bm25 = BM25(texts)
        self.lsa = LSA(texts) if len(texts) >= 3 else None
        self.alias_index = alias_index or {}

    def expand(self, query: str) -> str:
        low = query.lower()
        extra = []
        for surface, aliases in self.alias_index.items():
            if len(surface) > 3 and surface in low:
                extra += aliases
        return query + (" " + " ".join(extra) if extra else "")

    def _meta(self, prefer_types: set[str] | None, prefer_reliability: set[str] | None) -> np.ndarray:
        s = np.zeros(len(self.p))
        for i, x in enumerate(self.p):
            d = self.docs.get(x["document_id"], {})
            if prefer_types and d.get("document_type") in prefer_types:
                s[i] += 0.6
            if prefer_reliability and d.get("reliability") in prefer_reliability:
                s[i] += 0.4
            if x.get("kind") == "heading":
                s[i] -= 0.3
        return s

    def search(self, query: str, k: int = 10, mode: str | None = None, filters: Callable[[dict], bool] | None = None,
               prefer_types: set[str] | None = None, prefer_reliability: set[str] | None = None,
               expand: bool = True) -> list[Hit]:
        q = self.expand(query) if expand else query
        mode = mode or "hybrid"
        b = self.bm25.scores(q)
        s = self.lsa.scores(q) if self.lsa is not None else np.zeros(len(self.p))
        m = self._meta(prefer_types, prefer_reliability)
        if mode == "bm25":
            final = b
        elif mode == "semantic":
            final = s
        elif self.fusion == "rrf":
            rb, rs = np.argsort(-b).argsort(), np.argsort(-s).argsort()
            final = 1 / (60 + rb) + 1 / (60 + rs) + 0.002 * m
        else:
            final = self.alpha * _mm(b) + self.beta * _mm(s) + self.gamma * _mm(m)
        order = np.argsort(-final)
        out = []
        for i in order:
            x = self.p[int(i)]
            if filters and not filters(x):
                continue
            out.append(Hit(x["passage_id"], x["document_id"], float(final[i]), x["text"],
                           {"bm25": float(b[i]), "semantic": float(s[i]), "metadata": float(m[i])}))
            if len(out) >= k:
                break
        return out

    def tune(self, queries: list[tuple[str, set[str]]], k: int = 10, grid: Iterable[float] = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1)) -> dict[str, float]:
        """Grid-search α,β (γ = 1-α-β) maximising mean nDCG@k on labelled dev queries."""
        from drishti.bench.metrics import ndcg_at_k
        best = (-1.0, self.alpha, self.beta, self.gamma)
        cache = [(self.bm25.scores(self.expand(q)), self.lsa.scores(self.expand(q)) if self.lsa else np.zeros(len(self.p)), rel)
                 for q, rel in queries]
        meta = _mm(self._meta(None, None))
        ids = [x["passage_id"] for x in self.p]
        for a in grid:
            for b_ in grid:
                g = round(1 - a - b_, 4)
                if g < 0 or g > 0.3:   # metadata is a tie-breaker, never the main signal
                    continue
                tot = 0.0
                for bs, ss, rel in cache:
                    f = a * _mm(bs) + b_ * _mm(ss) + g * meta
                    ranked = [ids[i] for i in np.argsort(-f)[:k]]
                    tot += ndcg_at_k(ranked, rel, k)
                score = tot / max(1, len(cache))
                if score > best[0] + 1e-9:
                    best = (score, a, b_, g)
        _, self.alpha, self.beta, self.gamma = best
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "dev_ndcg": round(best[0], 4)}
