"""Evaluation metrics and gold<->system offset alignment."""
from __future__ import annotations

import difflib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


class OffsetMap:
    """Maps character offsets in a gold text onto a re-parsed text (whitespace/OCR drift)."""

    def __init__(self, src: str, dst: str):
        self.m = np.full(len(src) + 1, -1, dtype=np.int64)
        sm = difflib.SequenceMatcher(None, src, dst, autojunk=False)
        for a, b, n in sm.get_matching_blocks():
            self.m[a:a + n] = np.arange(b, b + n)
        self.m[len(src)] = len(dst)
        # forward-fill unmatched positions
        last = 0
        for i in range(len(self.m)):
            if self.m[i] < 0:
                self.m[i] = last
            else:
                last = self.m[i]

    def span(self, a: int, b: int) -> tuple[int, int]:
        x = int(self.m[a])
        y = int(self.m[max(a, b - 1)]) + 1
        return x, max(x, y)


def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def match_spans(gold: list[tuple[int, int, str]], pred: list[tuple[int, int, str]], strict: bool) -> tuple[int, int, int, dict]:
    used = set()
    tp = 0
    per = defaultdict(lambda: [0, 0, 0])
    for g in gold:
        best = None
        for j, p in enumerate(pred):
            if j in used or p[2] != g[2]:
                continue
            if (strict and p[0] == g[0] and p[1] == g[1]) or (not strict and overlap(g, p) > 0.5 * min(g[1] - g[0], p[1] - p[0])):
                best = j
                break
        if best is not None:
            used.add(best)
            tp += 1
            per[g[2]][0] += 1
        else:
            per[g[2]][2] += 1
    for j, p in enumerate(pred):
        if j not in used:
            per[p[2]][1] += 1
    return tp, len(pred) - tp, len(gold) - tp, {k: prf(*v) for k, v in per.items()}


def pairwise_er(pred: dict[str, str], gold: dict[str, str]) -> dict[str, float]:
    """pred/gold: item -> cluster id (same item keys). Counts via cluster contingency (no O(n²))."""
    items = [i for i in gold if i in pred]
    cont = Counter((pred[i], gold[i]) for i in items)
    pc = Counter(pred[i] for i in items)
    gc = Counter(gold[i] for i in items)
    c2 = lambda n: n * (n - 1) // 2
    tp = sum(c2(n) for n in cont.values())
    pp = sum(c2(n) for n in pc.values())
    gp = sum(c2(n) for n in gc.values())
    res = prf(tp, pp - tp, gp - tp)
    # B-cubed
    bp = sum(cont[(pred[i], gold[i])] / pc[pred[i]] for i in items) / max(1, len(items))
    br = sum(cont[(pred[i], gold[i])] / gc[gold[i]] for i in items) / max(1, len(items))
    res |= {"bcubed_precision": round(bp, 4), "bcubed_recall": round(br, 4),
            "bcubed_f1": round(2 * bp * br / (bp + br), 4) if bp + br else 0.0}
    golds_per_pred = defaultdict(set)
    for i in items:
        golds_per_pred[pred[i]].add(gold[i])
    multi = [p for p, g in golds_per_pred.items() if pc[p] > 1]
    res["false_merge_rate"] = round(sum(len(golds_per_pred[p]) > 1 for p in multi) / max(1, len(multi)), 4)
    preds_per_gold = defaultdict(set)
    for i in items:
        preds_per_gold[gold[i]].add(pred[i])
    res["fragmented_gold_rate"] = round(sum(len(v) > 1 for v in preds_per_gold.values()) / max(1, len(preds_per_gold)), 4)
    res["predicted_clusters"], res["gold_clusters"], res["items"] = len(pc), len(gc), len(items)
    return res


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked):
        if d in relevant:
            return 1 / (i + 1)
    return 0.0
