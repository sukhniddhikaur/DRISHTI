"""Confidence calibration.

Raw extractor scores are *not* probabilities. We fit isotonic regression per
extractor family on a labelled dev split and report Expected Calibration Error
(ECE) before/after. Until a calibrator is fitted, a conservative prior table
maps families to probabilities (rules > learned > co-occurrence).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

PRIOR = {"structured": 0.99, "rules": 0.85, "rules:by-passive": 0.8, "rules:right-context": 0.78, "hybrid": 0.85,
         "learned": 0.75, "cooccurrence": 0.15, "attachment": 0.8}


def family(extractor: str) -> str:
    return extractor.split("@")[0]


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    if len(probs) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if m.any():
            total += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(total)


class Calibrator:
    def __init__(self):
        self.models: dict[str, object] = {}

    def fit(self, extractors: list[str], scores: list[float], labels: list[int]) -> dict[str, dict[str, float]]:
        from sklearn.isotonic import IsotonicRegression

        report = {}
        fams = np.array([family(e) for e in extractors])
        s, y = np.array(scores, float), np.array(labels, int)
        for fam in sorted(set(fams)):
            m = fams == fam
            if m.sum() < 15 or len(set(y[m])) < 2:
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(s[m], y[m])
            self.models[fam] = iso
            report[fam] = {"n": int(m.sum()), "ece_raw": round(ece(s[m], y[m]), 4),
                           "ece_calibrated": round(ece(iso.predict(s[m]), y[m]), 4)}
        return report

    def __call__(self, extractor: str, raw: float) -> float:
        fam = family(extractor)
        if fam in self.models:
            return float(self.models[fam].predict([raw])[0])
        prior = PRIOR.get(fam, 0.6)
        return round(min(0.99, prior * (0.5 + raw / 2)), 4)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.models, f)

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibrator":
        c = cls()
        if path and Path(path).exists():
            with open(path, "rb") as f:
                c.models = pickle.load(f)
        return c
