"""String similarity + MinHash LSH (pure Python/numpy; rapidfuzz used if present)."""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

try:  # optional C++ acceleration
    from rapidfuzz.distance import JaroWinkler as _RFJW  # type: ignore
except Exception:  # pragma: no cover
    _RFJW = None


def jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    win = max(la, lb) // 2 - 1
    ma, mb = [False] * la, [False] * lb
    m = 0
    for i, ch in enumerate(a):
        lo, hi = max(0, i - win), min(i + win + 1, lb)
        for j in range(lo, hi):
            if not mb[j] and b[j] == ch:
                ma[i] = mb[j] = True
                m += 1
                break
    if not m:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            if a[i] != b[k]:
                t += 1
            k += 1
    return (m / la + m / lb + (m - t / 2) / m) / 3


@lru_cache(maxsize=200_000)
def jaro_winkler(a: str, b: str, p: float = 0.1) -> float:
    if _RFJW is not None:
        return float(_RFJW.similarity(a, b))
    j = jaro(a, b)
    pre = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        pre += 1
    return j + pre * p * (1 - j)


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def soft_token_overlap(a: str, b: str, thr: float = 0.9) -> float:
    """Monge-Elkan style: average best JW per token (symmetric)."""
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0

    def one(x, y):
        return sum(max(jaro_winkler(t, u) for u in y) for t in x) / len(x)
    return (one(ta, tb) + one(tb, ta)) / 2


def ngrams(s: str, n: int = 3) -> set[str]:
    s = f"  {s} "
    return {s[i:i + n] for i in range(len(s) - n + 1)}


class MinHashLSH:
    """MinHash signatures over character n-grams, banded LSH for candidate pairs.

    With b bands of r rows, P(candidate | Jaccard=s) = 1-(1-s^r)^b; defaults
    (b=10, r=3) give ~0.5 at s≈0.46 and ~0.97 at s=0.7.
    """

    def __init__(self, num_perm: int = 30, bands: int = 10, seed: int = 7):
        assert num_perm % bands == 0
        self.k, self.b, self.r = num_perm, bands, num_perm // bands
        rng = np.random.default_rng(seed)
        self.p = (1 << 61) - 1
        self.A = rng.integers(1, self.p, size=num_perm, dtype=np.uint64)
        self.B = rng.integers(0, self.p, size=num_perm, dtype=np.uint64)

    def signature(self, grams: set[str]) -> np.ndarray:
        if not grams:
            return np.full(self.k, np.iinfo(np.uint64).max, dtype=np.uint64)
        h = np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") >> 4 for g in grams],
                     dtype=np.uint64)
        # (a*h + b) mod p, computed in uint64 with wraparound acceptable for hashing purposes
        v = (self.A[:, None] * h[None, :] + self.B[:, None]) % np.uint64(self.p)
        return v.min(axis=1)

    def bands(self, sig: np.ndarray) -> list[tuple[int, bytes]]:
        return [(i, sig[i * self.r:(i + 1) * self.r].tobytes()) for i in range(self.b)]
