"""Shared test helpers."""

import hashlib

import numpy as np

from fixpilot.retrieval.bm25 import tokenize


class FakeEmbedder:
    """Deterministic fake: hashes words into a small vector. Records every query it sees."""

    def __init__(self):
        self.queries: list[str] = []

    def _vec(self, text):
        v = np.zeros(32, dtype=np.float32)
        for w in tokenize(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 32] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text):
        self.queries.append(text)
        return self._vec(text)
