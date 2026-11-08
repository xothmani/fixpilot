"""CodeIndex: search a repo with BM25, embeddings, or both combined (hybrid)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize
from .chunker import Chunk, chunk_repo
from .embeddings import Embedder

METHODS = ("bm25", "embedding", "hybrid")
RRF_K = 60  # standard constant for reciprocal rank fusion


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1 = best


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal rank fusion. Each ranking is a list of chunk indexes, best first.

    A chunk's score is the sum of 1 / (k + rank) over the rankings. It needs no score tuning,
    because it only looks at positions, not at the raw BM25 or cosine numbers.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + pos)
    return fused


class CodeIndex:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, stemming: bool = True):
        self.chunks = chunks
        self.embedder = embedder
        self.stemming = stemming
        self._bm25 = BM25([tokenize(c.index_text(), stemming) for c in chunks])
        self._matrix = None
        if embedder is not None and chunks:
            self._matrix = embedder.embed_passages([c.index_text() for c in chunks])

    @classmethod
    def build(cls, repo_dir: Path, embedder: Embedder | None = None, stemming: bool = True):
        return cls(chunk_repo(Path(repo_dir)), embedder, stemming)

    def _rank_bm25(self, query: str) -> list[tuple[int, float]]:
        scores = self._bm25.scores(tokenize(query, self.stemming))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [(i, scores[i]) for i in order]

    def _rank_embedding(self, query: str) -> list[tuple[int, float]]:
        if self._matrix is None or self.embedder is None:
            raise ValueError("This index was built without an embedder")
        sims = self._matrix @ self.embedder.embed_query(query)
        order = np.argsort(-sims, kind="stable")
        return [(int(i), float(sims[i])) for i in order]

    def rank_all(self, query: str, method: str = "hybrid") -> list[Hit]:
        """Every chunk, best first. Chunks with no keyword match still appear (score 0)."""
        if method == "bm25":
            ranked = self._rank_bm25(query)
        elif method == "embedding":
            ranked = self._rank_embedding(query)
        elif method == "hybrid":
            fused = rrf_fuse(
                [[i for i, _ in self._rank_bm25(query)], [i for i, _ in self._rank_embedding(query)]]
            )
            ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        else:
            raise ValueError(f"Unknown method {method!r}. Use one of {METHODS}")
        return [Hit(self.chunks[i], s, r) for r, (i, s) in enumerate(ranked, start=1)]

    def search(self, query: str, k: int = 5, method: str = "hybrid") -> list[Hit]:
        return self.rank_all(query, method)[:k]
