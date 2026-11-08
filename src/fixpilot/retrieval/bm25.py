"""Keyword search: BM25 (Okapi) written out in full, so there is no extra dependency.

BM25 scores a chunk higher when it contains the query words, especially rare ones, and it
does not reward very long chunks just for being long.
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    "a an and are as at be by for from has have if in is it its of on or that the this to was "  # noqa: SIM905
    "were will with should would could when what which who not no but so than then there their "
    "they them these those into out up we you your i he she do does did can py self return".split()
)


def stem(word: str) -> str:
    """A very small suffix stripper so 'expired', 'expires' and 'expiry' meet in the middle."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _split_identifier(word: str) -> list[str]:
    """'lowStock' -> ['low', 'Stock']; snake_case is split earlier by the word regex."""
    return re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word) or [word]


def tokenize(text: str, use_stemming: bool = True) -> list[str]:
    """Lowercase words. snake_case and camelCase names are split AND kept whole."""
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        pieces: list[str] = []
        for part in raw.split("_"):
            if part:
                pieces.extend(_split_identifier(part))
        lowered = [p.lower() for p in pieces]
        words = [w for w in lowered if w not in STOPWORDS and len(w) > 1]
        if use_stemming:
            words = [stem(w) for w in words]
        tokens.extend(words)
        if len(lowered) > 1:  # keep the whole identifier too, e.g. 'low_stock'
            whole = raw.lower().replace("_", "")
            tokens.append(stem(whole) if use_stemming else whole)
    return tokens


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tf = [Counter(d) for d in documents]
        self.doc_len = [len(d) for d in documents]
        self.avg_len = (sum(self.doc_len) / len(documents)) if documents else 0.0
        df: Counter[str] = Counter()
        for tf in self.doc_tf:
            df.update(tf.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        result = [0.0] * len(self.doc_tf)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.doc_tf):
                f = tf.get(term, 0)
                if f:
                    norm = 1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1)
                    result[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return result
