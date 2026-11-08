"""Meaning-based search: turn text into vectors with a small local model (fastembed).

Everything runs on this computer. The model is downloaded once into `.cache/fastembed`.
Vectors are cached on disk by a hash of (model, text), so unchanged code is never re-embedded.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from fixpilot.config import PROJECT_ROOT

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
CACHE_DIR = PROJECT_ROOT / ".cache"


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """Embeds code chunks and queries. Vectors are unit length, so a dot product is cosine."""

    def __init__(self, model_name: str | None = None, cache_dir: Path = CACHE_DIR):
        self.model_name = model_name or os.environ.get("FIXPILOT_EMBED_MODEL", DEFAULT_MODEL)
        self._cache_dir = Path(cache_dir)
        self._model = None
        self._vectors: dict[str, np.ndarray] = {}
        safe = self.model_name.replace("/", "_")
        self._file = self._cache_dir / "embeddings" / f"{safe}.npz"
        self._load()

    def _load(self) -> None:
        if self._file.exists():
            with np.load(self._file) as data:
                self._vectors = {k: data[k] for k in data.files}

    def save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._file, **self._vectors)

    def _model_instance(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, cache_dir=str(self._cache_dir / "fastembed"))
        return self._model

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\n{text}".encode()).hexdigest()

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in self._vectors]
        if missing:
            vectors = list(self._model_instance().embed([t for _, t in missing]))
            for (k, _), v in zip(missing, vectors, strict=True):
                self._vectors[k] = np.asarray(v, dtype=np.float32)
        return np.vstack([self._vectors[k] for k in keys]) if keys else np.zeros((0, 1))

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model_instance().query_embed(text))), dtype=np.float32)
