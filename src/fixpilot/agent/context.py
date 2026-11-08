"""Everything a tool needs: the repo copy, the state, the logger, and the search index."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fixpilot.config import SandboxConfig
from fixpilot.retrieval import CodeIndex
from fixpilot.retrieval.embeddings import Embedder

from .state import AgentState
from .tracing import Tracer

TEST_MODES = ("realistic", "blind")  # condition A and condition B of the measurement plan


@dataclass
class ToolContext:
    root: Path  # the temporary repo copy (never the original repo)
    state: AgentState
    tracer: Tracer = field(default_factory=Tracer)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig.from_env)
    test_mode: str = "realistic"
    search_enabled: bool = True  # False = the "no retrieval" baseline
    search_method: str = "hybrid"
    embedder: Embedder | None = None
    snapshot: dict[str, str] = field(default_factory=dict)  # original text of edited files
    _index: CodeIndex | None = field(default=None, repr=False)
    index_dirty: bool = True

    def __post_init__(self) -> None:
        if self.test_mode not in TEST_MODES:
            raise ValueError(f"test_mode must be one of {TEST_MODES}")
        self.root = Path(self.root).resolve()

    @property
    def effective_search_method(self) -> str:
        """Hybrid and embedding search need an embedder. Without one, fall back to BM25."""
        if self.embedder is None and self.search_method != "bm25":
            return "bm25"
        return self.search_method

    def get_index(self) -> CodeIndex:
        """The search index. It is rebuilt after a patch so results never show stale code."""
        if self._index is None or self.index_dirty:
            self._index = CodeIndex.build(self.root, self.embedder)
            self.index_dirty = False
        return self._index
