import pytest
from helpers import FakeEmbedder

from fixpilot.benchmark import load_bugs
from fixpilot.config import BENCHMARK_REPOS
from fixpilot.retrieval import CodeIndex, chunk_repo, chunk_source
from fixpilot.retrieval.bm25 import BM25, stem, tokenize
from fixpilot.retrieval.chunker import is_source_file
from fixpilot.retrieval.evaluate import QueryResult, decoy_verdict, evaluate, recall
from fixpilot.retrieval.index import rrf_fuse

SAMPLE = '''"""module doc"""
import math

LIMIT = 5


def top(x):
    """Return x."""
    return x


class Box:
    """A box."""

    size: int = 3

    @property
    def area(self):
        return self.size ** 2

    def grow(self):
        return Box()
'''


def test_chunker_splits_functions_methods_class_header_and_constants():
    chunks = {c.symbol: c for c in chunk_source(SAMPLE, "pkg/box.py")}
    assert set(chunks) == {"top", "Box", "Box.area", "Box.grow", "<module>"}
    assert chunks["top"].start_line == 7 and chunks["top"].end_line == 9
    assert chunks["Box"].kind == "class" and "size: int = 3" in chunks["Box"].code
    assert "def grow" not in chunks["Box"].code  # the header stops at the first method
    assert chunks["Box.area"].start_line == 17  # starts at the decorator line
    assert chunks["<module>"].code == "LIMIT = 5"
    assert chunks["Box.grow"].short_name == "grow"


def test_chunker_survives_bad_syntax():
    assert chunk_source("def broken(:\n", "x.py") == []


def test_tests_are_never_indexed():
    for repo in ("tinyutils", "shopkit", "taskboard"):
        paths = {c.path for c in chunk_repo(BENCHMARK_REPOS / repo)}
        assert paths and not any("test" in p for p in paths)


def test_is_source_file(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg").mkdir()
    assert is_source_file(tmp_path / "pkg" / "a.py", tmp_path)
    assert not is_source_file(tmp_path / "tests" / "a.py", tmp_path)
    assert not is_source_file(tmp_path / "pkg" / "test_a.py", tmp_path)
    assert not is_source_file(tmp_path / "pkg" / "conftest.py", tmp_path)


def test_tokenize_splits_and_keeps_names():
    toks = tokenize("low_stock lowStock is_expired")
    assert {"low", "stock", "lowstock", "expir"} <= set(toks)
    assert "the" not in tokenize("the cat")


def test_stem():
    assert stem("expired") == stem("expires") == "expir"
    assert stem("orders") == "order"
    assert stem("go") == "go"


def test_bm25_prefers_rare_matching_terms():
    docs = [["apple", "pie"], ["apple", "tart"], ["banana", "split"]]
    scores = BM25(docs).scores(["banana"])
    assert scores[2] > 0 and scores[0] == scores[1] == 0


def test_bm25_empty_inputs():
    assert BM25([]).scores(["x"]) == []
    assert BM25([["a"]]).scores([]) == [0.0]


def test_rrf_rewards_agreement():
    fused = rrf_fuse([[0, 1, 2], [2, 0, 1]])
    order = sorted(fused, key=fused.get, reverse=True)
    assert order[0] == 0  # first in one list, second in the other
    assert set(order) == {0, 1, 2}


def test_index_search_methods(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "def shipping_cost(weight):\n    return weight * 2\n\n\n"
        "def coupon_expiry(day):\n    return day > 3\n"
    )
    index = CodeIndex.build(tmp_path, FakeEmbedder())
    for method in ("bm25", "embedding", "hybrid"):
        top = index.search("shipping cost of heavy weight", k=1, method=method)[0]
        assert top.chunk.symbol == "shipping_cost" and top.rank == 1
    with pytest.raises(ValueError):
        index.search("x", method="magic")
    with pytest.raises(ValueError):
        CodeIndex.build(tmp_path).search("x", method="embedding")  # no embedder given


def test_decoy_verdicts():
    def r(real, **decoys):
        return QueryResult("b", "vague", "hybrid", 1, real, [], decoys)

    assert decoy_verdict(r(1, a=3)) == "ok"
    assert decoy_verdict(r(4, a=1)) == "outranked"
    assert decoy_verdict(r(9, a=2)) == "FOOLED"
    assert decoy_verdict(r(9, a=8)) == "missed"
    assert decoy_verdict(r(None, a=3)) == "FOOLED"


def test_recall_counts():
    rows = [
        QueryResult("a", "vague", "bm25", 1, 1),
        QueryResult("b", "vague", "bm25", 4, 7),
        QueryResult("c", "vague", "bm25", None, None),
    ]
    assert recall(rows, "vague", "bm25", "file", 5) == (2, 3)
    assert recall(rows, "vague", "bm25", "function", 5) == (1, 3)
    assert recall(rows, "vague", "bm25", "function", 10) == (2, 3)


def test_search_only_ever_sees_the_report_text():
    """No test names, errors or ground truth may reach the search."""
    bugs = load_bugs()[:3]
    fake = FakeEmbedder()
    results = evaluate(bugs, fake, progress=lambda _: None)
    allowed = {b.report_specific for b in bugs} | {b.report_vague for b in bugs}
    assert set(fake.queries) <= allowed
    assert len(results) == len(bugs) * 2 * 4  # 2 styles x 4 methods (incl. ablation)


def test_real_embedding_model_finds_an_obvious_match(tmp_path):
    pytest.importorskip("fastembed")
    from fixpilot.retrieval.embeddings import FastEmbedder

    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "def send_email(to):\n    return to\n\n\ndef compute_tax(amount):\n    return amount * 0.2\n"
    )
    emb = FastEmbedder()  # uses the shared .cache so the model is not downloaded again
    top = CodeIndex.build(tmp_path, emb).search("how much sales tax is charged", 1, "embedding")[0]
    assert top.chunk.symbol == "compute_tax"
