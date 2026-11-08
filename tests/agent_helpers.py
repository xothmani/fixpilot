"""Builds small fake repos and tool contexts for the agent tests."""

from pathlib import Path

from fixpilot.agent import AgentState, Limits, ToolContext, ToolRegistry, Tracer

CALC = '''"""A tiny module for tests."""


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def average(values):
    return sum(values) / len(values)
'''

DUP = '''def first():
    return 1


def second():
    return 1
'''

FILES = {
    "pkg/__init__.py": "",
    "pkg/calc.py": CALC,
    "pkg/dup.py": DUP,
    "pkg/long.py": "\n".join(f"x{i} = {i}" for i in range(1, 501)) + "\n",
    "tests/test_calc.py": "from pkg.calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    "pytest.ini": "[pytest]\ntestpaths = tests\n",
    "notes.txt": "plain text\n",
}


def make_repo(root: Path, files: dict[str, str] | None = None) -> Path:
    repo = root / "repo"
    for rel, text in (files or FILES).items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return repo


def make_ctx(root: Path, limits: Limits | None = None, files=None, **kw):
    repo = make_repo(root, files)
    state = AgentState.new("some bug report", limits or Limits())
    ctx = ToolContext(root=repo, state=state, tracer=Tracer(), **kw)
    return ctx, ToolRegistry(ctx)
