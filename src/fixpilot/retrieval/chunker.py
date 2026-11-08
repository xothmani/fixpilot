"""Split Python source files into function-sized chunks using the standard `ast` module.

Chunk kinds:
- "function": a top-level function, or a method (symbol is "Class.method")
- "class":    the class header (docstring and fields), up to its first method
- "module":   top-level constants of a file, joined into one chunk

Test files are never indexed: the agent must find bugs in source code, and tests mention
function names, which would make retrieval look better than it is.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {"tests", "test", ".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}


@dataclass(frozen=True)
class Chunk:
    path: str  # relative to the repo root, always with forward slashes
    start_line: int  # 1-based, inclusive
    end_line: int
    symbol: str  # "name", "Class.method", "Class", or "<module>"
    kind: str  # "function" | "class" | "module"
    code: str

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}:{self.symbol}"

    @property
    def short_name(self) -> str:
        """Last part of the symbol, e.g. 'method' for 'Class.method'."""
        return self.symbol.split(".")[-1]

    def index_text(self) -> str:
        """The text that gets indexed: location and name first, then the code."""
        return f"{self.path}\n{self.symbol}\n{self.code}"


def is_source_file(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in SKIP_DIRS for part in rel.parts[:-1]):
        return False
    name = path.name
    return name.endswith(".py") and not name.startswith("test_") and name != "conftest.py"


def source_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if is_source_file(p, root))


def _start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *[d.lineno for d in decorators]])


def chunk_source(source: str, rel_path: str) -> list[Chunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()

    def make(start: int, end: int, symbol: str, kind: str) -> Chunk:
        return Chunk(rel_path, start, end, symbol, kind, "\n".join(lines[start - 1 : end]))

    chunks: list[Chunk] = []
    constants: list[ast.stmt] = []
    funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
    for node in tree.body:
        if isinstance(node, funcs):
            chunks.append(make(_start(node), node.end_lineno, node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, funcs)]
            header_end = _start(methods[0]) - 1 if methods else node.end_lineno
            chunks.append(make(_start(node), header_end, node.name, "class"))
            for m in methods:
                chunks.append(make(_start(m), m.end_lineno, f"{node.name}.{m.name}", "function"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            constants.append(node)
    if constants:
        text = "\n".join("\n".join(lines[n.lineno - 1 : n.end_lineno]) for n in constants)
        chunks.append(
            Chunk(rel_path, constants[0].lineno, constants[-1].end_lineno, "<module>", "module", text)
        )
    return chunks


def chunk_repo(root: Path) -> list[Chunk]:
    root = Path(root).resolve()
    chunks: list[Chunk] = []
    for path in source_files(root):
        rel = path.relative_to(root).as_posix()
        chunks.extend(chunk_source(path.read_text(), rel))
    return chunks
