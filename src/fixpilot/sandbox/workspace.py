"""A throwaway copy of a repo. The agent only ever edits this copy."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

IGNORED = shutil.ignore_patterns(
    ".git", "__pycache__", ".pytest_cache", "*.pyc", ".venv", "venv", ".ruff_cache"
)


class Workspace:
    """Copies `source` into a fresh temp directory. Delete it with `cleanup()`."""

    def __init__(self, source: Path):
        source = Path(source).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Not a directory: {source}")
        self.source = source
        self._root = Path(tempfile.mkdtemp(prefix="fixpilot-")).resolve()
        self.path = self._root / "repo"
        shutil.copytree(source, self.path, ignore=IGNORED, symlinks=True)
        _make_world_readable(self._root)

    def cleanup(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()


def _make_world_readable(root: Path) -> None:
    """The container runs as another user, so it needs read access to the copy."""
    for dirpath, dirnames, filenames in os.walk(root):
        os.chmod(dirpath, 0o755)
        for name in filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, 0o644)
