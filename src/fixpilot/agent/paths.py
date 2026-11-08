"""Safe path handling. Every path the agent supplies is untrusted.

Rules:
- relative paths only, no '..', no NUL bytes
- the fully resolved path (symlinks followed) must stay inside the repo copy
- files the agent WRITES must not be reached through any symlink
- the agent may only edit Python source files, never tests or config (see `is_editable`)
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from fixpilot.retrieval.chunker import is_source_file


class PathError(ValueError):
    """The path is not allowed. The message is safe to show to the agent."""


def resolve_in_repo(root: Path, rel: str, must_exist: bool = True) -> Path:
    """Return the real path of `rel` inside `root`, or raise PathError."""
    if not isinstance(rel, str) or not rel.strip():
        raise PathError("path must be a non-empty string")
    if "\x00" in rel:
        raise PathError("path contains a NUL byte")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or rel.startswith(("~", "\\")):
        raise PathError("path must be relative to the repository root")
    if ".." in pure.parts:
        raise PathError("path must not contain '..'")
    root_real = root.resolve()
    try:
        resolved = (root_real / rel).resolve()
    except (OSError, RuntimeError):  # symlink loops and similar
        raise PathError("path cannot be resolved") from None
    if not resolved.is_relative_to(root_real):
        raise PathError("path escapes the repository")
    if must_exist and not resolved.exists():
        raise PathError(f"no such file or directory: {rel}")
    return resolved


def relative_posix(root: Path, resolved: Path) -> str:
    return resolved.relative_to(root.resolve()).as_posix()


def has_symlink_component(root: Path, rel: str) -> bool:
    """True if the file or any folder on the way to it is a symlink."""
    current = root.resolve()
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def check_editable(root: Path, rel: str) -> Path:
    """Resolve a path the agent wants to EDIT. Only existing Python source files qualify."""
    resolved = resolve_in_repo(root, rel)
    if has_symlink_component(root, rel):
        raise PathError("editing through a symlink is not allowed")
    if not resolved.is_file():
        raise PathError("only existing files can be edited")
    if not is_source_file(resolved, root.resolve()):
        raise PathError("only Python source files can be edited, not tests or configuration")
    return resolved
