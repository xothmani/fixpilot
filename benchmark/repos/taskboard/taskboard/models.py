"""Core data classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .priorities import Priority


@dataclass
class Task:
    id: int
    title: str
    project: str = "inbox"
    status: str = "todo"
    priority: Priority = Priority.NORMAL
    assignee: str | None = None
    due: date | None = None
    estimate_min: int | None = None
    tags: list[str] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)
    created: date | None = None
    done_on: date | None = None


@dataclass
class Project:
    name: str
    owner: str
    deadline: date | None = None
