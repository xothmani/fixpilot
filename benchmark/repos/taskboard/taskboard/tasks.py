"""In-memory task storage."""

from __future__ import annotations

from datetime import date

from .errors import NotFound, ValidationError
from .models import Task
from .status import require_move
from .validators import validate_title


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[int, Task] = {}
        self._next_id = 1

    def add(self, title: str, **fields) -> Task:
        task = Task(self._next_id, validate_title(title), **fields)
        self._tasks[task.id] = task
        self._next_id += 1
        return task

    def get(self, task_id: int) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise NotFound(f"No task {task_id}") from None

    def delete(self, task_id: int) -> None:
        self.get(task_id)
        del self._tasks[task_id]
        for t in self._tasks.values():
            if task_id in t.depends_on:
                t.depends_on.remove(task_id)

    def all(self) -> list[Task]:
        return list(self._tasks.values())

    def __len__(self) -> int:
        return len(self._tasks)

    def move(self, task_id: int, new_status: str, today: date | None = None) -> Task:
        task = self.get(task_id)
        require_move(task.status, new_status)
        task.status = new_status
        if new_status == "done":
            task.done_on = today
        elif task.done_on is not None:
            task.done_on = None
        return task

    def assign(self, task_id: int, user: str | None) -> Task:
        task = self.get(task_id)
        task.assignee = user
        return task

    def by_status(self, status: str) -> list[Task]:
        return [t for t in self._tasks.values() if t.status == status]

    def by_assignee(self, user: str) -> list[Task]:
        return [t for t in self._tasks.values() if t.assignee == user]

    def by_project(self, project: str) -> list[Task]:
        return [t for t in self._tasks.values() if t.project == project]

    def add_dependency(self, task_id: int, depends_on: int) -> None:
        if task_id == depends_on:
            raise ValidationError("A task cannot depend on itself")
        self.get(depends_on)
        task = self.get(task_id)
        if depends_on not in task.depends_on:
            task.depends_on.append(depends_on)
