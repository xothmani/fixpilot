"""Dependency ordering between tasks."""

from __future__ import annotations

from .errors import CycleError
from .models import Task


def blocked_tasks(tasks: list[Task]) -> list[Task]:
    """Open tasks that depend on at least one task that is not done yet."""
    by_id = {t.id: t for t in tasks}
    result = []
    for t in tasks:
        if t.status == "done":
            continue
        if any(d in by_id and by_id[d].status != "done" for d in t.depends_on):
            result.append(t)
    return result


def ready_tasks(tasks: list[Task]) -> list[Task]:
    blocked = {t.id for t in blocked_tasks(tasks)}
    return [t for t in tasks if t.status != "done" and t.id not in blocked]


def topological_order(tasks: list[Task]) -> list[int]:
    """Task ids so every task comes after the tasks it depends on. Raises CycleError."""
    ids = {t.id for t in tasks}
    deps = {t.id: [d for d in t.depends_on if d in ids] for t in tasks}
    order: list[int] = []
    state: dict[int, int] = {}  # 1 = visiting, 2 = done

    def visit(node: int) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            raise CycleError(f"Cycle through task {node}")
        state[node] = 1
        for d in sorted(deps[node]):
            visit(d)
        state[node] = 2
        order.append(node)

    for task_id in sorted(ids):
        visit(task_id)
    return order


def has_cycle(tasks: list[Task]) -> bool:
    try:
        topological_order(tasks)
    except CycleError:
        return True
    return False
