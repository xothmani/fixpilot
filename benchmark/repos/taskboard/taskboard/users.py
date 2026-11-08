"""People who can own tasks."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import NotFound, ValidationError
from .models import Task


@dataclass(frozen=True)
class User:
    name: str
    role: str = "member"
    capacity_min_per_day: int = 360


class Roster:
    def __init__(self) -> None:
        self._users: dict[str, User] = {}

    def add(self, user: User) -> None:
        if user.name in self._users:
            raise ValidationError(f"Duplicate user: {user.name}")
        self._users[user.name] = user

    def get(self, name: str) -> User:
        try:
            return self._users[name]
        except KeyError:
            raise NotFound(f"No user {name}") from None

    def names(self) -> list[str]:
        return sorted(self._users)

    def find_by_prefix(self, prefix: str) -> list[User]:
        p = prefix.lower()
        return sorted((u for u in self._users.values() if u.name.lower().startswith(p)),
                      key=lambda u: u.name)

    def least_loaded(self, tasks: list[Task]) -> User:
        """User with the fewest open tasks. Ties go to the alphabetically first name."""
        if not self._users:
            raise NotFound("Roster is empty")
        load = {name: 0 for name in self._users}
        for t in tasks:
            if t.assignee in load and t.status != "done":
                load[t.assignee] += 1
        best = min(load, key=lambda n: (load[n], n))
        return self._users[best]
