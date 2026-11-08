"""Role based permissions."""

from __future__ import annotations

from .errors import PermissionDenied

ROLE_RANK = {"viewer": 0, "member": 1, "lead": 2, "admin": 3}
ACTION_MIN_ROLE = {
    "view": "viewer",
    "comment": "viewer",
    "create": "member",
    "edit": "member",
    "assign": "lead",
    "delete": "lead",
    "manage_users": "admin",
}


def can(role: str, action: str) -> bool:
    if role not in ROLE_RANK or action not in ACTION_MIN_ROLE:
        return False
    return ROLE_RANK[role] >= ROLE_RANK[ACTION_MIN_ROLE[action]]


def require(role: str, action: str) -> None:
    if not can(role, action):
        raise PermissionDenied(f"{role} cannot {action}")


def allowed_actions(role: str) -> list[str]:
    return sorted(a for a in ACTION_MIN_ROLE if can(role, a))
