"""Page through a list."""

from __future__ import annotations

import math
from typing import TypeVar

from .errors import ValidationError

T = TypeVar("T")


def total_pages(total_items: int, per_page: int) -> int:
    if per_page <= 0:
        raise ValidationError("per_page must be positive")
    return max(1, math.ceil(total_items / per_page))


def paginate(items: list[T], page: int, per_page: int) -> list[T]:
    """Pages start at 1. A page past the end returns an empty list."""
    if page < 1:
        raise ValidationError("page must be 1 or greater")
    if per_page <= 0:
        raise ValidationError("per_page must be positive")
    start = (page - 1) * per_page
    return items[start : start + per_page]


def page_info(total_items: int, page: int, per_page: int) -> dict:
    pages = total_pages(total_items, per_page)
    return {
        "page": page,
        "pages": pages,
        "has_next": page < pages,
        "has_prev": page > 1,
    }
