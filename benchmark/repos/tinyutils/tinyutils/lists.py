"""List helpers."""


def chunk(items: list, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def unique(items: list) -> list:
    """Remove duplicates but keep the first occurrence order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def flatten(nested: list[list]) -> list:
    return [x for sub in nested for x in sub]


def last(items: list, default=None):
    return items[-1] if items else default


def rotate(items: list, k: int) -> list:
    if not items:
        return []
    k %= len(items)
    return items[k:] + items[:k]
