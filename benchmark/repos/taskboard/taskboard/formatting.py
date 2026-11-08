"""Text formatting helpers."""

from __future__ import annotations

from .models import Task


def format_duration(minutes: int) -> str:
    """90 -> '1h 30m', 120 -> '2h', 45 -> '45m', 0 -> '0m'."""
    if minutes <= 0:
        return "0m"
    hours, mins = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if mins:
        parts.append(f"{mins}m")
    return " ".join(parts)


def pluralize(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def truncate(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def render_table(tasks: list[Task], title_width: int = 24) -> str:
    header = f"{'ID':>3}  {'TITLE':<{title_width}}  {'STATUS':<11}  PRI"
    lines = [header, "-" * len(header)]
    for t in tasks:
        lines.append(
            f"{t.id:>3}  {truncate(t.title, title_width):<{title_width}}  "
            f"{t.status:<11}  {int(t.priority)}"
        )
    return "\n".join(lines)
