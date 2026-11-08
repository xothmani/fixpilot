"""Structured logs: one JSON line per tool call. Secrets are removed before anything is stored."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

_SECRET_ENV_NAMES = ("FIXPILOT_LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN",
                     "LANGFUSE_SECRET_KEY")
_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\b(\s*[:=]\s*)\S+"),
]
MAX_FIELD_CHARS = 4000


def redact(text: str) -> str:
    """Replace API keys and secret-looking assignments with [REDACTED]."""
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            text = text.replace(value, "[REDACTED]")
    text = _PATTERNS[0].sub("[REDACTED]", text)
    return _PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)


def clean(value: Any, limit: int = MAX_FIELD_CHARS) -> Any:
    """Redact every string inside a JSON-like value and cut very long ones."""
    if isinstance(value, str):
        text = redact(value)
        return text if len(text) <= limit else text[:limit] + f"...[{len(text) - limit} more chars]"
    if isinstance(value, dict):
        return {str(k): clean(v, limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v, limit) for v in value]
    return value


class Tracer:
    """Keeps records in memory and, if `run_dir` is given, also appends them to trace.jsonl."""

    def __init__(self, run_dir: Path | None = None, static: dict | None = None):
        self.run_dir = Path(run_dir) if run_dir else None
        self.static = clean(static or {})  # e.g. provider and model, added to every record
        self.records: list[dict] = []
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, **fields: Any) -> dict:
        record = {"ts": round(time.time(), 3), **self.static, **clean(fields)}
        self.records.append(record)
        if self.run_dir:
            with (self.run_dir / "trace.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
        return record

    def write_json(self, name: str, data: Any) -> None:
        if self.run_dir:
            (self.run_dir / name).write_text(json.dumps(clean(data, 50_000), indent=2) + "\n")
