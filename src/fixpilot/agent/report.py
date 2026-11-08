"""The final report. The agent writes the explanation; the app adds the facts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    note: str = Field(max_length=500)


class ReportInput(BaseModel):
    """What the agent supplies to write_report."""

    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=1500)
    root_cause: str = Field(min_length=1, max_length=1500)
    evidence: list[Evidence] = Field(default_factory=list, max_length=8)
    uncertainties: list[str] = Field(default_factory=list, max_length=8)
    confidence: Literal["low", "medium", "high"] = "low"


class Report(ReportInput):
    """ReportInput plus facts that come from the application state, never from the LLM."""

    status: str
    stop_reason: str | None
    diff: str
    attempts: int
    tests_verified: bool
    last_test_summary: str
