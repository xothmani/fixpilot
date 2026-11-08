"""Shipping addresses."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ValidationError
from .validators import is_valid_postal_code


@dataclass(frozen=True)
class Address:
    name: str
    line1: str
    city: str
    postal_code: str
    country: str
    line2: str = ""

    def validate(self) -> None:
        for label, value in (("name", self.name), ("line1", self.line1), ("city", self.city)):
            if not value.strip():
                raise ValidationError(f"{label} is required")
        if not is_valid_postal_code(self.postal_code, self.country):
            raise ValidationError(f"Bad postal code for {self.country}: {self.postal_code}")

    def one_line(self) -> str:
        parts = [self.name, self.line1, self.line2, f"{self.postal_code} {self.city}", self.country]
        return ", ".join(p for p in parts if p)

    def region(self, state: str | None = None) -> str:
        """Tax region key such as US-CA or EU-DE."""
        if self.country == "US" and state:
            return f"US-{state.upper()}"
        if self.country in ("DE", "FR"):
            return f"EU-{self.country}"
        return self.country
