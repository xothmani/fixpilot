"""Customer wishlists."""

from __future__ import annotations

from .catalog import Catalog
from .errors import NotFound


class Wishlist:
    def __init__(self, owner: str) -> None:
        self.owner = owner
        self._skus: list[str] = []

    def add(self, sku: str) -> bool:
        """Return True if added, False if it was already there."""
        if sku in self._skus:
            return False
        self._skus.append(sku)
        return True

    def remove(self, sku: str) -> None:
        if sku not in self._skus:
            raise NotFound(f"Not on wishlist: {sku}")
        self._skus.remove(sku)

    def skus(self) -> list[str]:
        return list(self._skus)

    def total_cents(self, catalog: Catalog) -> int:
        return sum(catalog.get(s).price_cents for s in self._skus)

    def affordable(self, catalog: Catalog, budget_cents: int) -> list[str]:
        """Items that fit in the budget one by one, in wishlist order."""
        return [s for s in self._skus if catalog.get(s).price_cents <= budget_cents]
