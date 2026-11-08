"""Sales tax and VAT."""

from __future__ import annotations

from .errors import NotFound
from .money import apply_percent

TAX_RATES = {
    "US-CA": 7.25,
    "US-NY": 8.0,
    "US-TX": 6.25,
    "EU-DE": 19.0,
    "EU-FR": 20.0,
}


def tax_rate(region: str) -> float:
    try:
        return TAX_RATES[region]
    except KeyError:
        raise NotFound(f"No tax rate for region: {region}") from None


def tax_cents(taxable_cents: int, region: str) -> int:
    return apply_percent(taxable_cents, tax_rate(region))


def split_gross(gross_cents: int, region: str) -> tuple[int, int]:
    """For tax-inclusive prices: return (net, tax) where net + tax == gross."""
    rate = tax_rate(region)
    net = int(gross_cents / (1 + rate / 100) + 0.5)
    return net, gross_cents - net
