"""Simple keyword search over a catalog."""

from __future__ import annotations

import re

from .catalog import Catalog
from .models import Product

NAME_WEIGHT = 3
TAG_WEIGHT = 2
CATEGORY_WEIGHT = 1


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score(product: Product, query_tokens: list[str]) -> int:
    name_tokens = set(tokenize(product.name))
    tag_tokens = {t.lower() for t in product.tags}
    category_tokens = set(tokenize(product.category))
    total = 0
    for token in query_tokens:
        if token in name_tokens:
            total += NAME_WEIGHT
        if token in tag_tokens:
            total += TAG_WEIGHT
        if token in category_tokens:
            total += CATEGORY_WEIGHT
    return total


def search(catalog: Catalog, query: str, limit: int = 10) -> list[Product]:
    tokens = tokenize(query)
    if not tokens:
        return []
    scored = [(score(p, tokens), p) for p in catalog.all()]
    scored = [(s, p) for s, p in scored if s > 0]
    scored.sort(key=lambda sp: (-sp[0], sp[1].name.lower()))
    return [p for _, p in scored[:limit]]
