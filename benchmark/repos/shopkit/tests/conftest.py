from datetime import date

import pytest

from shopkit.catalog import Catalog
from shopkit.models import Product


@pytest.fixture
def products():
    return [
        Product("PEN-001", "Blue Pen", 150, "office", 20, ("writing", "blue")),
        Product("PAD-002", "Notepad", 450, "office", 300, ("paper",)),
        Product("MUG-003", "Coffee Mug", 1200, "kitchen", 400, ("coffee", "ceramic")),
        Product("LMP-004", "Desk Lamp", 3500, "office", 1800, ("light",)),
        Product("KTL-005", "Electric Kettle", 4999, "kitchen", 1500, ("coffee", "tea")),
    ]


@pytest.fixture
def catalog(products):
    return Catalog(products)


@pytest.fixture
def today():
    return date(2025, 3, 15)
