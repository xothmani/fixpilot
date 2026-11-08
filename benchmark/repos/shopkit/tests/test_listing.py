import pytest

from shopkit.errors import ValidationError
from shopkit.pagination import page_info, paginate, total_pages
from shopkit.search import search, tokenize
from shopkit.sorting import sort_products


def test_paginate_basic():
    items = list(range(1, 11))
    assert paginate(items, 1, 4) == [1, 2, 3, 4]
    assert paginate(items, 3, 4) == [9, 10]
    assert paginate(items, 4, 4) == []


def test_paginate_validates():
    with pytest.raises(ValidationError):
        paginate([1], 0, 5)
    with pytest.raises(ValidationError):
        paginate([1], 1, 0)


def test_total_pages():
    assert total_pages(10, 5) == 2
    assert total_pages(11, 5) == 3
    assert total_pages(0, 5) == 1


def test_page_info():
    assert page_info(25, 1, 10) == {"page": 1, "pages": 3, "has_next": True, "has_prev": False}
    assert page_info(25, 3, 10)["has_next"] is False


def test_sort_by_price_and_name(products):
    assert [p.sku for p in sort_products(products, "price")][:2] == ["PEN-001", "PAD-002"]
    assert sort_products(products, "name")[0].name == "Blue Pen"
    assert sort_products(products, "price", descending=True)[0].sku == "KTL-005"


def test_sort_bad_key(products):
    with pytest.raises(ValidationError):
        sort_products(products, "color")


def test_tokenize():
    assert tokenize("Hello, World-2!") == ["hello", "world", "2"]


def test_search_ranks_name_over_tag(catalog):
    results = search(catalog, "coffee")
    assert [p.sku for p in results] == ["MUG-003", "KTL-005"]


def test_search_name_beats_category(catalog):
    assert search(catalog, "office pen")[0].sku == "PEN-001"


def test_search_empty_query(catalog):
    assert search(catalog, "   ") == []
    assert search(catalog, "zzzz") == []


def test_search_limit(catalog):
    assert len(search(catalog, "office", limit=2)) == 2
