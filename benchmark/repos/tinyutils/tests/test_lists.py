import pytest

from tinyutils.lists import chunk, flatten, last, rotate, unique


def test_chunk():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk([], 3) == []


def test_chunk_bad_size():
    with pytest.raises(ValueError):
        chunk([1], 0)


def test_unique_keeps_order():
    assert unique([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_flatten():
    assert flatten([[1], [], [2, 3]]) == [1, 2, 3]


def test_last():
    assert last([1, 2, 3]) == 3
    assert last([], default="x") == "x"


def test_rotate():
    assert rotate([1, 2, 3, 4], 1) == [2, 3, 4, 1]
    assert rotate([1, 2, 3, 4], 5) == [2, 3, 4, 1]
    assert rotate([], 3) == []
