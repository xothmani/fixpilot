import pytest

from tinyutils.mathx import average, clamp, factorial, is_prime, percent


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-5, 0, 10) == 0
    assert clamp(15, 0, 10) == 10


def test_clamp_bad_bounds():
    with pytest.raises(ValueError):
        clamp(1, 5, 2)


def test_average():
    assert average([1, 2, 3, 4]) == 2.5


def test_average_empty():
    with pytest.raises(ValueError):
        average([])


def test_is_prime():
    assert [n for n in range(12) if is_prime(n)] == [2, 3, 5, 7, 11]
    assert is_prime(49) is False
    assert is_prime(97)


def test_factorial():
    assert factorial(0) == 1
    assert factorial(5) == 120


def test_factorial_negative():
    with pytest.raises(ValueError):
        factorial(-1)


def test_percent():
    assert percent(1, 4) == 25
    assert percent(5, 0) == 0.0
