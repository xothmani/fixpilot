"""Number helpers."""


def clamp(value: float, low: float, high: float) -> float:
    if low > high:
        raise ValueError("low must not be greater than high")
    return max(low, min(value, high))


def average(numbers: list[float]) -> float:
    if not numbers:
        raise ValueError("average of empty list")
    return sum(numbers) / len(numbers)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("negative factorial")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def percent(part: float, whole: float) -> float:
    if whole == 0:
        return 0.0
    return part / whole * 100
