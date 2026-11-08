"""Exceptions used across shopkit."""


class ShopError(Exception):
    """Base class for all shopkit errors."""


class ValidationError(ShopError):
    """Input data is not acceptable."""


class NotFound(ShopError):
    """A product, coupon, order or region does not exist."""


class OutOfStock(ShopError):
    """Not enough stock to satisfy a reservation."""


class InvalidCoupon(ShopError):
    """A coupon cannot be used (unknown, expired, exhausted, or minimum not met)."""


class InvalidTransition(ShopError):
    """An order status change is not allowed."""
