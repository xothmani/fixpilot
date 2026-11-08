class BoardError(Exception):
    """Base class for taskboard errors."""


class ValidationError(BoardError):
    pass


class NotFound(BoardError):
    pass


class PermissionDenied(BoardError):
    pass


class CycleError(BoardError):
    """Task dependencies form a loop."""
