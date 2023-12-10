class BaseException(Exception):
    """Base exception class for gaps library."""


class FitnessException(BaseException):
    """Raised fitness values are not set or do not match population size."""
