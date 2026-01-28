from __future__ import annotations

from typing import Optional


class TidepoolError(Exception):
    """Base exception for Tidepool errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ValidationError(TidepoolError):
    """Invalid input data."""


class NotFoundError(TidepoolError):
    """Resource not found."""


class ServiceUnavailableError(TidepoolError):
    """Service is not available."""
