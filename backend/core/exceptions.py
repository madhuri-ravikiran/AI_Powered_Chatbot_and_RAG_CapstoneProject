"""Translate application/provider failures to safe, consistent API errors."""

import logging
from collections.abc import Callable
from typing import TypeVar

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class AppError(Exception):
    """An expected failure with a stable code and safe user-facing message."""

    def __init__(self, status: int, code: str, message: str):
        """Store the HTTP status and sanitized description."""
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def provider_call(operation: str, function: Callable[[], T]) -> T:
    """Invoke a provider and hide raw errors, which may contain private text."""
    try:
        return function()
    except AppError:
        raise
    except AuthenticationError:
        error = AppError(503, "provider_auth", "Provider credentials were rejected. Check .env.")
    except RateLimitError:
        error = AppError(
            429, "provider_rate_limit", "Provider rate limit reached. Try again later."
        )
    except (APITimeoutError, httpx.TimeoutException):
        error = AppError(504, "provider_timeout", "The AI provider timed out. Please retry.")
    except (APIConnectionError, httpx.NetworkError):
        error = AppError(503, "provider_unavailable", "Cannot connect to the AI provider.")
    except Exception:
        error = AppError(502, "provider_error", f"The {operation} provider request failed.")
    logger.warning("Provider operation failed: operation=%s code=%s", operation, error.code)
    raise error from None
