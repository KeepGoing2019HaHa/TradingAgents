from contextlib import suppress
from datetime import timedelta

MAX_ANALYSIS_FAILURES = 5


def provider_status(error) -> int | None:
    return getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)


def transient_provider_error(error) -> bool:
    code = provider_status(error)
    if code is not None:
        return code in {408, 409, 429} or code >= 500
    return any(name in type(error).__name__ for name in ("RateLimit", "Timeout", "Connection", "InternalServer", "ServiceUnavailable"))


def retry_delay(error, failures: int, now) -> timedelta:
    rate_limited = provider_status(error) == 429 or "RateLimit" in type(error).__name__
    seconds = min((900 if rate_limited else 60) * 2 ** (failures - 1), 7200)
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            seconds = max(seconds, float(retry_after))
        except (TypeError, ValueError):
            from email.utils import parsedate_to_datetime

            with suppress(TypeError, ValueError, OverflowError):
                seconds = max(seconds, (parsedate_to_datetime(retry_after) - now).total_seconds())
    return timedelta(seconds=seconds)
