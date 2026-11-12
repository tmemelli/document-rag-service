"""In-memory sliding-window rate limiting for public API operations."""

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RateLimitExceeded:
    """
    Describe a rejected rate-limited request.

    Attributes:
        retry_after_seconds: Whole number of seconds the client should wait
            before another request may be accepted by the current bucket.
    """

    retry_after_seconds: int


class RateLimiter:
    """
    Enforce per-key request limits using an in-memory sliding time window.

    Each rate-limit key maintains timestamps for requests that remain inside
    the active window. Access to the shared bucket collection is serialized
    with an asynchronous lock so concurrent requests cannot bypass configured
    limits through race conditions.

    The limiter is process-local and is intended to be shared by all requests
    handled by one application process. Deployments using multiple workers or
    horizontally scaled instances would require a shared backing store to
    enforce limits globally.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 60.0,
    ) -> None:
        """
        Initialize the sliding-window rate limiter.

        Args:
            window_seconds: Duration of each rate-limit window in seconds.

        Raises:
            ValueError: If the configured window is not greater than zero.
        """

        if window_seconds <= 0:
            raise ValueError("Rate limit window must be greater than zero")

        self._window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()

    async def check(
        self,
        *,
        key: str,
        limit: int,
    ) -> RateLimitExceeded | None:
        """
        Record a request when its rate-limit bucket has available capacity.

        Expired timestamps are discarded before evaluating the current request.
        When the bucket has reached its limit, the request is rejected without
        recording a new timestamp and the remaining wait time is returned.

        Args:
            key: Stable identifier for the protected client, user, or
                operation.
            limit: Maximum number of accepted requests within the active
                sliding window.

        Returns:
            ``None`` when the request is accepted, otherwise a
            ``RateLimitExceeded`` value describing when the client may retry.

        Raises:
            ValueError: If the key is empty or the request limit is not
                greater than zero.
        """

        normalized_key = key.strip()

        if not normalized_key:
            raise ValueError("Rate limit key cannot be empty")

        if limit <= 0:
            raise ValueError("Rate limit must be greater than zero")

        now = time.monotonic()
        cutoff = now - self._window_seconds

        async with self._lock:
            self._cleanup_if_due(
                now=now,
                cutoff=cutoff,
            )

            timestamps = self._requests.setdefault(
                normalized_key,
                deque(),
            )

            self._discard_expired(
                timestamps,
                cutoff=cutoff,
            )

            if len(timestamps) >= limit:
                oldest_timestamp = timestamps[0]

                retry_after_seconds = max(
                    1,
                    math.ceil(self._window_seconds - (now - oldest_timestamp)),
                )

                return RateLimitExceeded(
                    retry_after_seconds=retry_after_seconds,
                )

            timestamps.append(now)

        return None

    def _cleanup_if_due(
        self,
        *,
        now: float,
        cutoff: float,
    ) -> None:
        """
        Periodically remove expired rate-limit buckets.

        Global cleanup runs at most once per configured window. Each bucket is
        first stripped of expired timestamps and then removed entirely when no
        active requests remain.

        Args:
            now: Current monotonic timestamp.
            cutoff: Earliest timestamp still belonging to the active window.
        """

        if now - self._last_cleanup < self._window_seconds:
            return

        empty_keys: list[str] = []

        for key, timestamps in self._requests.items():
            self._discard_expired(
                timestamps,
                cutoff=cutoff,
            )

            if not timestamps:
                empty_keys.append(key)

        for key in empty_keys:
            del self._requests[key]

        self._last_cleanup = now

    @staticmethod
    def _discard_expired(
        timestamps: deque[float],
        *,
        cutoff: float,
    ) -> None:
        """
        Remove request timestamps outside the active sliding window.

        Because timestamps are appended chronologically, expired entries can be
        removed efficiently from the left side of the deque until the first
        active request is reached.

        Args:
            timestamps: Chronologically ordered request timestamps for one
                rate-limit bucket.
            cutoff: Earliest timestamp that remains inside the active window.
        """

        while timestamps and timestamps[0] <= cutoff:
            timestamps.popleft()
