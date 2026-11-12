"""
Tests for in-memory API rate limiting.
"""

import pytest

import document_rag_service.api.rate_limiter as rate_limiter_module
from document_rag_service.api.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_rate_limiter_blocks_requests_after_limit_is_reached() -> None:
    rate_limiter = RateLimiter(window_seconds=60.0)

    assert await rate_limiter.check(key="client:1", limit=2) is None
    assert await rate_limiter.check(key="client:1", limit=2) is None

    exceeded = await rate_limiter.check(
        key="client:1",
        limit=2,
    )

    assert isinstance(exceeded, RateLimitExceeded)
    assert 1 <= exceeded.retry_after_seconds <= 60


@pytest.mark.anyio
async def test_rate_limiter_keeps_keys_independent() -> None:
    rate_limiter = RateLimiter(window_seconds=60.0)

    assert await rate_limiter.check(key="client:1", limit=1) is None

    exceeded = await rate_limiter.check(
        key="client:1",
        limit=1,
    )

    assert isinstance(exceeded, RateLimitExceeded)

    assert await rate_limiter.check(key="client:2", limit=1) is None


@pytest.mark.anyio
async def test_rate_limiter_accepts_request_after_window_expires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_time = [1000.0]

    monkeypatch.setattr(
        rate_limiter_module.time,
        "monotonic",
        lambda: current_time[0],
    )

    rate_limiter = RateLimiter(window_seconds=60.0)

    assert await rate_limiter.check(key="client:1", limit=1) is None

    current_time[0] = 1030.0

    exceeded = await rate_limiter.check(
        key="client:1",
        limit=1,
    )

    assert isinstance(exceeded, RateLimitExceeded)
    assert exceeded.retry_after_seconds == 30

    current_time[0] = 1060.1

    assert await rate_limiter.check(key="client:1", limit=1) is None


@pytest.mark.anyio
async def test_rate_limiter_validates_key_and_limit() -> None:
    rate_limiter = RateLimiter()

    with pytest.raises(
        ValueError,
        match="Rate limit key cannot be empty",
    ):
        await rate_limiter.check(
            key="   ",
            limit=1,
        )

    with pytest.raises(
        ValueError,
        match="Rate limit must be greater than zero",
    ):
        await rate_limiter.check(
            key="client:1",
            limit=0,
        )


def test_rate_limiter_rejects_invalid_window() -> None:
    with pytest.raises(
        ValueError,
        match="Rate limit window must be greater than zero",
    ):
        RateLimiter(window_seconds=0)
