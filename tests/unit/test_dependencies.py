"""
Tests for API dependency helpers.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

import document_rag_service.api.dependencies as dependencies_module
from document_rag_service.api.dependencies import (
    _get_client_ip,
    enforce_document_upload_rate_limit,
)
from document_rag_service.api.rate_limiter import RateLimiter
from document_rag_service.models.user import User


def _build_request(
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] | None = ("127.0.0.1", 12345),
) -> Request:
    scope = {
        "type": "http",
        "headers": headers or [],
        "client": client,
    }

    return Request(scope)


def test_get_client_ip_ignores_untrusted_forwarded_for_header() -> None:
    request = _build_request(
        headers=[
            (
                b"x-forwarded-for",
                b"203.0.113.10, 10.0.0.1",
            )
        ],
        client=("198.51.100.20", 12345),
    )

    assert _get_client_ip(request) == "198.51.100.20"


def test_get_client_ip_uses_direct_client_when_header_is_missing() -> None:
    request = _build_request(
        client=("198.51.100.20", 12345),
    )

    assert _get_client_ip(request) == "198.51.100.20"


def test_get_client_ip_returns_none_when_client_cannot_be_identified() -> None:
    request = _build_request(
        client=None,
    )

    assert _get_client_ip(request) is None


@pytest.mark.asyncio
async def test_document_upload_rate_limit_is_keyed_by_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the authenticated user id for document-upload rate-limit buckets."""
    monkeypatch.setattr(
        dependencies_module,
        "get_settings",
        lambda: SimpleNamespace(
            document_upload_rate_limit_per_minute=3,
        ),
    )

    rate_limiter = MagicMock(spec=RateLimiter)
    rate_limiter.check = AsyncMock(return_value=None)
    user = User(
        id=42,
        email="user@example.com",
        password_hash="hash",
        is_active=True,
    )

    await enforce_document_upload_rate_limit(
        current_user=user,
        rate_limiter=rate_limiter,
    )

    rate_limiter.check.assert_awaited_once_with(
        key="documents:upload:user:42",
        limit=3,
    )
