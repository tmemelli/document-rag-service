"""
Integration tests for API rate limiting.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import document_rag_service.api.dependencies as dependencies_module
from document_rag_service.api.dependencies import (
    get_auth_token_service,
    get_user_service,
)
from document_rag_service.api.rate_limiter import RateLimiter
from document_rag_service.api.routes import api_router
from document_rag_service.models.user import User
from document_rag_service.services.auth_token_service import AuthTokenService
from document_rag_service.services.user_service import UserService


def _create_test_app() -> FastAPI:
    app = FastAPI()
    app.include_router(api_router)

    app.state.rate_limiter = RateLimiter(
        window_seconds=60.0,
    )

    return app


def test_login_rate_limit_returns_429_and_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _create_test_app()

    monkeypatch.setattr(
        dependencies_module,
        "get_settings",
        lambda: SimpleNamespace(
            auth_login_rate_limit_per_minute=2,
        ),
    )

    user = User(
        id=1,
        email="user@example.com",
        password_hash="hashed-password",
        is_active=True,
    )

    user_service = MagicMock(
        spec=UserService,
    )
    user_service.authenticate_user = AsyncMock(
        return_value=user,
    )

    token_service = MagicMock(
        spec=AuthTokenService,
    )
    token_service.create_access_token.return_value = "test-access-token"

    app.dependency_overrides[get_user_service] = lambda: user_service
    app.dependency_overrides[get_auth_token_service] = lambda: token_service

    client = TestClient(app)

    payload = {
        "email": "user@example.com",
        "password": "password123",
    }

    headers = {
        "X-Forwarded-For": "203.0.113.10",
    }

    first_response = client.post(
        "/v1/auth/login",
        json=payload,
        headers=headers,
    )

    second_response = client.post(
        "/v1/auth/login",
        json=payload,
        headers=headers,
    )

    third_response = client.post(
        "/v1/auth/login",
        json=payload,
        headers=headers,
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert third_response.status_code == 429
    assert third_response.json() == {
        "detail": "Rate limit exceeded. Please try again later.",
    }

    retry_after = int(
        third_response.headers["Retry-After"],
    )

    assert 1 <= retry_after <= 60


def test_register_rate_limit_returns_429_and_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _create_test_app()

    monkeypatch.setattr(
        dependencies_module,
        "get_settings",
        lambda: SimpleNamespace(
            auth_register_rate_limit_per_minute=2,
        ),
    )

    user = User(
        id=1,
        email="new-user@example.com",
        password_hash="hashed-password",
        is_active=True,
    )

    user_service = MagicMock(
        spec=UserService,
    )
    user_service.register_user = AsyncMock(
        return_value=user,
    )

    token_service = MagicMock(
        spec=AuthTokenService,
    )
    token_service.create_access_token.return_value = "test-access-token"

    app.dependency_overrides[get_user_service] = lambda: user_service
    app.dependency_overrides[get_auth_token_service] = lambda: token_service

    client = TestClient(app)

    payload = {
        "email": "new-user@example.com",
        "password": "password123",
    }

    headers = {
        "X-Forwarded-For": "203.0.113.10",
    }

    first_response = client.post(
        "/v1/auth/register",
        json=payload,
        headers=headers,
    )

    second_response = client.post(
        "/v1/auth/register",
        json=payload,
        headers=headers,
    )

    third_response = client.post(
        "/v1/auth/register",
        json=payload,
        headers=headers,
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 201

    assert third_response.status_code == 429
    assert third_response.json() == {
        "detail": "Rate limit exceeded. Please try again later.",
    }

    retry_after = int(
        third_response.headers["Retry-After"],
    )

    assert 1 <= retry_after <= 60
