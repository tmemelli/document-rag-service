"""
HTTP integration tests for authentication routes.
"""

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.api.dependencies import (
    enforce_login_rate_limit,
    enforce_register_rate_limit,
    get_auth_token_service,
    get_database_session,
)
from document_rag_service.api.routes import api_router
from document_rag_service.services.auth_token_service import AuthTokenService

TEST_JWT_SECRET = "integration-test-secret-that-is-long-enough-for-jwt-signing"


def _build_test_app(
    db_session: AsyncSession,
) -> FastAPI:
    """Build an isolated FastAPI app backed by the integration database."""
    app = FastAPI()
    app.include_router(api_router)

    async def override_database_session() -> AsyncIterator[AsyncSession]:
        """Provide the transaction-scoped integration session."""
        yield db_session

    def override_auth_token_service() -> AuthTokenService:
        """Provide deterministic JWT configuration for HTTP tests."""
        return AuthTokenService(
            secret=TEST_JWT_SECRET,
            algorithm="HS256",
            issuer="document-rag-service-test",
            audience="document-rag-service-test",
            expire_minutes=30,
        )

    async def bypass_rate_limit() -> None:
        """Disable rate limiting outside dedicated rate-limit tests."""
        return None

    app.dependency_overrides[get_database_session] = override_database_session
    app.dependency_overrides[get_auth_token_service] = override_auth_token_service
    app.dependency_overrides[enforce_register_rate_limit] = bypass_rate_limit
    app.dependency_overrides[enforce_login_rate_limit] = bypass_rate_limit

    return app


async def _build_client(
    db_session: AsyncSession,
) -> AsyncClient:
    """Build an HTTPX client connected directly to the FastAPI app."""
    app = _build_test_app(db_session)

    transport = ASGITransport(app=app)

    return AsyncClient(
        transport=transport,
        base_url="http://testserver",
    )


@pytest.mark.asyncio
async def test_register_user_returns_access_token_and_user(
    db_session: AsyncSession,
) -> None:
    """Register a user through the real HTTP route."""
    async with await _build_client(db_session) as client:
        response = await client.post(
            "/v1/auth/register",
            json={
                "email": "api-register@example.com",
                "password": "StrongPassword123!",
            },
        )

    assert response.status_code == 201

    payload = response.json()

    assert payload["access_token"]
    assert payload["token_type"] == "bearer"

    assert payload["user"]["email"] == "api-register@example.com"
    assert payload["user"]["is_active"] is True
    assert isinstance(payload["user"]["id"], int)


@pytest.mark.asyncio
async def test_registered_access_token_authenticates_me_route(
    db_session: AsyncSession,
) -> None:
    """Use a token issued by registration on an authenticated endpoint."""
    async with await _build_client(db_session) as client:
        register_response = await client.post(
            "/v1/auth/register",
            json={
                "email": "api-me@example.com",
                "password": "StrongPassword123!",
            },
        )

        assert register_response.status_code == 201

        access_token = register_response.json()["access_token"]

        me_response = await client.get(
            "/v1/auth/me",
            headers={
                "Authorization": f"Bearer {access_token}",
            },
        )

    assert me_response.status_code == 200

    payload = me_response.json()

    assert payload["email"] == "api-me@example.com"
    assert payload["is_active"] is True
    assert isinstance(payload["id"], int)


@pytest.mark.asyncio
async def test_login_returns_access_token_for_registered_user(
    db_session: AsyncSession,
) -> None:
    """Authenticate through the real login route."""
    async with await _build_client(db_session) as client:
        register_response = await client.post(
            "/v1/auth/register",
            json={
                "email": "api-login@example.com",
                "password": "StrongPassword123!",
            },
        )

        assert register_response.status_code == 201

        login_response = await client.post(
            "/v1/auth/login",
            json={
                "email": "api-login@example.com",
                "password": "StrongPassword123!",
            },
        )

    assert login_response.status_code == 200

    payload = login_response.json()

    assert payload["access_token"]
    assert payload["token_type"] == "bearer"
    assert payload["user"]["email"] == "api-login@example.com"


@pytest.mark.asyncio
async def test_login_rejects_invalid_password(
    db_session: AsyncSession,
) -> None:
    """Return HTTP 401 when credentials are invalid."""
    async with await _build_client(db_session) as client:
        register_response = await client.post(
            "/v1/auth/register",
            json={
                "email": "api-invalid-password@example.com",
                "password": "StrongPassword123!",
            },
        )

        assert register_response.status_code == 201

        login_response = await client.post(
            "/v1/auth/login",
            json={
                "email": "api-invalid-password@example.com",
                "password": "WrongPassword123!",
            },
        )

    assert login_response.status_code == 401

    assert login_response.json() == {
        "detail": "Invalid email or password",
    }

    assert login_response.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_register_rejects_duplicate_email(
    db_session: AsyncSession,
) -> None:
    """Return HTTP 409 when registering an existing email."""
    async with await _build_client(db_session) as client:
        first_response = await client.post(
            "/v1/auth/register",
            json={
                "email": "api-duplicate@example.com",
                "password": "StrongPassword123!",
            },
        )

        assert first_response.status_code == 201

        duplicate_response = await client.post(
            "/v1/auth/register",
            json={
                "email": "api-duplicate@example.com",
                "password": "AnotherPassword123!",
            },
        )

    assert duplicate_response.status_code == 409

    assert duplicate_response.json() == {
        "detail": "Email is already registered",
    }


@pytest.mark.asyncio
async def test_me_rejects_missing_bearer_token(
    db_session: AsyncSession,
) -> None:
    """Protect authenticated routes from anonymous requests."""
    async with await _build_client(db_session) as client:
        response = await client.get(
            "/v1/auth/me",
        )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid authentication credentials",
    }

    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_me_rejects_invalid_bearer_token(
    db_session: AsyncSession,
) -> None:
    """Reject malformed or unverifiable Bearer tokens."""
    async with await _build_client(db_session) as client:
        response = await client.get(
            "/v1/auth/me",
            headers={
                "Authorization": "Bearer definitely-not-a-valid-jwt",
            },
        )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid authentication credentials",
    }


@pytest.mark.asyncio
async def test_email_is_normalized_through_http_registration(
    db_session: AsyncSession,
) -> None:
    """Expose normalized email returned by the application service."""
    async with await _build_client(db_session) as client:
        response = await client.post(
            "/v1/auth/register",
            json={
                "email": "API-NORMALIZED@Example.COM",
                "password": "StrongPassword123!",
            },
        )

    assert response.status_code == 201

    assert response.json()["user"]["email"] == ("API-NORMALIZED@example.com")


@pytest.mark.asyncio
async def test_delete_account_requires_current_password(
    db_session: AsyncSession,
) -> None:
    """Reject permanent account deletion when password confirmation fails."""
    async with await _build_client(db_session) as client:
        register_response = await client.post(
            "/v1/auth/register",
            json={
                "email": "api-delete-account-invalid@example.com",
                "password": "StrongPassword123!",
            },
        )

        assert register_response.status_code == 201

        access_token = register_response.json()["access_token"]

        response = await client.post(
            "/v1/account/delete",
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            json={
                "password": "WrongPassword123!",
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid password",
    }


@pytest.mark.asyncio
async def test_delete_account_invalidates_existing_access_token(
    db_session: AsyncSession,
) -> None:
    """Delete the authenticated account and reject its previously issued token."""
    async with await _build_client(db_session) as client:
        register_response = await client.post(
            "/v1/auth/register",
            json={
                "email": "api-delete-account@example.com",
                "password": "StrongPassword123!",
            },
        )

        assert register_response.status_code == 201

        access_token = register_response.json()["access_token"]
        headers = {
            "Authorization": f"Bearer {access_token}",
        }

        delete_response = await client.post(
            "/v1/account/delete",
            headers=headers,
            json={
                "password": "StrongPassword123!",
            },
        )

        me_response = await client.get(
            "/v1/auth/me",
            headers=headers,
        )

    assert delete_response.status_code == 204
    assert delete_response.content == b""

    assert me_response.status_code == 401
    assert me_response.json() == {
        "detail": "Invalid authentication credentials",
    }
