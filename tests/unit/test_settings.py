"""
Tests for application settings loading, validation, and defaults.
"""

from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from document_rag_service.core.settings import Settings


def valid_settings_data() -> dict[str, Any]:
    """Return the minimum valid settings required by the application."""
    return {
        "postgres_host": "localhost",
        "postgres_db": "document_rag_test_db",
        "postgres_user": "document_rag_user",
        "postgres_password": SecretStr("test-database-password"),
        "auth_jwt_secret": SecretStr("x" * 32),
    }


def build_settings(**overrides: Any) -> Settings:
    """Build isolated settings without loading the project's .env file."""
    data = valid_settings_data()
    data.update(overrides)

    return Settings(_env_file=None, **data)  # type: ignore[call-arg]


def test_settings_accept_valid_local_configuration() -> None:
    """Accept a valid configuration using the local embedding provider."""
    settings = build_settings()

    assert settings.postgres_host == "localhost"
    assert settings.postgres_db == "document_rag_test_db"
    assert settings.embedding_provider == "local"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.document_upload_rate_limit_per_minute == 3


@pytest.mark.parametrize("chunk_overlap", [1000, 1001])
def test_settings_reject_chunk_overlap_not_smaller_than_chunk_size(
    chunk_overlap: int,
) -> None:
    """Reject chunk overlap values greater than or equal to chunk size."""
    with pytest.raises(
        ValidationError,
        match="CHUNK_OVERLAP must be smaller than CHUNK_SIZE",
    ):
        build_settings(
            chunk_size=1000,
            chunk_overlap=chunk_overlap,
        )


def test_settings_accept_chunk_overlap_smaller_than_chunk_size() -> None:
    """Accept the largest valid overlap immediately below chunk size."""
    settings = build_settings(
        chunk_size=1000,
        chunk_overlap=999,
    )

    assert settings.chunk_overlap == 999


def test_settings_reject_short_jwt_secret() -> None:
    """Reject JWT signing secrets shorter than the required minimum."""
    with pytest.raises(
        ValidationError,
        match="AUTH_JWT_SECRET must contain at least 32 characters",
    ):
        build_settings(
            auth_jwt_secret=SecretStr("too-short"),
        )


@pytest.mark.parametrize("expire_minutes", [0, -1])
def test_settings_reject_non_positive_access_token_lifetime(
    expire_minutes: int,
) -> None:
    """Reject zero or negative JWT access-token lifetimes."""
    with pytest.raises(
        ValidationError,
        match="AUTH_ACCESS_TOKEN_EXPIRE_MINUTES must be greater than zero",
    ):
        build_settings(
            auth_access_token_expire_minutes=expire_minutes,
        )


@pytest.mark.parametrize("account_id", [None, "", "   "])
def test_cloudflare_provider_requires_account_id(
    account_id: str | None,
) -> None:
    """Require a non-empty Cloudflare account ID for Cloudflare embeddings."""
    with pytest.raises(
        ValidationError,
        match="CLOUDFLARE_ACCOUNT_ID is required",
    ):
        build_settings(
            embedding_provider="cloudflare",
            cloudflare_account_id=account_id,
            cloudflare_api_token=SecretStr("test-cloudflare-token"),
        )


def test_cloudflare_provider_requires_api_token() -> None:
    """Require a Cloudflare API token when Cloudflare embeddings are enabled."""
    with pytest.raises(
        ValidationError,
        match="CLOUDFLARE_API_TOKEN is required",
    ):
        build_settings(
            embedding_provider="cloudflare",
            cloudflare_account_id="test-account-id",
        )


def test_cloudflare_provider_rejects_blank_api_token() -> None:
    """Reject Cloudflare API tokens containing only whitespace."""
    with pytest.raises(
        ValidationError,
        match="CLOUDFLARE_API_TOKEN is required",
    ):
        build_settings(
            embedding_provider="cloudflare",
            cloudflare_account_id="test-account-id",
            cloudflare_api_token=SecretStr("   "),
        )


def test_settings_accept_valid_cloudflare_configuration() -> None:
    """Accept a complete and valid Cloudflare embedding configuration."""
    settings = build_settings(
        embedding_provider="cloudflare",
        cloudflare_account_id="test-account-id",
        cloudflare_api_token=SecretStr("test-cloudflare-token"),
    )

    assert settings.embedding_provider == "cloudflare"
    assert settings.cloudflare_account_id == "test-account-id"
    assert settings.cloudflare_api_token is not None
    assert settings.cloudflare_embedding_batch_size == 32
