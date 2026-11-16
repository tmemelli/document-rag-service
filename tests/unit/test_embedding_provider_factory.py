"""
Tests for embedding-provider construction from application settings.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from document_rag_service.core.embedding_provider_factory import (
    create_embedding_provider,
)
from document_rag_service.core.settings import Settings
from document_rag_service.providers import (
    cloudflare_embedding_provider as cloudflare_provider_module,
)
from document_rag_service.providers import (
    local_embedding_provider as local_provider_module,
)


def _build_settings() -> MagicMock:
    """Create deterministic application settings for provider-factory tests."""
    settings = MagicMock(spec=Settings)

    settings.embedding_provider = "local"
    settings.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
    settings.embedding_device = "cpu"
    settings.embedding_dimension = 512

    settings.cloudflare_account_id = "test-account-id"
    settings.cloudflare_api_token = SecretStr("test-cloudflare-api-token")
    settings.cloudflare_embedding_model = "@cf/qwen/qwen3-embedding-0.6b"
    settings.cloudflare_embedding_max_sequence_length = 8192
    settings.cloudflare_embedding_batch_size = 32
    settings.cloudflare_timeout_seconds = 30.0
    settings.cloudflare_max_attempts = 3
    settings.cloudflare_retry_base_delay_seconds = 1.0

    return settings


def test_create_embedding_provider_builds_local_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the local embedding provider with configured model settings."""
    settings = _build_settings()
    settings.embedding_provider = "local"

    provider_instance = MagicMock()

    provider_class = MagicMock(
        return_value=provider_instance,
    )

    monkeypatch.setattr(
        local_provider_module,
        "LocalEmbeddingProvider",
        provider_class,
    )

    result = create_embedding_provider(settings)

    provider_class.assert_called_once_with(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device="cpu",
        dimension=512,
    )

    assert result is provider_instance


def test_create_embedding_provider_builds_cloudflare_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build Cloudflare embeddings from validated provider configuration."""
    settings = _build_settings()
    settings.embedding_provider = "cloudflare"

    provider_instance = MagicMock()

    provider_class = MagicMock(
        return_value=provider_instance,
    )

    monkeypatch.setattr(
        cloudflare_provider_module,
        "CloudflareEmbeddingProvider",
        provider_class,
    )

    result = create_embedding_provider(settings)

    provider_class.assert_called_once_with(
        account_id="test-account-id",
        api_token="test-cloudflare-api-token",
        model_name="@cf/qwen/qwen3-embedding-0.6b",
        tokenizer_model_name="Qwen/Qwen3-Embedding-0.6B",
        dimension=512,
        max_sequence_length=8192,
        batch_size=32,
        timeout_seconds=30.0,
        max_attempts=3,
        retry_base_delay_seconds=1.0,
    )

    assert result is provider_instance


def test_create_embedding_provider_extracts_cloudflare_secret_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the underlying API-token value rather than the SecretStr object."""
    settings = _build_settings()
    settings.embedding_provider = "cloudflare"

    settings.cloudflare_api_token = SecretStr("actual-secret-token")

    provider_class = MagicMock()

    monkeypatch.setattr(
        cloudflare_provider_module,
        "CloudflareEmbeddingProvider",
        provider_class,
    )

    create_embedding_provider(settings)

    assert provider_class.call_args.kwargs["api_token"] == "actual-secret-token"


def test_create_embedding_provider_rejects_missing_cloudflare_account_id() -> None:
    """Reject Cloudflare construction when the account ID is unavailable."""
    settings = _build_settings()
    settings.embedding_provider = "cloudflare"
    settings.cloudflare_account_id = None

    with pytest.raises(
        RuntimeError,
        match="Cloudflare embedding configuration is incomplete",
    ):
        create_embedding_provider(settings)


def test_create_embedding_provider_rejects_missing_cloudflare_api_token() -> None:
    """Reject Cloudflare construction when the API token is unavailable."""
    settings = _build_settings()
    settings.embedding_provider = "cloudflare"
    settings.cloudflare_api_token = None

    with pytest.raises(
        RuntimeError,
        match="Cloudflare embedding configuration is incomplete",
    ):
        create_embedding_provider(settings)


def test_create_embedding_provider_rejects_unsupported_provider() -> None:
    """Reject embedding backends unknown to the provider factory."""
    settings = _build_settings()

    settings.embedding_provider = "unsupported"

    with pytest.raises(
        ValueError,
        match="Unsupported embedding provider: 'unsupported'",
    ):
        create_embedding_provider(settings)
