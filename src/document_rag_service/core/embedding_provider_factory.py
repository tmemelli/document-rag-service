"""Create embedding providers from validated application configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from document_rag_service.core.settings import Settings

if TYPE_CHECKING:
    from document_rag_service.services.embedding_service import EmbeddingProvider


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """
    Create the embedding provider selected by application configuration.

    Provider imports are intentionally performed lazily inside each branch.
    This keeps provider-specific dependencies isolated and avoids importing the
    local sentence-transformer stack when production is configured to use the
    Cloudflare backend.

    The factory translates validated application settings into the constructor
    arguments required by each provider while exposing a common
    ``EmbeddingProvider`` interface to the service layer.

    Args:
        settings: Validated application settings containing provider selection
            and provider-specific configuration.

    Returns:
        Configured embedding provider matching ``settings.embedding_provider``.

    Raises:
        RuntimeError: If Cloudflare is selected but required credentials are
            unavailable at provider-construction time.
        ValueError: If the configured embedding provider is unsupported.
    """

    if settings.embedding_provider == "local":
        from document_rag_service.providers.local_embedding_provider import (
            LocalEmbeddingProvider,
        )

        return LocalEmbeddingProvider(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            dimension=settings.embedding_dimension,
        )

    if settings.embedding_provider == "cloudflare":
        from document_rag_service.providers.cloudflare_embedding_provider import (
            CloudflareEmbeddingProvider,
        )

        account_id = settings.cloudflare_account_id
        api_token = settings.cloudflare_api_token

        if account_id is None or api_token is None:
            raise RuntimeError("Cloudflare embedding configuration is incomplete")

        return CloudflareEmbeddingProvider(
            account_id=account_id,
            api_token=api_token.get_secret_value(),
            model_name=settings.cloudflare_embedding_model,
            tokenizer_model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
            max_sequence_length=settings.cloudflare_embedding_max_sequence_length,
            batch_size=settings.cloudflare_embedding_batch_size,
            timeout_seconds=settings.cloudflare_timeout_seconds,
            max_attempts=settings.cloudflare_max_attempts,
            retry_base_delay_seconds=settings.cloudflare_retry_base_delay_seconds,
        )

    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider!r}")
