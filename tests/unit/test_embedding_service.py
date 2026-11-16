"""
Tests for provider-agnostic embedding service validation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from document_rag_service.services.embedding_service import EmbeddingService


def _build_provider(
    *,
    model_name: str = "test-embedding-model",
    dimension: int = 3,
    max_sequence_length: int = 512,
) -> MagicMock:
    """Create a deterministic embedding provider test double."""
    provider = MagicMock()

    provider.model_name = model_name
    provider.dimension = dimension
    provider.max_sequence_length = max_sequence_length

    provider.embed_documents = AsyncMock()
    provider.embed_query = AsyncMock()
    provider.aclose = AsyncMock()
    provider.count_document_tokens = MagicMock()

    return provider


def test_model_name_returns_provider_model_name() -> None:
    """Expose the embedding model name reported by the provider."""
    provider = _build_provider(
        model_name="Qwen/Qwen3-Embedding-0.6B",
    )
    service = EmbeddingService(provider)

    assert service.model_name == "Qwen/Qwen3-Embedding-0.6B"


def test_dimension_returns_provider_dimension() -> None:
    """Expose the embedding vector dimension reported by the provider."""
    provider = _build_provider(
        dimension=512,
    )
    service = EmbeddingService(provider)

    assert service.dimension == 512


def test_max_sequence_length_returns_provider_value() -> None:
    """Expose a positive maximum sequence length from the provider."""
    provider = _build_provider(
        max_sequence_length=8192,
    )
    service = EmbeddingService(provider)

    assert service.max_sequence_length == 8192


@pytest.mark.parametrize(
    "max_sequence_length",
    [
        0,
        -1,
        -100,
    ],
)
def test_max_sequence_length_rejects_non_positive_provider_value(
    max_sequence_length: int,
) -> None:
    """Reject invalid provider sequence-length metadata."""
    provider = _build_provider(
        max_sequence_length=max_sequence_length,
    )
    service = EmbeddingService(provider)

    with pytest.raises(
        RuntimeError,
        match=("Embedding provider returned an invalid maximum sequence length"),
    ):
        _ = service.max_sequence_length


@pytest.mark.asyncio
async def test_aclose_delegates_resource_cleanup_to_provider() -> None:
    """Delegate asynchronous resource cleanup to the embedding provider."""
    provider = _build_provider()
    service = EmbeddingService(provider)

    await service.aclose()

    provider.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_embed_documents_returns_empty_list_without_calling_provider() -> None:
    """Return immediately when there are no document texts to embed."""
    provider = _build_provider()
    service = EmbeddingService(provider)

    result = await service.embed_documents([])

    assert result == []
    provider.embed_documents.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "texts",
    [
        [""],
        [" "],
        ["\t"],
        ["\n"],
        ["valid text", ""],
        ["valid text", "   "],
    ],
)
async def test_embed_documents_rejects_empty_document_text(
    texts: list[str],
) -> None:
    """Reject collections containing empty or whitespace-only documents."""
    provider = _build_provider()
    service = EmbeddingService(provider)

    with pytest.raises(
        ValueError,
        match="Document text cannot be empty",
    ):
        await service.embed_documents(texts)

    provider.embed_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_documents_delegates_texts_to_provider() -> None:
    """Send document texts to the configured embedding provider unchanged."""
    provider = _build_provider(
        dimension=3,
    )
    provider.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    service = EmbeddingService(provider)

    texts = [
        "first document",
        "second document",
    ]

    result = await service.embed_documents(texts)

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    provider.embed_documents.assert_awaited_once_with(texts)


@pytest.mark.asyncio
async def test_embed_documents_preserves_provider_embedding_order() -> None:
    """Return embeddings in the exact order produced for input documents."""
    provider = _build_provider(
        dimension=2,
    )
    provider.embed_documents.return_value = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]

    service = EmbeddingService(provider)

    result = await service.embed_documents(
        [
            "alpha",
            "beta",
            "gamma",
        ]
    )

    assert result == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]


@pytest.mark.asyncio
async def test_embed_documents_rejects_missing_embedding() -> None:
    """Reject provider output containing fewer embeddings than inputs."""
    provider = _build_provider(
        dimension=3,
    )
    provider.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
    ]

    service = EmbeddingService(provider)

    with pytest.raises(
        RuntimeError,
        match=("Embedding provider returned an unexpected number of embeddings"),
    ):
        await service.embed_documents(
            [
                "first document",
                "second document",
            ]
        )


@pytest.mark.asyncio
async def test_embed_documents_rejects_extra_embedding() -> None:
    """Reject provider output containing more embeddings than inputs."""
    provider = _build_provider(
        dimension=3,
    )
    provider.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    service = EmbeddingService(provider)

    with pytest.raises(
        RuntimeError,
        match=("Embedding provider returned an unexpected number of embeddings"),
    ):
        await service.embed_documents(["only one document"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_embedding",
    [
        [],
        [0.1],
        [0.1, 0.2],
        [0.1, 0.2, 0.3, 0.4],
    ],
)
async def test_embed_documents_rejects_unexpected_embedding_dimension(
    invalid_embedding: list[float],
) -> None:
    """Reject document vectors whose size differs from provider metadata."""
    provider = _build_provider(
        dimension=3,
    )
    provider.embed_documents.return_value = [
        invalid_embedding,
    ]

    service = EmbeddingService(provider)

    with pytest.raises(
        RuntimeError,
        match=("Embedding provider returned an unexpected embedding dimension"),
    ):
        await service.embed_documents(["document text"])


@pytest.mark.asyncio
async def test_embed_documents_rejects_invalid_dimension_in_any_vector() -> None:
    """Validate every returned document vector rather than only the first."""
    provider = _build_provider(
        dimension=3,
    )
    provider.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5],
        [0.6, 0.7, 0.8],
    ]

    service = EmbeddingService(provider)

    with pytest.raises(
        RuntimeError,
        match=("Embedding provider returned an unexpected embedding dimension"),
    ):
        await service.embed_documents(
            [
                "first",
                "second",
                "third",
            ]
        )


@pytest.mark.asyncio
async def test_embed_documents_propagates_provider_failure() -> None:
    """Propagate embedding backend failures to the caller."""
    provider = _build_provider()

    provider_error = RuntimeError("embedding backend unavailable")

    provider.embed_documents.side_effect = provider_error

    service = EmbeddingService(provider)

    with pytest.raises(
        RuntimeError,
        match="embedding backend unavailable",
    ) as exc_info:
        await service.embed_documents(["document text"])

    assert exc_info.value is provider_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\t",
        "\n",
        " \t\n ",
    ],
)
async def test_embed_query_rejects_empty_or_blank_text(
    text: str,
) -> None:
    """Reject empty or whitespace-only retrieval queries."""
    provider = _build_provider()
    service = EmbeddingService(provider)

    with pytest.raises(
        ValueError,
        match="Query text cannot be empty",
    ):
        await service.embed_query(text)

    provider.embed_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_query_delegates_query_to_provider() -> None:
    """Generate a query embedding through the configured provider."""
    provider = _build_provider(
        dimension=3,
    )
    provider.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    service = EmbeddingService(provider)

    result = await service.embed_query("What is retrieval augmented generation?")

    assert result == [
        0.1,
        0.2,
        0.3,
    ]

    provider.embed_query.assert_awaited_once_with(
        "What is retrieval augmented generation?"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "embedding",
    [
        [],
        [0.1],
        [0.1, 0.2],
        [0.1, 0.2, 0.3, 0.4],
    ],
)
async def test_embed_query_rejects_unexpected_embedding_dimension(
    embedding: list[float],
) -> None:
    """Reject query vectors whose dimension differs from provider metadata."""
    provider = _build_provider(
        dimension=3,
    )
    provider.embed_query.return_value = embedding

    service = EmbeddingService(provider)

    with pytest.raises(
        RuntimeError,
        match=("Embedding provider returned an unexpected embedding dimension"),
    ):
        await service.embed_query("query text")


@pytest.mark.asyncio
async def test_embed_query_propagates_provider_failure() -> None:
    """Propagate query-embedding backend failures to the caller."""
    provider = _build_provider()

    provider_error = RuntimeError("query embedding failed")

    provider.embed_query.side_effect = provider_error

    service = EmbeddingService(provider)

    with pytest.raises(
        RuntimeError,
        match="query embedding failed",
    ) as exc_info:
        await service.embed_query("query text")

    assert exc_info.value is provider_error


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\t",
        "\n",
        " \t\n ",
    ],
)
def test_count_document_tokens_rejects_empty_or_blank_text(
    text: str,
) -> None:
    """Reject token counting for empty or whitespace-only documents."""
    provider = _build_provider()
    service = EmbeddingService(provider)

    with pytest.raises(
        ValueError,
        match="Document text cannot be empty",
    ):
        service.count_document_tokens(text)

    provider.count_document_tokens.assert_not_called()


def test_count_document_tokens_delegates_to_provider() -> None:
    """Return the token count produced by the provider-compatible tokenizer."""
    provider = _build_provider()
    provider.count_document_tokens.return_value = 37

    service = EmbeddingService(provider)

    result = service.count_document_tokens("document text")

    assert result == 37

    provider.count_document_tokens.assert_called_once_with("document text")


@pytest.mark.parametrize(
    "token_count",
    [
        0,
        -1,
        -100,
    ],
)
def test_count_document_tokens_rejects_non_positive_provider_result(
    token_count: int,
) -> None:
    """Reject invalid token counts returned by the embedding provider."""
    provider = _build_provider()
    provider.count_document_tokens.return_value = token_count

    service = EmbeddingService(provider)

    with pytest.raises(
        RuntimeError,
        match="Embedding provider returned an invalid token count",
    ):
        service.count_document_tokens("document text")


def test_count_document_tokens_propagates_provider_failure() -> None:
    """Propagate tokenizer failures from the configured provider."""
    provider = _build_provider()

    provider_error = RuntimeError("tokenizer unavailable")

    provider.count_document_tokens.side_effect = provider_error

    service = EmbeddingService(provider)

    with pytest.raises(
        RuntimeError,
        match="tokenizer unavailable",
    ) as exc_info:
        service.count_document_tokens("document text")

    assert exc_info.value is provider_error
