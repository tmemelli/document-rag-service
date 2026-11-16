"""
Tests for the local sentence-transformer embedding provider.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from document_rag_service.providers import local_embedding_provider as provider_module
from document_rag_service.providers.local_embedding_provider import (
    LocalEmbeddingProvider,
)


def _build_model(
    *,
    dimension: int | None = 512,
    max_sequence_length: int | None = 8192,
) -> MagicMock:
    """Create a sentence-transformer model test double."""
    model = MagicMock()
    model.get_embedding_dimension.return_value = dimension
    model.max_seq_length = max_sequence_length

    tokenizer = MagicMock()
    model.tokenizer = tokenizer

    return model


def _patch_sentence_transformer(
    monkeypatch: pytest.MonkeyPatch,
    model: MagicMock,
) -> MagicMock:
    """Replace SentenceTransformer construction with a deterministic double."""
    model_class = MagicMock(
        return_value=model,
    )

    monkeypatch.setattr(
        provider_module,
        "SentenceTransformer",
        model_class,
    )

    return model_class


def test_init_rejects_empty_model_name() -> None:
    """Reject an empty embedding model identifier."""
    with pytest.raises(
        ValueError,
        match="Embedding model name cannot be empty",
    ):
        LocalEmbeddingProvider(
            model_name="",
            device="cpu",
            dimension=512,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        " ",
        "\t",
        "\n",
    ],
)
def test_init_rejects_whitespace_only_model_name(
    model_name: str,
) -> None:
    """Reject embedding model names containing only whitespace."""
    with pytest.raises(
        ValueError,
        match="Embedding model name cannot be empty",
    ):
        LocalEmbeddingProvider(
            model_name=model_name,
            device="cpu",
            dimension=512,
        )


def test_init_rejects_empty_device() -> None:
    """Reject an empty execution device."""
    with pytest.raises(
        ValueError,
        match="Embedding device cannot be empty",
    ):
        LocalEmbeddingProvider(
            model_name="test-model",
            device="",
            dimension=512,
        )


@pytest.mark.parametrize(
    "device",
    [
        " ",
        "\t",
        "\n",
    ],
)
def test_init_rejects_whitespace_only_device(
    device: str,
) -> None:
    """Reject execution-device values containing only whitespace."""
    with pytest.raises(
        ValueError,
        match="Embedding device cannot be empty",
    ):
        LocalEmbeddingProvider(
            model_name="test-model",
            device=device,
            dimension=512,
        )


@pytest.mark.parametrize(
    "dimension",
    [
        0,
        -1,
        -512,
    ],
)
def test_init_rejects_non_positive_dimension(
    dimension: int,
) -> None:
    """Reject zero and negative embedding dimensions."""
    with pytest.raises(
        ValueError,
        match="Embedding dimension must be greater than zero",
    ):
        LocalEmbeddingProvider(
            model_name="test-model",
            device="cpu",
            dimension=dimension,
        )


def test_init_builds_sentence_transformer_with_configured_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct the local model with model name, device, and truncation dimension."""
    model = _build_model(
        dimension=512,
    )

    model_class = _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    LocalEmbeddingProvider(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
        dimension=512,
    )

    model_class.assert_called_once_with(
        "Qwen/Qwen3-Embedding-0.6B",
        device="cuda",
        truncate_dim=512,
    )


def test_init_rejects_unknown_model_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject models that do not expose an embedding dimension."""
    model = _build_model(
        dimension=None,
    )

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    with pytest.raises(
        RuntimeError,
        match="Could not determine embedding dimension",
    ):
        LocalEmbeddingProvider(
            model_name="test-model",
            device="cpu",
            dimension=512,
        )


def test_init_rejects_embedding_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a loaded model whose effective dimension differs from configuration."""
    model = _build_model(
        dimension=384,
    )

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    with pytest.raises(
        RuntimeError,
        match=("Embedding dimension mismatch: " "expected 512, got 384"),
    ):
        LocalEmbeddingProvider(
            model_name="test-model",
            device="cpu",
            dimension=512,
        )


def test_init_rejects_missing_max_sequence_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject models that do not expose a maximum sequence length."""
    model = _build_model(
        dimension=512,
        max_sequence_length=None,
    )

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    with pytest.raises(
        RuntimeError,
        match="Could not determine maximum sequence length",
    ):
        LocalEmbeddingProvider(
            model_name="test-model",
            device="cpu",
            dimension=512,
        )


@pytest.mark.parametrize(
    "max_sequence_length",
    [
        0,
        -1,
        -100,
    ],
)
def test_init_rejects_non_positive_max_sequence_length(
    monkeypatch: pytest.MonkeyPatch,
    max_sequence_length: int,
) -> None:
    """Reject invalid maximum sequence lengths exposed by the model."""
    model = _build_model(
        dimension=512,
        max_sequence_length=max_sequence_length,
    )

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    with pytest.raises(
        RuntimeError,
        match=("Embedding model returned an invalid " "maximum sequence length"),
    ):
        LocalEmbeddingProvider(
            model_name="test-model",
            device="cpu",
            dimension=512,
        )


def test_model_name_returns_configured_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the model identifier supplied during provider construction."""
    model = _build_model()

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="custom/model",
        device="cpu",
        dimension=512,
    )

    assert provider.model_name == "custom/model"


def test_dimension_returns_validated_model_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the validated model embedding dimension."""
    model = _build_model(
        dimension=512,
    )

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    assert provider.dimension == 512


def test_max_sequence_length_returns_validated_model_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the validated tokenizer sequence limit."""
    model = _build_model(
        max_sequence_length=4096,
    )

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    assert provider.max_sequence_length == 4096


@pytest.mark.asyncio
async def test_embed_documents_calls_model_encode_with_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embed document texts directly without query-specific prompting."""
    model = _build_model()

    model.encode.return_value = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    result = await provider.embed_documents(
        [
            "first document",
            "second document",
        ]
    )

    model.encode.assert_called_once_with(
        [
            "first document",
            "second document",
        ],
        normalize_embeddings=True,
    )

    assert result == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


@pytest.mark.asyncio
async def test_embed_documents_normalizes_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request normalized document embeddings from sentence-transformers."""
    model = _build_model()

    model.encode.return_value = np.array(
        [
            [1.0, 0.0],
        ]
    )

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    await provider.embed_documents(["document"])

    assert model.encode.call_args.kwargs["normalize_embeddings"] is True


@pytest.mark.asyncio
async def test_embed_documents_preserves_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the caller's document order during model encoding."""
    model = _build_model()

    model.encode.return_value = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    await provider.embed_documents(
        [
            "first",
            "second",
            "third",
        ]
    )

    assert model.encode.call_args.args[0] == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
async def test_embed_documents_supports_empty_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate an empty document batch without altering provider behavior."""
    model = _build_model()

    model.encode.return_value = np.array([])

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    result = await provider.embed_documents([])

    model.encode.assert_called_once_with(
        [],
        normalize_embeddings=True,
    )

    assert result == []


@pytest.mark.asyncio
async def test_embed_query_uses_query_prompt_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the sentence-transformer query prompt for retrieval queries."""
    model = _build_model()

    model.encode.return_value = np.array([0.1, 0.2, 0.3])

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    result = await provider.embed_query("What is retrieval augmented generation?")

    model.encode.assert_called_once_with(
        "What is retrieval augmented generation?",
        prompt_name="query",
        normalize_embeddings=True,
    )

    assert result == [
        0.1,
        0.2,
        0.3,
    ]


@pytest.mark.asyncio
async def test_embed_query_normalizes_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request normalized query embeddings from sentence-transformers."""
    model = _build_model()

    model.encode.return_value = np.array([1.0, 0.0])

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    await provider.embed_query("query")

    assert model.encode.call_args.kwargs["normalize_embeddings"] is True


@pytest.mark.asyncio
async def test_embed_query_does_not_modify_query_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward query text exactly as supplied by the caller."""
    model = _build_model()

    model.encode.return_value = np.array([0.5])

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    query = "  preserve spacing\nand formatting  "

    await provider.embed_query(query)

    assert model.encode.call_args.args[0] == query


def test_count_document_tokens_uses_model_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count document tokens with the model's own tokenizer."""
    model = _build_model()

    model.tokenizer.encode.return_value = [
        101,
        10,
        20,
        30,
        102,
    ]

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    result = provider.count_document_tokens("document text")

    model.tokenizer.encode.assert_called_once_with(
        "document text",
        add_special_tokens=True,
        truncation=False,
        verbose=False,
    )

    assert result == 5


def test_count_document_tokens_includes_special_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request tokenizer counts including model-specific special tokens."""
    model = _build_model()

    model.tokenizer.encode.return_value = [
        1,
        2,
    ]

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    provider.count_document_tokens("document")

    assert model.tokenizer.encode.call_args.kwargs["add_special_tokens"] is True


def test_count_document_tokens_disables_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count the full document text rather than a truncated tokenizer input."""
    model = _build_model()

    model.tokenizer.encode.return_value = [
        1,
        2,
    ]

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    provider.count_document_tokens("document")

    assert model.tokenizer.encode.call_args.kwargs["truncation"] is False


def test_count_document_tokens_rejects_missing_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject token counting when the loaded model exposes no tokenizer."""
    model = _build_model()
    model.tokenizer = None

    _patch_sentence_transformer(
        monkeypatch,
        model,
    )

    provider = LocalEmbeddingProvider(
        model_name="test-model",
        device="cpu",
        dimension=512,
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding model tokenizer is not available",
    ):
        provider.count_document_tokens("document")
