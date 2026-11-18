"""
Tests for the Cloudflare Workers AI embedding provider.
"""

from unittest.mock import AsyncMock, MagicMock, call

import httpx
import pytest

from document_rag_service.providers import (
    cloudflare_embedding_provider as provider_module,
)
from document_rag_service.providers.cloudflare_embedding_provider import (
    CloudflareEmbeddingProvider,
)


def _build_response(
    *,
    status_code: int = 200,
    json_data: object | None = None,
    text: str = "",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Create an HTTPX response with an attached request."""
    request = httpx.Request(
        "POST",
        "https://api.cloudflare.com/test",
    )

    if json_data is not None:
        return httpx.Response(
            status_code=status_code,
            json=json_data,
            headers=headers,
            request=request,
        )

    return httpx.Response(
        status_code=status_code,
        text=text,
        headers=headers,
        request=request,
    )


def _success_response(
    vectors: list[list[float]],
) -> httpx.Response:
    """Create a successful Cloudflare embedding response."""
    return _build_response(
        json_data={
            "success": True,
            "result": {
                "data": vectors,
            },
        }
    )


def _input_too_large_response() -> httpx.Response:
    """Create Cloudflare's aggregate-input-too-large error response."""
    return _build_response(
        status_code=400,
        json_data={
            "errors": [
                {
                    "message": "AiError: input too big",
                    "code": 3030,
                }
            ],
            "success": False,
            "result": {},
            "messages": [],
        },
    )


def _build_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dimension: int = 2,
    max_sequence_length: int = 8192,
    batch_size: int = 32,
    timeout_seconds: float = 30.0,
    max_attempts: int = 3,
    retry_base_delay_seconds: float = 1.0,
) -> tuple[
    CloudflareEmbeddingProvider,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    """Create a provider with isolated tokenizer and HTTP client doubles."""
    tokenizer = MagicMock()

    tokenizer_factory = MagicMock(
        return_value=tokenizer,
    )

    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.aclose = AsyncMock()

    client_factory = MagicMock(
        return_value=client,
    )

    monkeypatch.setattr(
        provider_module.AutoTokenizer,
        "from_pretrained",
        tokenizer_factory,
    )

    monkeypatch.setattr(
        provider_module.httpx,
        "AsyncClient",
        client_factory,
    )

    provider = CloudflareEmbeddingProvider(
        account_id="account-123",
        api_token="secret-token",
        model_name="@cf/qwen/qwen3-embedding-0.6b",
        tokenizer_model_name="Qwen/Qwen3-Embedding-0.6B",
        dimension=dimension,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        retry_base_delay_seconds=retry_base_delay_seconds,
    )

    return (
        provider,
        tokenizer,
        client,
        tokenizer_factory,
        client_factory,
    )


@pytest.mark.parametrize(
    "account_id",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_init_rejects_empty_account_id(
    account_id: str,
) -> None:
    """Reject empty Cloudflare account identifiers."""
    with pytest.raises(
        ValueError,
        match="Cloudflare account ID cannot be empty",
    ):
        CloudflareEmbeddingProvider(
            account_id=account_id,
            api_token="token",
            model_name="model",
            tokenizer_model_name="tokenizer",
            dimension=512,
            max_sequence_length=8192,
            batch_size=32,
            timeout_seconds=30.0,
            max_attempts=3,
            retry_base_delay_seconds=1.0,
        )


@pytest.mark.parametrize(
    "api_token",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_init_rejects_empty_api_token(
    api_token: str,
) -> None:
    """Reject empty Cloudflare authentication tokens."""
    with pytest.raises(
        ValueError,
        match="Cloudflare API token cannot be empty",
    ):
        CloudflareEmbeddingProvider(
            account_id="account",
            api_token=api_token,
            model_name="model",
            tokenizer_model_name="tokenizer",
            dimension=512,
            max_sequence_length=8192,
            batch_size=32,
            timeout_seconds=30.0,
            max_attempts=3,
            retry_base_delay_seconds=1.0,
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_init_rejects_empty_model_name(
    model_name: str,
) -> None:
    """Reject empty Cloudflare embedding model names."""
    with pytest.raises(
        ValueError,
        match="Cloudflare embedding model name cannot be empty",
    ):
        CloudflareEmbeddingProvider(
            account_id="account",
            api_token="token",
            model_name=model_name,
            tokenizer_model_name="tokenizer",
            dimension=512,
            max_sequence_length=8192,
            batch_size=32,
            timeout_seconds=30.0,
            max_attempts=3,
            retry_base_delay_seconds=1.0,
        )


@pytest.mark.parametrize(
    "tokenizer_model_name",
    [
        "",
        " ",
        "\t",
        "\n",
    ],
)
def test_init_rejects_empty_tokenizer_model_name(
    tokenizer_model_name: str,
) -> None:
    """Reject empty tokenizer model identifiers."""
    with pytest.raises(
        ValueError,
        match="Tokenizer model name cannot be empty",
    ):
        CloudflareEmbeddingProvider(
            account_id="account",
            api_token="token",
            model_name="model",
            tokenizer_model_name=tokenizer_model_name,
            dimension=512,
            max_sequence_length=8192,
            batch_size=32,
            timeout_seconds=30.0,
            max_attempts=3,
            retry_base_delay_seconds=1.0,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "dimension",
            0,
            "Embedding dimension must be greater than zero",
        ),
        (
            "dimension",
            -1,
            "Embedding dimension must be greater than zero",
        ),
        (
            "max_sequence_length",
            0,
            "Maximum sequence length must be greater than zero",
        ),
        (
            "max_sequence_length",
            -1,
            "Maximum sequence length must be greater than zero",
        ),
        (
            "batch_size",
            0,
            "Cloudflare embedding batch size must be greater than zero",
        ),
        (
            "batch_size",
            -1,
            "Cloudflare embedding batch size must be greater than zero",
        ),
        (
            "timeout_seconds",
            0,
            "Cloudflare timeout must be greater than zero",
        ),
        (
            "timeout_seconds",
            -1,
            "Cloudflare timeout must be greater than zero",
        ),
        (
            "max_attempts",
            0,
            "Cloudflare maximum attempts must be greater than zero",
        ),
        (
            "max_attempts",
            -1,
            "Cloudflare maximum attempts must be greater than zero",
        ),
        (
            "retry_base_delay_seconds",
            0,
            "Cloudflare retry base delay must be greater than zero",
        ),
        (
            "retry_base_delay_seconds",
            -1,
            "Cloudflare retry base delay must be greater than zero",
        ),
    ],
)
def test_init_rejects_non_positive_numeric_configuration(
    field: str,
    value: int,
    message: str,
) -> None:
    """Reject non-positive numeric Cloudflare configuration values."""
    kwargs: dict[str, object] = {
        "account_id": "account",
        "api_token": "token",
        "model_name": "model",
        "tokenizer_model_name": "tokenizer",
        "dimension": 512,
        "max_sequence_length": 8192,
        "batch_size": 32,
        "timeout_seconds": 30.0,
        "max_attempts": 3,
        "retry_base_delay_seconds": 1.0,
    }

    kwargs[field] = value

    with pytest.raises(
        ValueError,
        match=message,
    ):
        CloudflareEmbeddingProvider(**kwargs)  # type: ignore[arg-type]


def test_init_loads_configured_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load the tokenizer corresponding to the configured local model."""
    (
        _,
        _,
        _,
        tokenizer_factory,
        _,
    ) = _build_provider(monkeypatch)

    tokenizer_factory.assert_called_once_with("Qwen/Qwen3-Embedding-0.6B")


def test_init_creates_http_client_with_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create the provider-owned HTTP client with configured timeout."""
    (
        _,
        _,
        _,
        _,
        client_factory,
    ) = _build_provider(
        monkeypatch,
        timeout_seconds=17.5,
    )

    client_factory.assert_called_once_with(
        timeout=17.5,
    )


def test_properties_return_configured_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose model, dimension, and sequence-length metadata."""
    provider, _, _, _, _ = _build_provider(
        monkeypatch,
        dimension=512,
        max_sequence_length=4096,
    )

    assert provider.model_name == "@cf/qwen/qwen3-embedding-0.6b"
    assert provider.dimension == 512
    assert provider.max_sequence_length == 4096


@pytest.mark.asyncio
async def test_embed_documents_returns_empty_list_without_http_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid remote requests when there are no documents to embed."""
    provider, _, client, _, _ = _build_provider(monkeypatch)

    result = await provider.embed_documents([])

    assert result == []
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_documents_batches_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split document embeddings according to configured batch size."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        batch_size=2,
    )

    client.post.side_effect = [
        _success_response(
            [
                [3.0, 4.0],
                [0.0, 2.0],
            ]
        ),
        _success_response(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        _success_response(
            [
                [6.0, 8.0],
            ]
        ),
    ]

    result = await provider.embed_documents(
        [
            "one",
            "two",
            "three",
            "four",
            "five",
        ]
    )

    assert client.post.await_count == 3

    assert client.post.await_args_list[0].kwargs["json"] == {
        "documents": [
            "one",
            "two",
        ]
    }

    assert client.post.await_args_list[1].kwargs["json"] == {
        "documents": [
            "three",
            "four",
        ]
    }

    assert client.post.await_args_list[2].kwargs["json"] == {
        "documents": [
            "five",
        ]
    }

    expected = [
        [0.6, 0.8],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.6, 0.8],
    ]

    assert len(result) == len(expected)

    for actual_vector, expected_vector in zip(
        result,
        expected,
        strict=True,
    ):
        assert actual_vector == pytest.approx(expected_vector)


@pytest.mark.asyncio
async def test_embed_documents_adaptively_splits_input_too_large_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split Cloudflare code 3030 batches and preserve embedding order."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        batch_size=4,
    )

    client.post.side_effect = [
        _input_too_large_response(),
        _success_response(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        _success_response(
            [
                [3.0, 4.0],
                [4.0, 3.0],
            ]
        ),
    ]

    result = await provider.embed_documents(
        [
            "one",
            "two",
            "three",
            "four",
        ]
    )

    assert client.post.await_count == 3
    assert [
        request.kwargs["json"] for request in client.post.await_args_list
    ] == [
        {"documents": ["one", "two", "three", "four"]},
        {"documents": ["one", "two"]},
        {"documents": ["three", "four"]},
    ]

    expected = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.6, 0.8],
        [0.8, 0.6],
    ]

    for actual_vector, expected_vector in zip(
        result,
        expected,
        strict=True,
    ):
        assert actual_vector == pytest.approx(expected_vector)


@pytest.mark.asyncio
async def test_embed_documents_completes_left_split_before_right_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process recursively split batches depth-first to preserve request order."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        batch_size=4,
    )

    client.post.side_effect = [
        _input_too_large_response(),
        _input_too_large_response(),
        _success_response([[1.0, 0.0]]),
        _success_response([[0.0, 1.0]]),
        _success_response(
            [
                [3.0, 4.0],
                [4.0, 3.0],
            ]
        ),
    ]

    result = await provider.embed_documents(
        [
            "one",
            "two",
            "three",
            "four",
        ]
    )

    assert [
        request.kwargs["json"] for request in client.post.await_args_list
    ] == [
        {"documents": ["one", "two", "three", "four"]},
        {"documents": ["one", "two"]},
        {"documents": ["one"]},
        {"documents": ["two"]},
        {"documents": ["three", "four"]},
    ]

    expected = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.6, 0.8],
        [0.8, 0.6],
    ]

    for actual_vector, expected_vector in zip(
        result,
        expected,
        strict=True,
    ):
        assert actual_vector == pytest.approx(expected_vector)


@pytest.mark.asyncio
async def test_embed_documents_does_not_retry_input_too_large_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split code 3030 immediately instead of retrying the same request."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        batch_size=2,
        max_attempts=3,
    )

    sleep = AsyncMock()

    monkeypatch.setattr(
        provider_module.asyncio,
        "sleep",
        sleep,
    )

    client.post.side_effect = [
        _input_too_large_response(),
        _success_response([[1.0, 0.0]]),
        _success_response([[0.0, 1.0]]),
    ]

    result = await provider.embed_documents(
        [
            "one",
            "two",
        ]
    )

    expected = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    for actual_vector, expected_vector in zip(
        result,
        expected,
        strict=True,
    ):
        assert actual_vector == pytest.approx(expected_vector)

    assert client.post.await_count == 3
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_documents_raises_when_single_input_is_still_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail explicitly when code 3030 persists for one indivisible text."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        batch_size=1,
    )

    client.post.return_value = _input_too_large_response()

    with pytest.raises(
        RuntimeError,
        match="Cloudflare embedding input is too large",
    ):
        await provider.embed_documents(
            [
                "one oversized document chunk",
            ]
        )

    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_embed_query_uses_queries_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send retrieval queries through Cloudflare's queries input field."""
    provider, _, client, _, _ = _build_provider(monkeypatch)

    client.post.return_value = _success_response(
        [
            [3.0, 4.0],
        ]
    )

    result = await provider.embed_query("What is RAG?")

    assert client.post.await_args is not None

    assert client.post.await_args.kwargs["json"] == {
        "queries": [
            "What is RAG?",
        ]
    }

    assert result == pytest.approx(
        [
            0.6,
            0.8,
        ]
    )


@pytest.mark.asyncio
async def test_embed_sends_expected_url_and_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send requests to the configured account and model with bearer auth."""
    provider, _, client, _, _ = _build_provider(monkeypatch)

    client.post.return_value = _success_response(
        [
            [1.0, 0.0],
        ]
    )

    await provider.embed_query("query")

    client.post.assert_awaited_once_with(
        (
            "https://api.cloudflare.com/client/v4/accounts/"
            "account-123/ai/run/@cf/qwen/qwen3-embedding-0.6b"
        ),
        headers={
            "Authorization": "Bearer secret-token",
        },
        json={
            "queries": [
                "query",
            ],
        },
    )


def test_count_document_tokens_uses_matching_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count complete document tokens with the configured tokenizer."""
    provider, tokenizer, _, _, _ = _build_provider(monkeypatch)

    tokenizer.encode.return_value = [
        1,
        2,
        3,
        4,
    ]

    result = provider.count_document_tokens("document text")

    tokenizer.encode.assert_called_once_with(
        "document text",
        add_special_tokens=True,
        truncation=False,
    )

    assert result == 4


@pytest.mark.asyncio
async def test_aclose_closes_provider_owned_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close the HTTP client owned by the Cloudflare provider."""
    provider, _, client, _, _ = _build_provider(monkeypatch)

    await provider.aclose()

    client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_non_retryable_http_failure_fails_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not retry HTTP errors outside the transient status set."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        max_attempts=3,
    )

    response = _build_response(
        status_code=400,
        text="bad request",
    )

    client.post.return_value = response

    with pytest.raises(
        RuntimeError,
        match=("Cloudflare embedding request failed: " "status=400 body=bad request"),
    ) as exc_info:
        await provider.embed_query("query")

    assert client.post.await_count == 1
    assert isinstance(
        exc_info.value.__cause__,
        httpx.HTTPStatusError,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code",
    [
        408,
        429,
        500,
        501,
        502,
        503,
        504,
    ],
)
async def test_retryable_http_failure_is_retried(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    """Retry configured transient Cloudflare HTTP failures."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        max_attempts=3,
    )

    sleep = AsyncMock()

    monkeypatch.setattr(
        provider_module.asyncio,
        "sleep",
        sleep,
    )

    client.post.side_effect = [
        _build_response(
            status_code=status_code,
            text="temporary failure",
        ),
        _success_response(
            [
                [1.0, 0.0],
            ]
        ),
    ]

    result = await provider.embed_query("query")

    assert result == pytest.approx(
        [
            1.0,
            0.0,
        ]
    )

    assert client.post.await_count == 2
    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_retry_after_header_controls_retry_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Honor a numeric Retry-After response header."""
    provider, _, client, _, _ = _build_provider(monkeypatch)

    sleep = AsyncMock()

    monkeypatch.setattr(
        provider_module.asyncio,
        "sleep",
        sleep,
    )

    client.post.side_effect = [
        _build_response(
            status_code=429,
            text="rate limited",
            headers={
                "Retry-After": "2.5",
            },
        ),
        _success_response(
            [
                [1.0, 0.0],
            ]
        ),
    ]

    await provider.embed_query("query")

    sleep.assert_awaited_once_with(2.5)


@pytest.mark.asyncio
async def test_invalid_retry_after_falls_back_to_exponential_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use exponential backoff when Retry-After is not numeric."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        retry_base_delay_seconds=1.5,
    )

    sleep = AsyncMock()

    monkeypatch.setattr(
        provider_module.asyncio,
        "sleep",
        sleep,
    )

    client.post.side_effect = [
        _build_response(
            status_code=429,
            text="rate limited",
            headers={
                "Retry-After": "not-a-number",
            },
        ),
        _success_response(
            [
                [1.0, 0.0],
            ]
        ),
    ]

    await provider.embed_query("query")

    sleep.assert_awaited_once_with(1.5)


@pytest.mark.asyncio
async def test_request_error_retries_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry transport failures using exponential backoff."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        max_attempts=3,
        retry_base_delay_seconds=2.0,
    )

    sleep = AsyncMock()

    monkeypatch.setattr(
        provider_module.asyncio,
        "sleep",
        sleep,
    )

    request = httpx.Request(
        "POST",
        "https://api.cloudflare.com/test",
    )

    client.post.side_effect = [
        httpx.ConnectError(
            "network down",
            request=request,
        ),
        httpx.ReadTimeout(
            "slow network",
            request=request,
        ),
        _success_response(
            [
                [1.0, 0.0],
            ]
        ),
    ]

    await provider.embed_query("query")

    assert sleep.await_args_list == [
        call(2.0),
        call(4.0),
    ]


@pytest.mark.asyncio
async def test_request_error_exhaustion_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate the final transport failure after retries are exhausted."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        max_attempts=2,
    )

    sleep = AsyncMock()

    monkeypatch.setattr(
        provider_module.asyncio,
        "sleep",
        sleep,
    )

    request = httpx.Request(
        "POST",
        "https://api.cloudflare.com/test",
    )

    network_error = httpx.ConnectError(
        "network unavailable",
        request=request,
    )

    client.post.side_effect = network_error

    with pytest.raises(
        RuntimeError,
        match=("Cloudflare embedding request failed: " "network unavailable"),
    ) as exc_info:
        await provider.embed_query("query")

    assert exc_info.value.__cause__ is network_error
    assert client.post.await_count == 2

    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_retryable_http_error_exhaustion_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail after the final retryable HTTP attempt is exhausted."""
    provider, _, client, _, _ = _build_provider(
        monkeypatch,
        max_attempts=2,
    )

    sleep = AsyncMock()

    monkeypatch.setattr(
        provider_module.asyncio,
        "sleep",
        sleep,
    )

    client.post.return_value = _build_response(
        status_code=503,
        text="service unavailable",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Cloudflare embedding request failed: "
            "status=503 body=service unavailable"
        ),
    ):
        await provider.embed_query("query")

    assert client.post.await_count == 2

    sleep.assert_awaited_once_with(1.0)


@pytest.mark.asyncio
async def test_embed_rejects_non_object_json_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject valid JSON responses whose top-level value is not an object."""
    provider, _, client, _, _ = _build_provider(monkeypatch)

    client.post.return_value = _build_response(
        json_data=[1, 2, 3],
    )

    with pytest.raises(
        RuntimeError,
        match="Cloudflare embedding response has invalid structure",
    ):
        await provider.embed_query("query")


@pytest.mark.asyncio
async def test_embed_rejects_unsuccessful_cloudflare_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject HTTP-success responses marked unsuccessful by Cloudflare."""
    provider, _, client, _, _ = _build_provider(monkeypatch)

    client.post.return_value = _build_response(
        json_data={
            "success": False,
            "errors": [
                {
                    "message": "provider error",
                }
            ],
        }
    )

    with pytest.raises(
        RuntimeError,
        match="Cloudflare embedding request was not successful",
    ):
        await provider.embed_query("query")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
async def test_embed_rejects_invalid_result_structure(
    monkeypatch: pytest.MonkeyPatch,
    result: object,
) -> None:
    """Reject responses whose result field is not an object."""
    provider, _, client, _, _ = _build_provider(monkeypatch)

    client.post.return_value = _build_response(
        json_data={
            "success": True,
            "result": result,
        }
    )

    with pytest.raises(
        RuntimeError,
        match="Cloudflare embedding response has no valid result",
    ):
        await provider.embed_query("query")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        "invalid",
        123,
    ],
)
async def test_embed_rejects_invalid_data_structure(
    monkeypatch: pytest.MonkeyPatch,
    data: object,
) -> None:
    """Reject responses whose embedding data field is not a list."""
    provider, _, client, _, _ = _build_provider(monkeypatch)

    client.post.return_value = _build_response(
        json_data={
            "success": True,
            "result": {
                "data": data,
            },
        }
    )

    with pytest.raises(
        RuntimeError,
        match="Cloudflare embedding response has no valid data",
    ):
        await provider.embed_query("query")


@pytest.mark.asyncio
async def test_embed_rejects_unexpected_embedding_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require exactly one embedding per submitted text."""
    provider, _, client, _, _ = _build_provider(monkeypatch)

    client.post.return_value = _success_response(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="Cloudflare returned an unexpected number of embeddings",
    ):
        await provider.embed_query("query")


def test_calculate_backoff_is_exponential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calculate exponential retry delays from the configured base."""
    provider, _, _, _, _ = _build_provider(
        monkeypatch,
        retry_base_delay_seconds=1.5,
    )

    assert provider._calculate_backoff(1) == pytest.approx(1.5)
    assert provider._calculate_backoff(2) == pytest.approx(3.0)
    assert provider._calculate_backoff(3) == pytest.approx(6.0)
    assert provider._calculate_backoff(4) == pytest.approx(12.0)


def test_get_retry_delay_uses_retry_after_when_numeric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer a numeric Retry-After header over local backoff."""
    provider, _, _, _, _ = _build_provider(monkeypatch)

    response = _build_response(
        status_code=429,
        headers={
            "Retry-After": "7.25",
        },
    )

    assert provider._get_retry_delay(
        response=response,
        attempt=3,
    ) == pytest.approx(7.25)


def test_get_retry_delay_uses_backoff_without_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fall back to exponential delay when Retry-After is absent."""
    provider, _, _, _, _ = _build_provider(
        monkeypatch,
        retry_base_delay_seconds=2.0,
    )

    response = _build_response(
        status_code=429,
    )

    assert provider._get_retry_delay(
        response=response,
        attempt=3,
    ) == pytest.approx(8.0)


def test_truncate_and_normalize_truncates_to_configured_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discard dimensions beyond the configured vector size."""
    provider, _, _, _, _ = _build_provider(
        monkeypatch,
        dimension=2,
    )

    result = provider._truncate_and_normalize(
        [
            3.0,
            4.0,
            100.0,
            200.0,
        ]
    )

    assert result == pytest.approx(
        [
            0.6,
            0.8,
        ]
    )


def test_truncate_and_normalize_returns_unit_length_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize the truncated embedding to L2 unit length."""
    provider, _, _, _, _ = _build_provider(
        monkeypatch,
        dimension=3,
    )

    result = provider._truncate_and_normalize(
        [
            2.0,
            3.0,
            6.0,
        ]
    )

    squared_norm = sum(value * value for value in result)

    assert squared_norm == pytest.approx(1.0)


def test_truncate_and_normalize_rejects_vector_smaller_than_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject Cloudflare vectors shorter than the configured dimension."""
    provider, _, _, _, _ = _build_provider(
        monkeypatch,
        dimension=3,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Cloudflare embedding dimension is smaller than expected: "
            "expected at least 3, got 2"
        ),
    ):
        provider._truncate_and_normalize(
            [
                1.0,
                2.0,
            ]
        )


def test_truncate_and_normalize_rejects_zero_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject zero-norm embeddings because they cannot be normalized."""
    provider, _, _, _, _ = _build_provider(
        monkeypatch,
        dimension=3,
    )

    with pytest.raises(
        RuntimeError,
        match="Cloudflare returned a zero-length embedding vector",
    ):
        provider._truncate_and_normalize(
            [
                0.0,
                0.0,
                0.0,
            ]
        )
