"""
Tests for the HTTP client adapter used by LLM inference.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from document_rag_service.providers.llm_inference_provider import (
    LLMInferenceProvider,
)


def _build_client() -> MagicMock:
    """Create an asynchronous HTTP client test double."""
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    return client


def _build_provider(
    client: MagicMock,
    *,
    provider: str = "groq",
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> LLMInferenceProvider:
    """Create an inference provider with deterministic configuration."""
    return LLMInferenceProvider(
        client=client,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _build_response(
    *,
    status_code: int = 200,
    json_data: object | None = None,
    text: str | None = None,
) -> httpx.Response:
    """Create a real HTTPX response suitable for status validation."""
    request = httpx.Request(
        "POST",
        "https://example.com/v1/inference",
    )

    if json_data is not None:
        return httpx.Response(
            status_code=status_code,
            json=json_data,
            request=request,
        )

    return httpx.Response(
        status_code=status_code,
        text=text or "",
        request=request,
    )


@pytest.mark.asyncio
async def test_generate_posts_to_inference_endpoint() -> None:
    """Send inference requests to the expected relative API endpoint."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "content": "Generated answer",
        }
    )

    provider = _build_provider(client)

    await provider.generate(
        system_prompt="System instructions",
        prompt="User prompt",
    )

    client.post.assert_awaited_once()

    assert client.post.await_args is not None
    assert client.post.await_args.args[0] == "v1/inference"


@pytest.mark.asyncio
async def test_generate_builds_complete_request_payload() -> None:
    """Send provider, model, prompts, temperature, and token budget."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "content": "Generated answer",
        }
    )

    provider = _build_provider(
        client,
        provider="gemini",
        model="gemini-3.1-flash-lite",
        temperature=0.7,
        max_tokens=1500,
    )

    await provider.generate(
        system_prompt="You are a helpful assistant.",
        prompt="Explain RAG.",
    )

    client.post.assert_awaited_once_with(
        "v1/inference",
        json={
            "provider": "gemini",
            "model": "gemini-3.1-flash-lite",
            "system_prompt": "You are a helpful assistant.",
            "prompt": "Explain RAG.",
            "temperature": 0.7,
            "max_tokens": 1500,
        },
    )


@pytest.mark.asyncio
async def test_generate_uses_default_generation_configuration() -> None:
    """Use documented temperature and token defaults when omitted."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "content": "Generated answer",
        }
    )

    provider = LLMInferenceProvider(
        client=client,
        provider="groq",
        model="test-model",
    )

    await provider.generate(
        system_prompt="system",
        prompt="prompt",
    )

    assert client.post.await_args is not None

    payload = client.post.await_args.kwargs["json"]

    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 500


@pytest.mark.asyncio
async def test_generate_returns_content_from_successful_response() -> None:
    """Return generated textual content from a valid inference response."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "content": "Generated answer",
        }
    )

    provider = _build_provider(client)

    result = await provider.generate(
        system_prompt="system",
        prompt="prompt",
    )

    assert result == "Generated answer"


@pytest.mark.asyncio
async def test_generate_strips_surrounding_response_whitespace() -> None:
    """Strip transport-level surrounding whitespace from generated content."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "content": "\n   Generated answer   \t",
        }
    )

    provider = _build_provider(client)

    result = await provider.generate(
        system_prompt="system",
        prompt="prompt",
    )

    assert result == "Generated answer"


@pytest.mark.asyncio
async def test_generate_preserves_internal_content_whitespace() -> None:
    """Preserve meaningful formatting inside generated content."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "content": "First paragraph\n\nSecond paragraph",
        }
    )

    provider = _build_provider(client)

    result = await provider.generate(
        system_prompt="system",
        prompt="prompt",
    )

    assert result == "First paragraph\n\nSecond paragraph"


@pytest.mark.asyncio
async def test_generate_rejects_missing_content_field() -> None:
    """Reject successful HTTP responses without generated content."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "provider": "groq",
        }
    )

    provider = _build_provider(client)

    with pytest.raises(
        RuntimeError,
        match="LLM inference service returned an invalid response",
    ):
        await provider.generate(
            system_prompt="system",
            prompt="prompt",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        None,
        123,
        1.5,
        True,
        [],
        {},
    ],
)
async def test_generate_rejects_non_string_content(
    content: object,
) -> None:
    """Reject response content whose type does not match the provider contract."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "content": content,
        }
    )

    provider = _build_provider(client)

    with pytest.raises(
        RuntimeError,
        match="LLM inference service returned an invalid response",
    ):
        await provider.generate(
            system_prompt="system",
            prompt="prompt",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "",
        " ",
        "\t",
        "\n",
        " \n\t ",
    ],
)
async def test_generate_rejects_empty_content(
    content: str,
) -> None:
    """Reject generated content containing only whitespace."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "content": content,
        }
    )

    provider = _build_provider(client)

    with pytest.raises(
        RuntimeError,
        match="LLM inference service returned empty content",
    ):
        await provider.generate(
            system_prompt="system",
            prompt="prompt",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_code",
    [
        400,
        401,
        403,
        404,
        422,
        429,
        500,
        502,
        503,
    ],
)
async def test_generate_translates_http_status_failure(
    status_code: int,
) -> None:
    """Translate upstream HTTP failures into provider runtime errors."""
    client = _build_client()

    client.post.return_value = _build_response(
        status_code=status_code,
        text="upstream failure",
    )

    provider = _build_provider(client)

    with pytest.raises(
        RuntimeError,
        match=(
            rf"LLM inference service request failed: "
            rf"status={status_code} body=upstream failure"
        ),
    ):
        await provider.generate(
            system_prompt="system",
            prompt="prompt",
        )


@pytest.mark.asyncio
async def test_generate_preserves_http_status_error_as_exception_cause() -> None:
    """Preserve the original HTTPX status error for debugging and tracing."""
    client = _build_client()

    client.post.return_value = _build_response(
        status_code=422,
        text="invalid request",
    )

    provider = _build_provider(client)

    with pytest.raises(RuntimeError) as exc_info:
        await provider.generate(
            system_prompt="system",
            prompt="prompt",
        )

    assert isinstance(
        exc_info.value.__cause__,
        httpx.HTTPStatusError,
    )


@pytest.mark.asyncio
async def test_generate_includes_upstream_error_body_in_runtime_error() -> None:
    """Expose the upstream response body when inference returns an HTTP error."""
    client = _build_client()

    client.post.return_value = _build_response(
        status_code=422,
        text='{"detail":"prompt too long"}',
    )

    provider = _build_provider(client)

    with pytest.raises(RuntimeError) as exc_info:
        await provider.generate(
            system_prompt="system",
            prompt="prompt",
        )

    message = str(exc_info.value)

    assert "status=422" in message
    assert '{"detail":"prompt too long"}' in message


@pytest.mark.asyncio
async def test_generate_propagates_network_failure() -> None:
    """Translate transport errors into provider runtime errors."""
    client = _build_client()

    request = httpx.Request(
        "POST",
        "https://example.com/v1/inference",
    )

    network_error = httpx.ConnectError(
        "connection failed",
        request=request,
    )

    client.post.side_effect = network_error

    provider = _build_provider(client)

    with pytest.raises(
        RuntimeError,
        match="LLM inference service request failed",
    ) as exc_info:
        await provider.generate(
            system_prompt="system",
            prompt="prompt",
        )

    assert exc_info.value.__cause__ is network_error


@pytest.mark.asyncio
async def test_generate_propagates_timeout_failure() -> None:
    """Translate timeout errors into provider runtime errors."""
    client = _build_client()

    request = httpx.Request(
        "POST",
        "https://example.com/v1/inference",
    )

    timeout_error = httpx.ReadTimeout(
        "inference timed out",
        request=request,
    )

    client.post.side_effect = timeout_error

    provider = _build_provider(client)

    with pytest.raises(
        RuntimeError,
        match="LLM inference service request failed",
    ) as exc_info:
        await provider.generate(
            system_prompt="system",
            prompt="prompt",
        )

    assert exc_info.value.__cause__ is timeout_error


@pytest.mark.asyncio
async def test_generate_propagates_invalid_json_response() -> None:
    """Translate malformed JSON responses into provider runtime errors."""
    client = _build_client()

    client.post.return_value = _build_response(
        status_code=200,
        text="this is not JSON",
    )

    provider = _build_provider(client)

    with pytest.raises(
        RuntimeError,
        match="LLM inference service returned invalid JSON",
    ) as exc_info:
        await provider.generate(
            system_prompt="system",
            prompt="prompt",
        )

    assert isinstance(
        exc_info.value.__cause__,
        ValueError,
    )


@pytest.mark.asyncio
async def test_generate_accepts_empty_system_prompt() -> None:
    """Forward an empty system prompt without modifying the caller's request."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "content": "answer",
        }
    )

    provider = _build_provider(client)

    await provider.generate(
        system_prompt="",
        prompt="prompt",
    )

    assert client.post.await_args is not None

    payload = client.post.await_args.kwargs["json"]

    assert payload["system_prompt"] == ""


@pytest.mark.asyncio
async def test_generate_does_not_modify_user_prompt() -> None:
    """Forward the prompt exactly as supplied by the caller."""
    client = _build_client()

    client.post.return_value = _build_response(
        json_data={
            "content": "answer",
        }
    )

    provider = _build_provider(client)

    prompt = "  Preserve these spaces and\nthis formatting.  "

    await provider.generate(
        system_prompt="system",
        prompt=prompt,
    )

    assert client.post.await_args is not None

    payload = client.post.await_args.kwargs["json"]

    assert payload["prompt"] == prompt
