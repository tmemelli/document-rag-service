"""
Tests for grounded LLM answer generation and prompt-size enforcement.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from document_rag_service.services.generation_service import (
    SYSTEM_PROMPT,
    GenerationService,
)


def _build_provider() -> MagicMock:
    """Create a deterministic LLM provider test double."""
    provider = MagicMock()
    provider.generate = AsyncMock()
    return provider


@pytest.mark.parametrize(
    "max_prompt_characters",
    [
        0,
        -1,
        -100,
    ],
)
def test_generation_service_rejects_non_positive_prompt_limit(
    max_prompt_characters: int,
) -> None:
    """Reject zero or negative maximum prompt sizes."""
    provider = _build_provider()

    with pytest.raises(
        ValueError,
        match="max_prompt_characters must be greater than zero",
    ):
        GenerationService(
            provider,
            max_prompt_characters=max_prompt_characters,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "   ",
        "\t",
        "\n",
        " \t\n ",
    ],
)
async def test_generate_rejects_empty_or_blank_query(
    query: str,
) -> None:
    """Reject generation requests without a meaningful user question."""
    provider = _build_provider()
    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        await service.generate(
            query=query,
            context="document context",
            language="Portuguese",
        )

    provider.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context",
    [
        "",
        " ",
        "   ",
        "\t",
        "\n",
        " \t\n ",
    ],
)
async def test_generate_rejects_empty_or_blank_context(
    context: str,
) -> None:
    """Reject generation requests without retrieved document context."""
    provider = _build_provider()
    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    with pytest.raises(
        ValueError,
        match="Context cannot be empty",
    ):
        await service.generate(
            query="What happened?",
            context=context,
            language="Portuguese",
        )

    provider.generate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language",
    [
        "",
        " ",
        "   ",
        "\t",
        "\n",
        " \t\n ",
    ],
)
async def test_generate_rejects_empty_or_blank_language(
    language: str,
) -> None:
    """Reject generation requests without a response language."""
    provider = _build_provider()
    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    with pytest.raises(
        ValueError,
        match="Response language cannot be empty",
    ):
        await service.generate(
            query="What happened?",
            context="document context",
            language=language,
        )

    provider.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_calls_provider_with_system_and_user_prompts() -> None:
    """Send the grounded system prompt and assembled user prompt to the provider."""
    provider = _build_provider()
    provider.generate.return_value = "Generated answer"

    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    result = await service.generate(
        query="What happened?",
        context="The event happened in Baker Street.",
        language="Portuguese",
    )

    assert result == "Generated answer"

    provider.generate.assert_awaited_once_with(
        system_prompt=SYSTEM_PROMPT,
        prompt=(
            "Requested language:\n"
            "Portuguese\n\n"
            "Question:\n"
            "What happened?\n\n"
            "Context:\n"
            "The event happened in Baker Street."
        ),
    )


@pytest.mark.asyncio
async def test_generate_strips_input_fields_before_building_prompt() -> None:
    """Strip surrounding whitespace from query, context, and language."""
    provider = _build_provider()
    provider.generate.return_value = "answer"

    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    await service.generate(
        query="  What happened?  ",
        context="  document context  ",
        language="  Portuguese  ",
    )

    provider.generate.assert_awaited_once_with(
        system_prompt=SYSTEM_PROMPT,
        prompt=(
            "Requested language:\n"
            "Portuguese\n\n"
            "Question:\n"
            "What happened?\n\n"
            "Context:\n"
            "document context"
        ),
    )


@pytest.mark.asyncio
async def test_generate_strips_provider_response() -> None:
    """Remove surrounding whitespace from the generated answer."""
    provider = _build_provider()
    provider.generate.return_value = "  Generated grounded answer.\n"

    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    result = await service.generate(
        query="Question",
        context="Context",
        language="English",
    )

    assert result == "Generated grounded answer."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "",
        " ",
        "   ",
        "\t",
        "\n",
        " \t\n ",
    ],
)
async def test_generate_rejects_empty_provider_response(
    answer: str,
) -> None:
    """Reject empty or whitespace-only responses returned by the LLM provider."""
    provider = _build_provider()
    provider.generate.return_value = answer

    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    with pytest.raises(
        RuntimeError,
        match="LLM provider returned an empty response",
    ):
        await service.generate(
            query="Question",
            context="Context",
            language="English",
        )


@pytest.mark.asyncio
async def test_generate_accepts_prompt_exactly_at_character_limit() -> None:
    """Accept a generated prompt whose length exactly matches the configured limit."""
    provider = _build_provider()
    provider.generate.return_value = "answer"

    bootstrap_service = GenerationService(
        provider,
        max_prompt_characters=100_000,
    )

    prompt = bootstrap_service._build_prompt(
        query="Question",
        context="Context",
        language="English",
    )

    service = GenerationService(
        provider,
        max_prompt_characters=len(prompt),
    )

    result = await service.generate(
        query="Question",
        context="Context",
        language="English",
    )

    assert result == "answer"
    provider.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_rejects_prompt_above_character_limit() -> None:
    """Reject a generated prompt that exceeds the configured character budget."""
    provider = _build_provider()

    bootstrap_service = GenerationService(
        provider,
        max_prompt_characters=100_000,
    )

    prompt = bootstrap_service._build_prompt(
        query="Question",
        context="Context",
        language="English",
    )

    service = GenerationService(
        provider,
        max_prompt_characters=len(prompt) - 1,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Generated prompt exceeds the maximum allowed size: "
            rf"{len(prompt)} > {len(prompt) - 1}"
        ),
    ):
        await service.generate(
            query="Question",
            context="Context",
            language="English",
        )

    provider.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_propagates_provider_failure() -> None:
    """Propagate generation backend failures unchanged to the caller."""
    provider = _build_provider()

    provider_error = RuntimeError("inference backend unavailable")

    provider.generate.side_effect = provider_error

    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    with pytest.raises(
        RuntimeError,
        match="inference backend unavailable",
    ) as exc_info:
        await service.generate(
            query="Question",
            context="Context",
            language="English",
        )

    assert exc_info.value is provider_error


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\t",
        "\n",
        " \t\n ",
    ],
)
def test_get_available_context_characters_rejects_empty_query(
    query: str,
) -> None:
    """Reject context-budget calculation without a meaningful question."""
    provider = _build_provider()

    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        service.get_available_context_characters(
            query=query,
            language="English",
        )


@pytest.mark.parametrize(
    "language",
    [
        "",
        " ",
        "\t",
        "\n",
        " \t\n ",
    ],
)
def test_get_available_context_characters_rejects_empty_language(
    language: str,
) -> None:
    """Reject context-budget calculation without a response language."""
    provider = _build_provider()

    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    with pytest.raises(
        ValueError,
        match="Response language cannot be empty",
    ):
        service.get_available_context_characters(
            query="Question",
            language=language,
        )


def test_get_available_context_characters_reserves_prompt_overhead() -> None:
    """Subtract prompt labels, metadata, separators, and context newline."""
    provider = _build_provider()

    max_prompt_characters = 1_000

    service = GenerationService(
        provider,
        max_prompt_characters=max_prompt_characters,
    )

    prompt_without_context = service._build_prompt(
        query="Question",
        context="",
        language="English",
    )

    expected = max_prompt_characters - len(prompt_without_context) - 1

    available = service.get_available_context_characters(
        query="Question",
        language="English",
    )

    assert available == expected


def test_get_available_context_characters_strips_query_and_language() -> None:
    """Calculate context capacity using normalized query and language values."""
    provider = _build_provider()

    service = GenerationService(
        provider,
        max_prompt_characters=1_000,
    )

    normalized = service.get_available_context_characters(
        query="Question",
        language="English",
    )

    padded = service.get_available_context_characters(
        query="  Question  ",
        language="  English  ",
    )

    assert padded == normalized


def test_available_context_exactly_fills_prompt_limit() -> None:
    """Return a context budget that produces a prompt exactly at the size limit."""
    provider = _build_provider()

    max_prompt_characters = 500

    service = GenerationService(
        provider,
        max_prompt_characters=max_prompt_characters,
    )

    available = service.get_available_context_characters(
        query="Question",
        language="English",
    )

    prompt = service._build_prompt(
        query="Question",
        context="x" * available,
        language="English",
    )

    assert len(prompt) == max_prompt_characters


def test_get_available_context_characters_rejects_zero_remaining_capacity() -> None:
    """Reject metadata that leaves no character available for document context."""
    provider = _build_provider()

    bootstrap_service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    prompt_without_context = bootstrap_service._build_prompt(
        query="Question",
        context="",
        language="English",
    )

    service = GenerationService(
        provider,
        max_prompt_characters=len(prompt_without_context) + 1,
    )

    with pytest.raises(
        ValueError,
        match=("Query and prompt metadata exceed the maximum prompt size"),
    ):
        service.get_available_context_characters(
            query="Question",
            language="English",
        )


def test_get_available_context_characters_rejects_negative_remaining_capacity() -> None:
    """Reject metadata whose size already exceeds the configured prompt budget."""
    provider = _build_provider()

    bootstrap_service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    prompt_without_context = bootstrap_service._build_prompt(
        query="Question",
        context="",
        language="English",
    )

    service = GenerationService(
        provider,
        max_prompt_characters=len(prompt_without_context),
    )

    with pytest.raises(
        ValueError,
        match=("Query and prompt metadata exceed the maximum prompt size"),
    ):
        service.get_available_context_characters(
            query="Question",
            language="English",
        )


def test_build_prompt_returns_expected_structure() -> None:
    """Build the exact provider prompt structure expected by the RAG pipeline."""
    provider = _build_provider()

    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    prompt = service._build_prompt(
        query="Who arrived?",
        context="[SOURCE 1]\nA young woman arrived.",
        language="Portuguese",
    )

    assert prompt == (
        "Requested language:\n"
        "Portuguese\n\n"
        "Question:\n"
        "Who arrived?\n\n"
        "Context:\n"
        "[SOURCE 1]\n"
        "A young woman arrived."
    )


def test_build_prompt_strips_surrounding_input_whitespace() -> None:
    """Normalize only surrounding whitespace while preserving internal content."""
    provider = _build_provider()

    service = GenerationService(
        provider,
        max_prompt_characters=10_000,
    )

    prompt = service._build_prompt(
        query="  Who arrived?  ",
        context="  line one\nline two  ",
        language="  Portuguese  ",
    )

    assert prompt == (
        "Requested language:\n"
        "Portuguese\n\n"
        "Question:\n"
        "Who arrived?\n\n"
        "Context:\n"
        "line one\n"
        "line two"
    )


def test_system_prompt_contains_grounding_requirements() -> None:
    """Keep the generation policy explicitly grounded in retrieved documents."""
    assert "using only the provided context" in SYSTEM_PROMPT
    assert "Do not use external knowledge" in SYSTEM_PROMPT
    assert "not enough information" in SYSTEM_PROMPT


def test_system_prompt_hides_internal_rag_mechanics() -> None:
    """Prevent generated answers from exposing internal retrieval implementation."""
    assert "Do not expose internal source labels" in SYSTEM_PROMPT
    assert "Do not mention chunks" in SYSTEM_PROMPT
    assert "retrieval" in SYSTEM_PROMPT
    assert "embeddings" in SYSTEM_PROMPT


def test_system_prompt_requires_requested_language() -> None:
    """Require the final answer to honor the response language selected by the user."""
    assert "Always answer in the requested language" in SYSTEM_PROMPT


def test_system_prompt_requires_plain_text_output() -> None:
    """Require provider responses to remain presentation-independent plain text."""
    assert (
        "Do not use Markdown or any other presentation-specific formatting"
        in SYSTEM_PROMPT
    )
