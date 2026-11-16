"""
Tests for retrieval-query transformation, parsing, retries, and fallback.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from document_rag_service.services.query_transformation_service import (
    QUERY_TRANSFORMATION_SYSTEM_PROMPT,
    QueryTransformationResult,
    QueryTransformationService,
)


def _build_provider() -> MagicMock:
    """Create a deterministic query-transformation provider test double."""
    provider = MagicMock()
    provider.generate = AsyncMock()
    return provider


@pytest.mark.parametrize(
    "max_generated_queries",
    [
        0,
        -1,
        -100,
    ],
)
def test_query_transformation_service_rejects_non_positive_generated_query_limit(
    max_generated_queries: int,
) -> None:
    """Reject zero or negative generated-query limits."""
    provider = _build_provider()

    with pytest.raises(
        ValueError,
        match="Max generated queries must be greater than zero",
    ):
        QueryTransformationService(
            provider,
            max_generated_queries=max_generated_queries,
        )


@pytest.mark.parametrize(
    "max_attempts",
    [
        0,
        -1,
        -100,
    ],
)
def test_query_transformation_service_rejects_non_positive_attempt_limit(
    max_attempts: int,
) -> None:
    """Reject zero or negative transformation-attempt limits."""
    provider = _build_provider()

    with pytest.raises(
        ValueError,
        match="Max attempts must be greater than zero",
    ):
        QueryTransformationService(
            provider,
            max_attempts=max_attempts,
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
async def test_transform_rejects_empty_or_blank_query(
    query: str,
) -> None:
    """Reject transformation requests without a meaningful question."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        await service.transform(query)

    provider.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_transform_strips_original_query() -> None:
    """Normalize surrounding whitespace from the original user question."""
    provider = _build_provider()
    provider.generate.return_value = json.dumps(
        {
            "queries": [
                "focused retrieval query",
            ]
        }
    )

    service = QueryTransformationService(provider)

    result = await service.transform("  What happened at Baker Street?  ")

    assert result.original_query == "What happened at Baker Street?"
    assert result.retrieval_queries[0] == "What happened at Baker Street?"


@pytest.mark.asyncio
async def test_transform_preserves_original_query_as_first_candidate() -> None:
    """Keep the original question as the highest-priority retrieval query."""
    provider = _build_provider()

    provider.generate.return_value = json.dumps(
        {
            "queries": [
                "focused query one",
                "focused query two",
            ]
        }
    )

    service = QueryTransformationService(provider)

    result = await service.transform("Original user question")

    assert result.retrieval_queries == [
        "Original user question",
        "focused query one",
        "focused query two",
    ]


@pytest.mark.asyncio
async def test_transform_returns_success_metadata() -> None:
    """Report successful transformation state and attempt count."""
    provider = _build_provider()

    provider.generate.return_value = json.dumps(
        {
            "queries": [
                "focused query",
            ]
        }
    )

    service = QueryTransformationService(provider)

    result = await service.transform("Original question")

    assert isinstance(
        result,
        QueryTransformationResult,
    )
    assert result.original_query == "Original question"
    assert result.used_fallback is False
    assert result.attempt_count == 1


@pytest.mark.asyncio
async def test_transform_calls_provider_with_expected_prompts() -> None:
    """Send the transformation policy and bounded request prompt to the provider."""
    provider = _build_provider()

    provider.generate.return_value = json.dumps(
        {
            "queries": [
                "focused query",
            ]
        }
    )

    service = QueryTransformationService(
        provider,
        max_generated_queries=2,
    )

    await service.transform("What happened at Baker Street?")

    provider.generate.assert_awaited_once_with(
        system_prompt=QUERY_TRANSFORMATION_SYSTEM_PROMPT,
        prompt=(
            "Original question:\n"
            "What happened at Baker Street?\n\n"
            "Generate at most 2 retrieval queries.\n"
            "Write every retrieval query in the same language as "
            "the original question."
        ),
    )


@pytest.mark.asyncio
async def test_transform_retries_after_provider_failure() -> None:
    """Retry transformation when the provider fails transiently."""
    provider = _build_provider()

    provider.generate.side_effect = [
        RuntimeError("temporary failure"),
        json.dumps(
            {
                "queries": [
                    "focused query",
                ]
            }
        ),
    ]

    service = QueryTransformationService(
        provider,
        max_attempts=2,
    )

    result = await service.transform("Original question")

    assert result.retrieval_queries == [
        "Original question",
        "focused query",
    ]
    assert result.used_fallback is False
    assert result.attempt_count == 2
    assert provider.generate.await_count == 2


@pytest.mark.asyncio
async def test_transform_retries_after_invalid_provider_response() -> None:
    """Retry transformation when provider output cannot be parsed."""
    provider = _build_provider()

    provider.generate.side_effect = [
        "not valid json",
        json.dumps(
            {
                "queries": [
                    "focused query",
                ]
            }
        ),
    ]

    service = QueryTransformationService(
        provider,
        max_attempts=2,
    )

    result = await service.transform("Original question")

    assert result.used_fallback is False
    assert result.attempt_count == 2
    assert result.retrieval_queries == [
        "Original question",
        "focused query",
    ]


@pytest.mark.asyncio
async def test_transform_falls_back_to_original_query_after_all_attempts_fail() -> None:
    """Return the original question when every transformation attempt fails."""
    provider = _build_provider()

    provider.generate.side_effect = RuntimeError("provider unavailable")

    service = QueryTransformationService(
        provider,
        max_attempts=3,
    )

    result = await service.transform("Original question")

    assert result == QueryTransformationResult(
        original_query="Original question",
        retrieval_queries=[
            "Original question",
        ],
        used_fallback=True,
        attempt_count=3,
    )

    assert provider.generate.await_count == 3


@pytest.mark.asyncio
async def test_transform_falls_back_after_repeated_invalid_json() -> None:
    """Fall back safely when every provider response is malformed JSON."""
    provider = _build_provider()

    provider.generate.return_value = "invalid json"

    service = QueryTransformationService(
        provider,
        max_attempts=2,
    )

    result = await service.transform("Original question")

    assert result.used_fallback is True
    assert result.attempt_count == 2
    assert result.retrieval_queries == [
        "Original question",
    ]


@pytest.mark.asyncio
async def test_transform_deduplicates_generated_query_matching_original() -> None:
    """Discard generated paraphrases equivalent to the original query."""
    provider = _build_provider()

    provider.generate.return_value = json.dumps(
        {
            "queries": [
                "original question",
                "different query",
            ]
        }
    )

    service = QueryTransformationService(provider)

    result = await service.transform("Original Question")

    assert result.retrieval_queries == [
        "Original Question",
        "different query",
    ]


@pytest.mark.asyncio
async def test_transform_deduplicates_generated_queries_against_each_other() -> None:
    """Remove equivalent generated retrieval queries while preserving priority."""
    provider = _build_provider()

    provider.generate.return_value = json.dumps(
        {
            "queries": [
                "Sherlock Holmes clues",
                "sherlock holmes clues",
            ]
        }
    )

    service = QueryTransformationService(provider)

    result = await service.transform("Original question")

    assert result.retrieval_queries == [
        "Original question",
        "Sherlock Holmes clues",
    ]


def test_build_prompt_includes_original_question() -> None:
    """Include the normalized original question in the provider prompt."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    prompt = service._build_prompt("What happened at Baker Street?")

    assert "Original question:\n" "What happened at Baker Street?" in prompt


def test_build_prompt_includes_generated_query_limit() -> None:
    """Tell the provider how many transformed queries it may return."""
    provider = _build_provider()

    service = QueryTransformationService(
        provider,
        max_generated_queries=4,
    )

    prompt = service._build_prompt("Original question")

    assert "Generate at most 4 retrieval queries." in prompt


def test_build_prompt_requires_original_language() -> None:
    """Require transformed queries to preserve the user's original language."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    prompt = service._build_prompt("Original question")

    assert (
        "Write every retrieval query in the same language as "
        "the original question." in prompt
    )


def test_parse_queries_accepts_valid_json() -> None:
    """Parse a valid provider response containing usable retrieval queries."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._parse_queries('{"queries": ["query one", "query two"]}')

    assert result == [
        "query one",
        "query two",
    ]


def test_parse_queries_strips_query_whitespace() -> None:
    """Normalize surrounding whitespace from generated retrieval queries."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._parse_queries('{"queries": ["  query one  ", "\\tquery two\\n"]}')

    assert result == [
        "query one",
        "query two",
    ]


@pytest.mark.parametrize(
    "response",
    [
        ("```json\n" '{"queries": ["query one"]}\n' "```"),
        ("```\n" '{"queries": ["query one"]}\n' "```"),
    ],
)
def test_parse_queries_accepts_markdown_code_fences(
    response: str,
) -> None:
    """Tolerate fenced JSON returned by otherwise compliant providers."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._parse_queries(response)

    assert result == [
        "query one",
    ]


def test_parse_queries_limits_generated_query_count() -> None:
    """Ignore generated queries beyond the configured transformation limit."""
    provider = _build_provider()

    service = QueryTransformationService(
        provider,
        max_generated_queries=2,
    )

    result = service._parse_queries(
        json.dumps(
            {
                "queries": [
                    "query one",
                    "query two",
                    "query three",
                ]
            }
        )
    )

    assert result == [
        "query one",
        "query two",
    ]


def test_parse_queries_skips_non_string_entries() -> None:
    """Ignore non-string values returned inside the queries array."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._parse_queries(
        json.dumps(
            {
                "queries": [
                    "valid query",
                    123,
                ]
            }
        )
    )

    assert result == [
        "valid query",
    ]


def test_parse_queries_skips_empty_string_entries() -> None:
    """Ignore empty or whitespace-only generated retrieval queries."""
    provider = _build_provider()

    service = QueryTransformationService(
        provider,
        max_generated_queries=4,
    )

    result = service._parse_queries(
        json.dumps(
            {
                "queries": [
                    "",
                    "   ",
                    "valid query",
                    "\t",
                ]
            }
        )
    )

    assert result == [
        "valid query",
    ]


@pytest.mark.parametrize(
    "response",
    [
        "[]",
        '"text"',
        "123",
        "true",
        "null",
    ],
)
def test_parse_queries_rejects_non_object_json(
    response: str,
) -> None:
    """Require the top-level provider response to be a JSON object."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    with pytest.raises(
        ValueError,
        match="Query transformer response must be a JSON object",
    ):
        service._parse_queries(response)


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        '{"queries": "query"}',
        '{"queries": 123}',
        '{"queries": null}',
        '{"queries": {}}',
    ],
)
def test_parse_queries_rejects_invalid_queries_field(
    response: str,
) -> None:
    """Require the queries field to contain a JSON array."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    with pytest.raises(
        ValueError,
        match="Query transformer returned an invalid queries field",
    ):
        service._parse_queries(response)


@pytest.mark.parametrize(
    "response",
    [
        '{"queries": []}',
        '{"queries": [""]}',
        '{"queries": ["   "]}',
        '{"queries": [123]}',
        '{"queries": [null]}',
    ],
)
def test_parse_queries_rejects_response_without_usable_queries(
    response: str,
) -> None:
    """Reject responses that produce no valid retrieval query."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    with pytest.raises(
        ValueError,
        match="Query transformer returned no usable queries",
    ):
        service._parse_queries(response)


def test_parse_queries_propagates_json_decode_error() -> None:
    """Expose malformed JSON to the retry mechanism."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    with pytest.raises(json.JSONDecodeError):
        service._parse_queries("not-json")


def test_deduplicate_queries_preserves_first_occurrence_order() -> None:
    """Keep the first occurrence of each unique retrieval query."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._deduplicate_queries(
        [
            "first query",
            "second query",
            "first query",
            "third query",
        ]
    )

    assert result == [
        "first query",
        "second query",
        "third query",
    ]


def test_deduplicate_queries_is_case_insensitive() -> None:
    """Treat case-only variations as duplicate retrieval queries."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._deduplicate_queries(
        [
            "Sherlock Holmes",
            "sherlock holmes",
            "SHERLOCK HOLMES",
        ]
    )

    assert result == [
        "Sherlock Holmes",
    ]


def test_deduplicate_queries_ignores_punctuation() -> None:
    """Treat punctuation-only variations as duplicate retrieval queries."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._deduplicate_queries(
        [
            "Who is Sherlock Holmes?",
            "Who is Sherlock Holmes",
            "Who is Sherlock Holmes!!!",
        ]
    )

    assert result == [
        "Who is Sherlock Holmes?",
    ]


def test_deduplicate_queries_ignores_repeated_whitespace() -> None:
    """Treat whitespace-only formatting differences as duplicates."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._deduplicate_queries(
        [
            "Sherlock Holmes clues",
            "Sherlock   Holmes   clues",
            "Sherlock\tHolmes\nclues",
        ]
    )

    assert result == [
        "Sherlock Holmes clues",
    ]


def test_deduplicate_queries_strips_returned_values() -> None:
    """Return normalized surrounding whitespace from retained queries."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._deduplicate_queries(
        [
            "  first query  ",
            "  second query\t",
        ]
    )

    assert result == [
        "first query",
        "second query",
    ]


def test_deduplicate_queries_discards_empty_values() -> None:
    """Discard empty retrieval-query candidates."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._deduplicate_queries(
        [
            "",
            " ",
            "\t",
            "valid query",
        ]
    )

    assert result == [
        "valid query",
    ]


def test_deduplicate_queries_preserves_distinct_relationships() -> None:
    """Keep queries that differ in meaningful terms or relationships."""
    provider = _build_provider()

    service = QueryTransformationService(provider)

    result = service._deduplicate_queries(
        [
            "Holmes observed the woman's sleeve",
            "Holmes observed the woman's hand",
        ]
    )

    assert result == [
        "Holmes observed the woman's sleeve",
        "Holmes observed the woman's hand",
    ]


def test_system_prompt_requires_json_output() -> None:
    """Keep the provider contract explicitly constrained to JSON."""
    assert "Return only valid JSON" in QUERY_TRANSFORMATION_SYSTEM_PROMPT
    assert '{"queries": ["query 1", "query 2"]}' in (QUERY_TRANSFORMATION_SYSTEM_PROMPT)


def test_system_prompt_prohibits_external_knowledge() -> None:
    """Prevent transformation from answering or enriching the user question."""
    assert (
        "Do not answer the question or add external knowledge."
        in QUERY_TRANSFORMATION_SYSTEM_PROMPT
    )


def test_system_prompt_requires_atomic_queries() -> None:
    """Keep decomposition focused on independent information needs."""
    assert (
        "Each query must target one independent information need"
        in QUERY_TRANSFORMATION_SYSTEM_PROMPT
    )
    assert "A query is atomic" in QUERY_TRANSFORMATION_SYSTEM_PROMPT


def test_system_prompt_preserves_original_constraints() -> None:
    """Require transformation to retain important entities and constraints."""
    assert (
        "Preserve the original language and important entities"
        in QUERY_TRANSFORMATION_SYSTEM_PROMPT
    )
    assert (
        "comparisons, negations, and constraints" in QUERY_TRANSFORMATION_SYSTEM_PROMPT
    )
