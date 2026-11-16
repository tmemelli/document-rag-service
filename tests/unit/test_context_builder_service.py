"""
Tests for bounded LLM context construction from document chunks.
"""

import pytest

from document_rag_service.models.document_chunk import DocumentChunk
from document_rag_service.services.context_builder_service import (
    ContextBuilderService,
    ContextBuildResult,
)


def _build_chunk(
    *,
    document_id: int,
    chunk_index: int,
    content: str,
) -> DocumentChunk:
    """Create an in-memory document chunk for context-builder tests."""
    return DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        token_count=0,
        embedding=[0.0] * 512,
    )


def test_build_rejects_empty_context_chunks() -> None:
    """Reject context construction when no chunks are provided."""
    service = ContextBuilderService()

    with pytest.raises(
        ValueError,
        match="Context chunks cannot be empty",
    ):
        service.build([])


@pytest.mark.parametrize("max_characters", [0, -1, -100])
def test_build_rejects_non_positive_character_budget(
    max_characters: int,
) -> None:
    """Reject zero or negative context character budgets."""
    service = ContextBuilderService()
    chunk = _build_chunk(
        document_id=1,
        chunk_index=0,
        content="alpha",
    )

    with pytest.raises(
        ValueError,
        match="Maximum context characters must be greater than zero",
    ):
        service.build(
            [chunk],
            max_characters=max_characters,
        )


def test_build_formats_single_chunk_with_source_metadata() -> None:
    """Format one chunk with stable source, document, and index metadata."""
    service = ContextBuilderService()
    chunk = _build_chunk(
        document_id=7,
        chunk_index=3,
        content="Sherlock Holmes observed the visitor.",
    )

    context = service.build([chunk])

    assert context == "\n".join(
        [
            "[SOURCE 1]",
            "document_id=7",
            "chunk_index=3",
            "Sherlock Holmes observed the visitor.",
        ]
    )


def test_build_strips_surrounding_chunk_content_whitespace() -> None:
    """Remove surrounding whitespace without altering internal content."""
    service = ContextBuilderService()
    chunk = _build_chunk(
        document_id=1,
        chunk_index=0,
        content="  alpha\n beta  ",
    )

    context = service.build([chunk])

    assert context.endswith("alpha\n beta")


def test_build_preserves_received_chunk_order() -> None:
    """Preserve the exact chunk order supplied by the caller."""
    service = ContextBuilderService()
    chunks = [
        _build_chunk(
            document_id=3,
            chunk_index=9,
            content="first received",
        ),
        _build_chunk(
            document_id=1,
            chunk_index=2,
            content="second received",
        ),
        _build_chunk(
            document_id=3,
            chunk_index=1,
            content="third received",
        ),
    ]

    context = service.build(chunks)

    assert context == "\n\n".join(
        [
            "\n".join(
                [
                    "[SOURCE 1]",
                    "document_id=3",
                    "chunk_index=9",
                    "first received",
                ]
            ),
            "\n".join(
                [
                    "[SOURCE 2]",
                    "document_id=1",
                    "chunk_index=2",
                    "second received",
                ]
            ),
            "\n".join(
                [
                    "[SOURCE 3]",
                    "document_id=3",
                    "chunk_index=1",
                    "third received",
                ]
            ),
        ]
    )


def test_build_numbers_sources_sequentially() -> None:
    """Assign one-based sequential source labels independent of chunk indexes."""
    service = ContextBuilderService()
    chunks = [
        _build_chunk(
            document_id=10,
            chunk_index=20,
            content="alpha",
        ),
        _build_chunk(
            document_id=10,
            chunk_index=25,
            content="beta",
        ),
    ]

    context = service.build(chunks)

    assert "[SOURCE 1]" in context
    assert "[SOURCE 2]" in context
    assert "[SOURCE 20]" not in context
    assert "[SOURCE 25]" not in context


def test_build_accepts_context_exactly_at_character_budget() -> None:
    """Include a complete context whose length exactly matches the budget."""
    service = ContextBuilderService()
    chunk = _build_chunk(
        document_id=1,
        chunk_index=0,
        content="alpha",
    )
    expected_context = service.build([chunk])

    context = service.build(
        [chunk],
        max_characters=len(expected_context),
    )

    assert context == expected_context
    assert len(context) == len(expected_context)


def test_build_rejects_when_first_complete_chunk_exceeds_budget() -> None:
    """Reject a budget that cannot contain even the first complete chunk."""
    service = ContextBuilderService()
    chunk = _build_chunk(
        document_id=1,
        chunk_index=0,
        content="alpha",
    )
    required_length = len(service.build([chunk]))

    with pytest.raises(
        ValueError,
        match="No complete context chunk fits within the context budget",
    ):
        service.build(
            [chunk],
            max_characters=required_length - 1,
        )


def test_build_stops_before_chunk_that_would_exceed_budget() -> None:
    """Stop before an oversized next chunk rather than truncating its content."""
    service = ContextBuilderService()
    first = _build_chunk(
        document_id=1,
        chunk_index=0,
        content="alpha",
    )
    second = _build_chunk(
        document_id=1,
        chunk_index=1,
        content="beta",
    )

    first_context = service.build([first])
    full_context = service.build([first, second])

    context = service.build(
        [first, second],
        max_characters=len(full_context) - 1,
    )

    assert context == first_context
    assert "beta" not in context
    assert "[SOURCE 2]" not in context


def test_build_counts_double_newline_separator_in_budget() -> None:
    """Include the two-character separator when enforcing the context budget."""
    service = ContextBuilderService()
    first = _build_chunk(
        document_id=1,
        chunk_index=0,
        content="alpha",
    )
    second = _build_chunk(
        document_id=1,
        chunk_index=1,
        content="beta",
    )

    first_context = service.build([first])
    full_context = service.build([first, second])

    assert len(full_context) > len(first_context)

    context = service.build(
        [first, second],
        max_characters=len(full_context) - 2,
    )

    assert context == first_context


def test_build_result_returns_context_and_exact_included_chunks() -> None:
    """Report both formatted context and the exact chunks that were included."""
    service = ContextBuilderService()
    first = _build_chunk(
        document_id=1,
        chunk_index=0,
        content="alpha",
    )
    second = _build_chunk(
        document_id=2,
        chunk_index=4,
        content="beta",
    )

    result = service.build_result([first, second])

    assert isinstance(result, ContextBuildResult)
    assert result.context == service.build([first, second])
    assert result.included_chunks == (first, second)


def test_build_result_reports_only_chunks_that_fit_budget() -> None:
    """Exclude chunks from metadata when they do not fit the context budget."""
    service = ContextBuilderService()
    first = _build_chunk(
        document_id=1,
        chunk_index=0,
        content="alpha",
    )
    second = _build_chunk(
        document_id=1,
        chunk_index=1,
        content="beta",
    )

    first_context = service.build([first])

    result = service.build_result(
        [first, second],
        max_characters=len(first_context),
    )

    assert result.context == first_context
    assert result.included_chunks == (first,)


def test_build_result_returns_included_chunks_as_tuple() -> None:
    """Expose included chunks as an immutable tuple in the result object."""
    service = ContextBuilderService()
    chunk = _build_chunk(
        document_id=1,
        chunk_index=0,
        content="alpha",
    )

    result = service.build_result([chunk])

    assert isinstance(result.included_chunks, tuple)
    assert result.included_chunks == (chunk,)
