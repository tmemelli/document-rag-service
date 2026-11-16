"""
Tests for relevance-prioritized neighboring context expansion.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from document_rag_service.models.document_chunk import DocumentChunk
from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from document_rag_service.services.context_expansion_service import (
    ContextExpansionService,
)


def _build_chunk(
    *,
    document_id: int,
    chunk_index: int,
    content: str | None = None,
) -> DocumentChunk:
    """Create an in-memory document chunk for expansion tests."""
    return DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content or f"document {document_id} chunk {chunk_index}",
        token_count=1,
        embedding=[0.0] * 512,
    )


def _build_repository() -> MagicMock:
    """Create a document-chunk repository test double."""
    repository = MagicMock(spec=DocumentChunkRepository)
    repository.list_by_locations = AsyncMock()
    return repository


@pytest.mark.asyncio
async def test_expand_rejects_empty_retrieved_chunks() -> None:
    """Reject context expansion when retrieval returned no anchor chunks."""
    repository = _build_repository()
    service = ContextExpansionService(repository)

    with pytest.raises(
        ValueError,
        match="Retrieved chunks cannot be empty",
    ):
        await service.expand(
            user_id=42,
            retrieved_chunks=[],
        )

    repository.list_by_locations.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "neighbor_window",
    [
        -1,
        -2,
        -100,
    ],
)
async def test_expand_rejects_negative_neighbor_window(
    neighbor_window: int,
) -> None:
    """Reject negative neighboring-window sizes."""
    repository = _build_repository()
    service = ContextExpansionService(repository)

    with pytest.raises(
        ValueError,
        match="Neighbor window cannot be negative",
    ):
        await service.expand(
            user_id=42,
            retrieved_chunks=[
                _build_chunk(
                    document_id=1,
                    chunk_index=5,
                )
            ],
            neighbor_window=neighbor_window,
        )

    repository.list_by_locations.assert_not_awaited()


@pytest.mark.asyncio
async def test_expand_with_zero_window_requests_only_anchor() -> None:
    """Request only the retrieved anchor when neighbor expansion is disabled."""
    repository = _build_repository()

    anchor = _build_chunk(
        document_id=1,
        chunk_index=5,
    )

    repository.list_by_locations.return_value = [
        anchor,
    ]

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[anchor],
        neighbor_window=0,
    )

    assert result == [anchor]

    repository.list_by_locations.assert_awaited_once_with(
        user_id=42,
        locations=[
            (1, 5),
        ],
    )


@pytest.mark.asyncio
async def test_expand_requests_neighbors_before_and_after_anchor() -> None:
    """Expand one retrieval anchor symmetrically to neighboring chunks."""
    repository = _build_repository()

    anchor = _build_chunk(
        document_id=1,
        chunk_index=5,
    )

    chunks = [
        _build_chunk(document_id=1, chunk_index=4),
        anchor,
        _build_chunk(document_id=1, chunk_index=6),
    ]

    repository.list_by_locations.return_value = chunks

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[anchor],
        neighbor_window=1,
    )

    assert result == chunks

    repository.list_by_locations.assert_awaited_once_with(
        user_id=42,
        locations=[
            (1, 4),
            (1, 5),
            (1, 6),
        ],
    )


@pytest.mark.asyncio
async def test_expand_never_requests_negative_chunk_indexes() -> None:
    """Clamp the lower expansion boundary to the first document chunk."""
    repository = _build_repository()

    anchor = _build_chunk(
        document_id=1,
        chunk_index=0,
    )

    chunks = [
        anchor,
        _build_chunk(document_id=1, chunk_index=1),
        _build_chunk(document_id=1, chunk_index=2),
    ]

    repository.list_by_locations.return_value = chunks

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[anchor],
        neighbor_window=2,
    )

    assert result == chunks

    repository.list_by_locations.assert_awaited_once_with(
        user_id=42,
        locations=[
            (1, 0),
            (1, 1),
            (1, 2),
        ],
    )


@pytest.mark.asyncio
async def test_expand_merges_overlapping_regions_from_same_document() -> None:
    """Merge overlapping anchor windows into one contiguous document region."""
    repository = _build_repository()

    first_anchor = _build_chunk(
        document_id=1,
        chunk_index=5,
    )
    second_anchor = _build_chunk(
        document_id=1,
        chunk_index=6,
    )

    expanded = [_build_chunk(document_id=1, chunk_index=index) for index in range(4, 8)]

    repository.list_by_locations.return_value = expanded

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[
            first_anchor,
            second_anchor,
        ],
        neighbor_window=1,
    )

    assert result == expanded

    repository.list_by_locations.assert_awaited_once_with(
        user_id=42,
        locations=[
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
        ],
    )


@pytest.mark.asyncio
async def test_expand_deduplicates_identical_retrieval_anchors() -> None:
    """Request a location once when retrieval returns the same anchor twice."""
    repository = _build_repository()

    anchor = _build_chunk(
        document_id=1,
        chunk_index=5,
    )

    repository.list_by_locations.return_value = [
        anchor,
    ]

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[
            anchor,
            anchor,
        ],
        neighbor_window=0,
    )

    assert result == [anchor]

    repository.list_by_locations.assert_awaited_once_with(
        user_id=42,
        locations=[
            (1, 5),
        ],
    )


@pytest.mark.asyncio
async def test_expand_preserves_natural_chunk_order_inside_merged_region() -> None:
    """Return merged-region chunks in ascending document chunk order."""
    repository = _build_repository()

    higher_index_anchor = _build_chunk(
        document_id=1,
        chunk_index=6,
    )
    lower_index_anchor = _build_chunk(
        document_id=1,
        chunk_index=5,
    )

    expanded = [_build_chunk(document_id=1, chunk_index=index) for index in range(4, 8)]

    repository.list_by_locations.return_value = list(reversed(expanded))

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[
            higher_index_anchor,
            lower_index_anchor,
        ],
        neighbor_window=1,
    )

    assert [chunk.chunk_index for chunk in result] == [
        4,
        5,
        6,
        7,
    ]


@pytest.mark.asyncio
async def test_expand_orders_disjoint_regions_by_semantic_priority() -> None:
    """Prioritize retrieval relevance over physical chunk position."""
    repository = _build_repository()

    highest_priority_anchor = _build_chunk(
        document_id=1,
        chunk_index=10,
    )
    lower_priority_anchor = _build_chunk(
        document_id=1,
        chunk_index=2,
    )

    repository.list_by_locations.return_value = [
        lower_priority_anchor,
        highest_priority_anchor,
    ]

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[
            highest_priority_anchor,
            lower_priority_anchor,
        ],
        neighbor_window=0,
    )

    assert result == [
        highest_priority_anchor,
        lower_priority_anchor,
    ]

    repository.list_by_locations.assert_awaited_once_with(
        user_id=42,
        locations=[
            (1, 10),
            (1, 2),
        ],
    )


@pytest.mark.asyncio
async def test_expand_orders_regions_across_documents_by_semantic_priority() -> None:
    """Preserve retrieval-region priority across different documents."""
    repository = _build_repository()

    first_anchor = _build_chunk(
        document_id=3,
        chunk_index=20,
    )
    second_anchor = _build_chunk(
        document_id=1,
        chunk_index=5,
    )
    third_anchor = _build_chunk(
        document_id=2,
        chunk_index=8,
    )

    repository.list_by_locations.return_value = [
        third_anchor,
        second_anchor,
        first_anchor,
    ]

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=99,
        retrieved_chunks=[
            first_anchor,
            second_anchor,
            third_anchor,
        ],
        neighbor_window=0,
    )

    assert result == [
        first_anchor,
        second_anchor,
        third_anchor,
    ]

    repository.list_by_locations.assert_awaited_once_with(
        user_id=99,
        locations=[
            (3, 20),
            (1, 5),
            (2, 8),
        ],
    )


@pytest.mark.asyncio
async def test_expand_merged_region_inherits_highest_anchor_priority() -> None:
    """Give a merged region the priority of its most relevant anchor."""
    repository = _build_repository()

    highest_priority_anchor = _build_chunk(
        document_id=1,
        chunk_index=10,
    )
    other_document_anchor = _build_chunk(
        document_id=2,
        chunk_index=3,
    )
    overlapping_lower_priority_anchor = _build_chunk(
        document_id=1,
        chunk_index=11,
    )

    document_one_chunks = [
        _build_chunk(
            document_id=1,
            chunk_index=index,
        )
        for index in range(9, 13)
    ]

    repository.list_by_locations.return_value = [
        other_document_anchor,
        *reversed(document_one_chunks),
    ]

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[
            highest_priority_anchor,
            other_document_anchor,
            overlapping_lower_priority_anchor,
        ],
        neighbor_window=1,
    )

    assert result == [
        *document_one_chunks,
        other_document_anchor,
    ]

    repository.list_by_locations.assert_awaited_once_with(
        user_id=42,
        locations=[
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (2, 2),
            (2, 3),
            (2, 4),
        ],
    )


@pytest.mark.asyncio
async def test_expand_keeps_non_overlapping_regions_separate() -> None:
    """Keep distant windows from the same document as independent regions."""
    repository = _build_repository()

    first_anchor = _build_chunk(
        document_id=1,
        chunk_index=2,
    )
    second_anchor = _build_chunk(
        document_id=1,
        chunk_index=10,
    )

    first_region = [
        _build_chunk(document_id=1, chunk_index=index) for index in range(1, 4)
    ]
    second_region = [
        _build_chunk(document_id=1, chunk_index=index) for index in range(9, 12)
    ]

    repository.list_by_locations.return_value = [
        *second_region,
        *first_region,
    ]

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[
            first_anchor,
            second_anchor,
        ],
        neighbor_window=1,
    )

    assert result == [
        *first_region,
        *second_region,
    ]

    repository.list_by_locations.assert_awaited_once_with(
        user_id=42,
        locations=[
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 9),
            (1, 10),
            (1, 11),
        ],
    )


@pytest.mark.asyncio
async def test_expand_skips_missing_neighbor_locations() -> None:
    """Ignore requested neighbor locations that do not exist in storage."""
    repository = _build_repository()

    anchor = _build_chunk(
        document_id=1,
        chunk_index=5,
    )
    previous_chunk = _build_chunk(
        document_id=1,
        chunk_index=4,
    )

    repository.list_by_locations.return_value = [
        previous_chunk,
        anchor,
    ]

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[anchor],
        neighbor_window=1,
    )

    assert result == [
        previous_chunk,
        anchor,
    ]

    repository.list_by_locations.assert_awaited_once_with(
        user_id=42,
        locations=[
            (1, 4),
            (1, 5),
            (1, 6),
        ],
    )


@pytest.mark.asyncio
async def test_expand_ignores_repository_return_order() -> None:
    """Reconstruct final context order independently of repository row order."""
    repository = _build_repository()

    anchor = _build_chunk(
        document_id=1,
        chunk_index=5,
    )

    chunk_four = _build_chunk(
        document_id=1,
        chunk_index=4,
    )
    chunk_six = _build_chunk(
        document_id=1,
        chunk_index=6,
    )

    repository.list_by_locations.return_value = [
        chunk_six,
        anchor,
        chunk_four,
    ]

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[anchor],
        neighbor_window=1,
    )

    assert result == [
        chunk_four,
        anchor,
        chunk_six,
    ]


@pytest.mark.asyncio
async def test_expand_ignores_unrequested_repository_chunks() -> None:
    """Exclude repository rows that were not part of requested locations."""
    repository = _build_repository()

    anchor = _build_chunk(
        document_id=1,
        chunk_index=5,
    )

    unrelated = _build_chunk(
        document_id=99,
        chunk_index=99,
    )

    repository.list_by_locations.return_value = [
        unrelated,
        anchor,
    ]

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[anchor],
        neighbor_window=0,
    )

    assert result == [
        anchor,
    ]
    assert unrelated not in result


@pytest.mark.asyncio
async def test_expand_returns_empty_list_when_repository_finds_no_locations() -> None:
    """Return no expanded chunks when none of the requested locations exist."""
    repository = _build_repository()

    repository.list_by_locations.return_value = []

    service = ContextExpansionService(repository)

    result = await service.expand(
        user_id=42,
        retrieved_chunks=[
            _build_chunk(
                document_id=1,
                chunk_index=5,
            )
        ],
        neighbor_window=1,
    )

    assert result == []


@pytest.mark.asyncio
async def test_expand_passes_user_id_to_repository() -> None:
    """Scope neighbor retrieval to the authenticated document owner."""
    repository = _build_repository()

    anchor = _build_chunk(
        document_id=7,
        chunk_index=3,
    )

    repository.list_by_locations.return_value = [
        anchor,
    ]

    service = ContextExpansionService(repository)

    await service.expand(
        user_id=12345,
        retrieved_chunks=[anchor],
        neighbor_window=0,
    )

    repository.list_by_locations.assert_awaited_once_with(
        user_id=12345,
        locations=[
            (7, 3),
        ],
    )


@pytest.mark.asyncio
async def test_expand_propagates_repository_failure() -> None:
    """Propagate storage failures instead of returning incomplete context."""
    repository = _build_repository()

    repository_error = RuntimeError("database unavailable")

    repository.list_by_locations.side_effect = repository_error

    service = ContextExpansionService(repository)

    with pytest.raises(
        RuntimeError,
        match="database unavailable",
    ) as exc_info:
        await service.expand(
            user_id=42,
            retrieved_chunks=[
                _build_chunk(
                    document_id=1,
                    chunk_index=5,
                )
            ],
            neighbor_window=1,
        )

    assert exc_info.value is repository_error
