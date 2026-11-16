"""
Tests for dense retrieval and multi-query Reciprocal Rank Fusion.
"""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from document_rag_service.models.document import Document
from document_rag_service.models.document_chunk import DocumentChunk
from document_rag_service.services.retrieval_service import (
    RetrievalResult,
    RetrievalService,
)


def _build_document(
    *,
    document_id: int,
    status: str = "ready",
    user_id: int = 42,
) -> Document:
    """Create an in-memory document for retrieval tests."""
    return Document(
        id=document_id,
        user_id=user_id,
        title=f"Document {document_id}",
        source_type="upload",
        status=status,
    )


def _build_chunk(
    *,
    chunk_id: int,
    document_id: int = 1,
    chunk_index: int = 0,
) -> DocumentChunk:
    """Create an in-memory document chunk for retrieval tests."""
    return DocumentChunk(
        id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=f"chunk {chunk_id}",
        token_count=10,
        embedding=[0.0] * 512,
    )


def _build_service() -> tuple[
    RetrievalService,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    """Create the retrieval service with deterministic dependency doubles."""
    chunk_repository = MagicMock()
    chunk_repository.search_similar = AsyncMock()

    document_repository = MagicMock()
    document_repository.list_by_ids = AsyncMock()

    embedding_service = MagicMock()
    embedding_service.embed_query = AsyncMock()

    service = RetrievalService(
        chunk_repository=chunk_repository,
        document_repository=document_repository,
        embedding_service=embedding_service,
    )

    return (
        service,
        chunk_repository,
        document_repository,
        embedding_service,
    )


def test_normalize_queries_rejects_empty_query_list() -> None:
    """Reject retrieval when no query candidates are provided."""
    service, _, _, _ = _build_service()

    with pytest.raises(
        ValueError,
        match="At least one retrieval query is required",
    ):
        service._normalize_queries([])


@pytest.mark.parametrize(
    "queries",
    [
        [""],
        [" "],
        ["\t"],
        ["\n"],
        ["valid query", ""],
        ["valid query", "   "],
    ],
)
def test_normalize_queries_rejects_empty_query(
    queries: list[str],
) -> None:
    """Reject any retrieval-query candidate that is empty after stripping."""
    service, _, _, _ = _build_service()

    with pytest.raises(
        ValueError,
        match="Retrieval queries cannot be empty",
    ):
        service._normalize_queries(queries)


def test_normalize_queries_strips_surrounding_whitespace() -> None:
    """Strip surrounding whitespace while preserving query content."""
    service, _, _, _ = _build_service()

    result = service._normalize_queries(
        [
            "  first query  ",
            "\tsecond query\n",
        ]
    )

    assert result == [
        "first query",
        "second query",
    ]


def test_normalize_queries_deduplicates_case_insensitively() -> None:
    """Treat case-only query variations as duplicates."""
    service, _, _, _ = _build_service()

    result = service._normalize_queries(
        [
            "Sherlock Holmes",
            "sherlock holmes",
            "SHERLOCK HOLMES",
        ]
    )

    assert result == [
        "Sherlock Holmes",
    ]


def test_normalize_queries_preserves_first_occurrence_order() -> None:
    """Keep the first occurrence of each distinct retrieval query."""
    service, _, _, _ = _build_service()

    result = service._normalize_queries(
        [
            "first",
            "second",
            "FIRST",
            "third",
        ]
    )

    assert result == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
async def test_retrieve_rejects_empty_document_selection() -> None:
    """Reject retrieval when no document is selected."""
    service, chunks, documents, embedding = _build_service()

    with pytest.raises(
        ValueError,
        match="At least one document must be selected",
    ):
        await service.retrieve(
            user_id=42,
            document_ids=[],
            queries=["query"],
        )

    documents.list_by_ids.assert_not_awaited()
    embedding.embed_query.assert_not_awaited()
    chunks.search_similar.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limit",
    [
        0,
        -1,
        -100,
    ],
)
async def test_retrieve_rejects_non_positive_limit(
    limit: int,
) -> None:
    """Reject zero or negative final retrieval limits."""
    service, _, documents, _ = _build_service()

    with pytest.raises(
        ValueError,
        match="Limit must be greater than zero",
    ):
        await service.retrieve(
            user_id=42,
            document_ids=[1],
            queries=["query"],
            limit=limit,
        )

    documents.list_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_rejects_candidate_limit_smaller_than_final_limit() -> None:
    """Require per-query candidate search to cover at least the final limit."""
    service, _, documents, _ = _build_service()

    with pytest.raises(
        ValueError,
        match="Candidate limit cannot be smaller than final limit",
    ):
        await service.retrieve(
            user_id=42,
            document_ids=[1],
            queries=["query"],
            limit=5,
            candidate_limit=4,
        )

    documents.list_by_ids.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rrf_k",
    [
        0,
        -1,
        -100,
    ],
)
async def test_retrieve_rejects_non_positive_rrf_rank_constant(
    rrf_k: int,
) -> None:
    """Reject invalid Reciprocal Rank Fusion constants."""
    service, _, documents, _ = _build_service()

    with pytest.raises(
        ValueError,
        match="RRF rank constant must be greater than zero",
    ):
        await service.retrieve(
            user_id=42,
            document_ids=[1],
            queries=["query"],
            rrf_k=rrf_k,
        )

    documents.list_by_ids.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_uses_default_candidate_limit_of_ten() -> None:
    """Retrieve at least ten candidates per query by default."""
    service, chunks, documents, embedding = _build_service()

    document = _build_document(document_id=1)

    documents.list_by_ids.return_value = [document]
    embedding.embed_query.return_value = [0.1, 0.2]
    chunks.search_similar.return_value = []

    await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=["query"],
        limit=5,
    )

    chunks.search_similar.assert_awaited_once_with(
        user_id=42,
        document_ids=[1],
        query_embedding=[0.1, 0.2],
        limit=10,
    )


@pytest.mark.asyncio
async def test_retrieve_default_candidate_limit_never_below_final_limit() -> None:
    """Raise the default candidate limit when final limit exceeds ten."""
    service, chunks, documents, embedding = _build_service()

    documents.list_by_ids.return_value = [_build_document(document_id=1)]
    embedding.embed_query.return_value = [0.1]
    chunks.search_similar.return_value = []

    await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=["query"],
        limit=15,
    )

    chunks.search_similar.assert_awaited_once_with(
        user_id=42,
        document_ids=[1],
        query_embedding=[0.1],
        limit=15,
    )


@pytest.mark.asyncio
async def test_retrieve_deduplicates_document_ids_before_validation() -> None:
    """Validate and search each selected document identifier only once."""
    service, chunks, documents, embedding = _build_service()

    documents.list_by_ids.return_value = [
        _build_document(document_id=1),
        _build_document(document_id=2),
    ]
    embedding.embed_query.return_value = [0.1]
    chunks.search_similar.return_value = []

    await service.retrieve(
        user_id=42,
        document_ids=[
            1,
            2,
            1,
            2,
        ],
        queries=["query"],
    )

    documents.list_by_ids.assert_awaited_once_with(
        document_ids=[1, 2],
        user_id=42,
    )

    chunks.search_similar.assert_awaited_once_with(
        user_id=42,
        document_ids=[1, 2],
        query_embedding=[0.1],
        limit=10,
    )


@pytest.mark.asyncio
async def test_retrieve_rejects_invalid_document_selection() -> None:
    """Reject selections containing documents unavailable to the user."""
    service, chunks, documents, embedding = _build_service()

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    with pytest.raises(
        ValueError,
        match="One or more selected documents are invalid",
    ):
        await service.retrieve(
            user_id=42,
            document_ids=[1, 2],
            queries=["query"],
        )

    embedding.embed_query.assert_not_awaited()
    chunks.search_similar.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_value",
    [
        "pending",
        "processing",
        "failed",
    ],
)
async def test_retrieve_rejects_document_not_ready(
    status_value: str,
) -> None:
    """Require every selected document to be fully ready for retrieval."""
    service, chunks, documents, embedding = _build_service()

    documents.list_by_ids.return_value = [
        _build_document(
            document_id=1,
            status=status_value,
        )
    ]

    with pytest.raises(
        ValueError,
        match="All selected documents must be ready",
    ):
        await service.retrieve(
            user_id=42,
            document_ids=[1],
            queries=["query"],
        )

    embedding.embed_query.assert_not_awaited()
    chunks.search_similar.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_dense_embeds_normalized_query() -> None:
    """Generate the vector embedding from the normalized retrieval query."""
    service, chunks, documents, embedding = _build_service()

    documents.list_by_ids.return_value = [_build_document(document_id=1)]
    embedding.embed_query.return_value = [0.5]
    chunks.search_similar.return_value = []

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=["  semantic query  "],
    )

    embedding.embed_query.assert_awaited_once_with("semantic query")

    assert result.retrieval_queries == ["semantic query"]


@pytest.mark.asyncio
async def test_retrieve_dense_searches_selected_documents() -> None:
    """Delegate dense vector search with user scope and query embedding."""
    service, chunks, documents, embedding = _build_service()

    documents.list_by_ids.return_value = [
        _build_document(document_id=1),
        _build_document(document_id=2),
    ]

    embedding.embed_query.return_value = [
        0.1,
        0.2,
        0.3,
    ]

    chunks.search_similar.return_value = []

    await service.retrieve(
        user_id=99,
        document_ids=[1, 2],
        queries=["query"],
        candidate_limit=20,
    )

    chunks.search_similar.assert_awaited_once_with(
        user_id=99,
        document_ids=[1, 2],
        query_embedding=[
            0.1,
            0.2,
            0.3,
        ],
        limit=20,
    )


@pytest.mark.asyncio
async def test_retrieve_dense_converts_cosine_distance_to_similarity() -> None:
    """Convert repository cosine distance into dense similarity score."""
    service, chunks, documents, embedding = _build_service()

    document = _build_document(document_id=1)
    chunk = _build_chunk(
        chunk_id=10,
        document_id=1,
    )

    documents.list_by_ids.return_value = [document]
    embedding.embed_query.return_value = [0.1]

    chunks.search_similar.return_value = [
        (
            chunk,
            0.25,
        )
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=["query"],
    )

    assert len(result.hits) == 1

    hit = result.hits[0]

    assert hit.chunk is chunk
    assert hit.score == pytest.approx(0.75)
    assert hit.best_similarity == pytest.approx(0.75)
    assert hit.matched_query_indexes == (1,)


@pytest.mark.asyncio
async def test_retrieve_dense_preserves_repository_ranking() -> None:
    """Keep dense search result order returned by the vector repository."""
    service, chunks, documents, embedding = _build_service()

    first = _build_chunk(
        chunk_id=1,
        chunk_index=1,
    )
    second = _build_chunk(
        chunk_id=2,
        chunk_index=2,
    )
    third = _build_chunk(
        chunk_id=3,
        chunk_index=3,
    )

    documents.list_by_ids.return_value = [_build_document(document_id=1)]
    embedding.embed_query.return_value = [0.1]

    chunks.search_similar.return_value = [
        (first, 0.1),
        (second, 0.2),
        (third, 0.3),
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=["query"],
    )

    assert [hit.chunk for hit in result.hits] == [
        first,
        second,
        third,
    ]


@pytest.mark.asyncio
async def test_retrieve_dense_limits_final_hits() -> None:
    """Return only the configured number of final dense hits."""
    service, chunks, documents, embedding = _build_service()

    chunk_one = _build_chunk(chunk_id=1)
    chunk_two = _build_chunk(chunk_id=2)
    chunk_three = _build_chunk(chunk_id=3)

    documents.list_by_ids.return_value = [_build_document(document_id=1)]
    embedding.embed_query.return_value = [0.1]

    chunks.search_similar.return_value = [
        (chunk_one, 0.1),
        (chunk_two, 0.2),
        (chunk_three, 0.3),
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=["query"],
        limit=2,
        candidate_limit=3,
    )

    assert [hit.chunk for hit in result.hits] == [
        chunk_one,
        chunk_two,
    ]


@pytest.mark.asyncio
async def test_retrieve_dense_returns_expected_result_metadata() -> None:
    """Return validated documents, normalized queries, and dense strategy."""
    service, chunks, documents, embedding = _build_service()

    document = _build_document(document_id=1)

    documents.list_by_ids.return_value = [document]
    embedding.embed_query.return_value = [0.1]
    chunks.search_similar.return_value = []

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[" Query "],
    )

    assert isinstance(
        result,
        RetrievalResult,
    )
    assert result.documents == [document]
    assert result.retrieval_queries == ["Query"]
    assert result.strategy == "dense"
    assert result.hits == []


@pytest.mark.asyncio
async def test_retrieve_multi_query_embeds_and_searches_each_query() -> None:
    """Execute an independent embedding and vector search for each query."""
    service, chunks, documents, embedding = _build_service()

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    embedding.embed_query.side_effect = [
        [0.1],
        [0.2],
    ]

    chunks.search_similar.side_effect = [
        [],
        [],
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[
            "query one",
            "query two",
        ],
    )

    assert embedding.embed_query.await_args_list == [
        call("query one"),
        call("query two"),
    ]

    assert chunks.search_similar.await_count == 2
    assert result.strategy == "multi_query_rrf"


@pytest.mark.asyncio
async def test_retrieve_deduplicates_queries_before_searching() -> None:
    """Avoid repeated vector searches for case-insensitive duplicate queries."""
    service, chunks, documents, embedding = _build_service()

    documents.list_by_ids.return_value = [_build_document(document_id=1)]
    embedding.embed_query.return_value = [0.1]
    chunks.search_similar.return_value = []

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[
            "Sherlock Holmes",
            "sherlock holmes",
        ],
    )

    embedding.embed_query.assert_awaited_once_with("Sherlock Holmes")
    chunks.search_similar.assert_awaited_once()

    assert result.strategy == "dense"
    assert result.retrieval_queries == ["Sherlock Holmes"]


@pytest.mark.asyncio
async def test_retrieve_rrf_accumulates_score_across_queries() -> None:
    """Accumulate Reciprocal Rank Fusion contributions for repeated candidates."""
    service, chunks, documents, embedding = _build_service()

    shared = _build_chunk(
        chunk_id=1,
    )

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    embedding.embed_query.side_effect = [
        [0.1],
        [0.2],
    ]

    chunks.search_similar.side_effect = [
        [
            (
                shared,
                0.1,
            )
        ],
        [
            (
                shared,
                0.2,
            )
        ],
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[
            "query one",
            "query two",
        ],
        rrf_k=60,
    )

    expected_score = 1.0 / 61 + 1.0 / 61

    assert len(result.hits) == 1
    assert result.hits[0].score == pytest.approx(expected_score)


@pytest.mark.asyncio
async def test_retrieve_rrf_tracks_best_dense_similarity() -> None:
    """Keep the highest dense similarity observed for each fused candidate."""
    service, chunks, documents, embedding = _build_service()

    shared = _build_chunk(
        chunk_id=1,
    )

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    embedding.embed_query.side_effect = [
        [0.1],
        [0.2],
    ]

    chunks.search_similar.side_effect = [
        [
            (
                shared,
                0.40,
            )
        ],
        [
            (
                shared,
                0.10,
            )
        ],
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[
            "query one",
            "query two",
        ],
    )

    assert result.hits[0].best_similarity == pytest.approx(0.90)


@pytest.mark.asyncio
async def test_retrieve_rrf_tracks_matching_query_indexes() -> None:
    """Record every one-based retrieval query that matched a fused chunk."""
    service, chunks, documents, embedding = _build_service()

    shared = _build_chunk(
        chunk_id=1,
    )

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    embedding.embed_query.side_effect = [
        [0.1],
        [0.2],
        [0.3],
    ]

    chunks.search_similar.side_effect = [
        [
            (
                shared,
                0.1,
            )
        ],
        [],
        [
            (
                shared,
                0.2,
            )
        ],
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[
            "query one",
            "query two",
            "query three",
        ],
    )

    assert result.hits[0].matched_query_indexes == (
        1,
        3,
    )


@pytest.mark.asyncio
async def test_retrieve_rrf_ranks_candidate_with_higher_fused_score_first() -> None:
    """Rank candidates by descending accumulated Reciprocal Rank Fusion score."""
    service, chunks, documents, embedding = _build_service()

    shared = _build_chunk(
        chunk_id=1,
    )
    single_query = _build_chunk(
        chunk_id=2,
    )

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    embedding.embed_query.side_effect = [
        [0.1],
        [0.2],
    ]

    chunks.search_similar.side_effect = [
        [
            (single_query, 0.01),
            (shared, 0.02),
        ],
        [
            (shared, 0.50),
        ],
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[
            "query one",
            "query two",
        ],
    )

    assert result.hits[0].chunk is shared
    assert result.hits[1].chunk is single_query


@pytest.mark.asyncio
async def test_retrieve_rrf_uses_best_similarity_as_secondary_sort_key() -> None:
    """Break equal RRF scores using the strongest dense similarity."""
    service, chunks, documents, embedding = _build_service()

    lower_similarity = _build_chunk(
        chunk_id=1,
    )
    higher_similarity = _build_chunk(
        chunk_id=2,
    )

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    embedding.embed_query.side_effect = [
        [0.1],
        [0.2],
    ]

    chunks.search_similar.side_effect = [
        [
            (
                lower_similarity,
                0.40,
            )
        ],
        [
            (
                higher_similarity,
                0.10,
            )
        ],
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[
            "query one",
            "query two",
        ],
    )

    assert [hit.chunk for hit in result.hits] == [
        higher_similarity,
        lower_similarity,
    ]


@pytest.mark.asyncio
async def test_retrieve_rrf_uses_chunk_id_as_final_tie_breaker() -> None:
    """Break complete ranking ties deterministically by ascending chunk id."""
    service, chunks, documents, embedding = _build_service()

    higher_id = _build_chunk(
        chunk_id=20,
    )
    lower_id = _build_chunk(
        chunk_id=10,
    )

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    embedding.embed_query.side_effect = [
        [0.1],
        [0.2],
    ]

    chunks.search_similar.side_effect = [
        [
            (
                higher_id,
                0.20,
            )
        ],
        [
            (
                lower_id,
                0.20,
            )
        ],
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[
            "query one",
            "query two",
        ],
    )

    assert [hit.chunk.id for hit in result.hits] == [
        10,
        20,
    ]


@pytest.mark.asyncio
async def test_retrieve_rrf_limits_final_hits() -> None:
    """Apply the final hit limit after multi-query fusion and sorting."""
    service, chunks, documents, embedding = _build_service()

    first = _build_chunk(chunk_id=1)
    second = _build_chunk(chunk_id=2)
    third = _build_chunk(chunk_id=3)

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    embedding.embed_query.side_effect = [
        [0.1],
        [0.2],
    ]

    chunks.search_similar.side_effect = [
        [
            (first, 0.1),
            (second, 0.2),
            (third, 0.3),
        ],
        [],
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[
            "query one",
            "query two",
        ],
        limit=2,
    )

    assert len(result.hits) == 2


@pytest.mark.asyncio
async def test_retrieve_multi_query_returns_expected_metadata() -> None:
    """Return validated documents, normalized queries, and RRF strategy."""
    service, chunks, documents, embedding = _build_service()

    document = _build_document(document_id=1)

    documents.list_by_ids.return_value = [document]

    embedding.embed_query.side_effect = [
        [0.1],
        [0.2],
    ]

    chunks.search_similar.side_effect = [
        [],
        [],
    ]

    result = await service.retrieve(
        user_id=42,
        document_ids=[1],
        queries=[
            " query one ",
            " query two ",
        ],
    )

    assert result.documents == [document]

    assert result.retrieval_queries == [
        "query one",
        "query two",
    ]

    assert result.strategy == "multi_query_rrf"
    assert result.hits == []


@pytest.mark.asyncio
async def test_retrieve_propagates_document_repository_failure() -> None:
    """Propagate document-validation storage failures unchanged."""
    service, _, documents, _ = _build_service()

    repository_error = RuntimeError("document database unavailable")

    documents.list_by_ids.side_effect = repository_error

    with pytest.raises(
        RuntimeError,
        match="document database unavailable",
    ) as exc_info:
        await service.retrieve(
            user_id=42,
            document_ids=[1],
            queries=["query"],
        )

    assert exc_info.value is repository_error


@pytest.mark.asyncio
async def test_retrieve_propagates_embedding_failure() -> None:
    """Propagate query-embedding failures unchanged."""
    service, chunks, documents, embedding = _build_service()

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    embedding_error = RuntimeError("embedding provider unavailable")

    embedding.embed_query.side_effect = embedding_error

    with pytest.raises(
        RuntimeError,
        match="embedding provider unavailable",
    ) as exc_info:
        await service.retrieve(
            user_id=42,
            document_ids=[1],
            queries=["query"],
        )

    assert exc_info.value is embedding_error
    chunks.search_similar.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_propagates_chunk_repository_failure() -> None:
    """Propagate vector-search storage failures unchanged."""
    service, chunks, documents, embedding = _build_service()

    documents.list_by_ids.return_value = [_build_document(document_id=1)]

    embedding.embed_query.return_value = [0.1]

    repository_error = RuntimeError("vector database unavailable")

    chunks.search_similar.side_effect = repository_error

    with pytest.raises(
        RuntimeError,
        match="vector database unavailable",
    ) as exc_info:
        await service.retrieve(
            user_id=42,
            document_ids=[1],
            queries=["query"],
        )

    assert exc_info.value is repository_error
