"""
Tests for end-to-end RAG service orchestration and result metadata.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from document_rag_service.models.document import Document
from document_rag_service.models.document_chunk import DocumentChunk
from document_rag_service.services.context_builder_service import (
    ContextBuildResult,
)
from document_rag_service.services.query_transformation_service import (
    QueryTransformationResult,
)
from document_rag_service.services.rag_service import (
    RagContextChunk,
    RagResult,
    RagRetrievalHit,
    RagService,
)
from document_rag_service.services.retrieval_service import (
    RetrievalHit,
    RetrievalResult,
)


def _build_document(
    *,
    document_id: int,
    title: str,
    user_id: int = 42,
) -> Document:
    """Create an in-memory ready document for RAG tests."""
    document = Document(
        user_id=user_id,
        title=title,
        source_type="upload",
        status="ready",
    )
    document.id = document_id
    return document


def _build_chunk(
    *,
    chunk_id: int,
    document_id: int,
    chunk_index: int,
    content: str | None = None,
) -> DocumentChunk:
    """Create an in-memory document chunk for RAG tests."""
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content or f"document {document_id} chunk {chunk_index}",
        token_count=10,
        embedding=[0.0] * 512,
    )
    chunk.id = chunk_id
    return chunk


def _build_hit(
    chunk: DocumentChunk,
    *,
    score: float = 0.9,
    best_similarity: float = 0.9,
    matched_query_indexes: tuple[int, ...] = (1,),
) -> RetrievalHit:
    """Create a retrieval hit for RAG orchestration tests."""
    return RetrievalHit(
        chunk=chunk,
        score=score,
        best_similarity=best_similarity,
        matched_query_indexes=matched_query_indexes,
    )


def _build_service(
    *,
    with_query_transformation: bool = True,
) -> tuple[
    RagService,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock | None,
]:
    """Create a RAG service with isolated dependency doubles."""
    retrieval = MagicMock()
    retrieval.retrieve = AsyncMock()

    context_expansion = MagicMock()
    context_expansion.expand = AsyncMock()

    context_builder = MagicMock()
    context_builder.build_result = MagicMock()

    generation = MagicMock()
    generation.get_available_context_characters = MagicMock()
    generation.generate = AsyncMock()

    query_transformation: MagicMock | None = None

    if with_query_transformation:
        query_transformation = MagicMock()
        query_transformation.transform = AsyncMock()

    service = RagService(
        retrieval_service=retrieval,
        context_expansion_service=context_expansion,
        context_builder=context_builder,
        generation_service=generation,
        query_transformation_service=query_transformation,
    )

    return (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        query_transformation,
    )


def _configure_successful_pipeline(
    *,
    retrieval: MagicMock,
    context_expansion: MagicMock,
    context_builder: MagicMock,
    generation: MagicMock,
) -> tuple[
    Document,
    DocumentChunk,
]:
    """Configure a minimal successful RAG pipeline."""
    document = _build_document(
        document_id=1,
        title="Test document",
    )

    chunk = _build_chunk(
        chunk_id=10,
        document_id=1,
        chunk_index=3,
        content="Relevant context",
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(chunk),
        ],
        documents=[
            document,
        ],
        retrieval_queries=[
            "What happened?",
        ],
        strategy="dense",
    )

    context_expansion.expand.return_value = [
        chunk,
    ]

    generation.get_available_context_characters.return_value = 500

    context_builder.build_result.return_value = ContextBuildResult(
        context="formatted context",
        included_chunks=(chunk,),
    )

    generation.generate.return_value = "Grounded answer"

    return document, chunk


@pytest.mark.asyncio
async def test_answer_uses_original_query_when_transformation_is_disabled() -> None:
    """Use the original user question directly for standard retrieval."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        query_transformation,
    ) = _build_service()

    _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    await service.answer(
        user_id=42,
        document_ids=[1],
        query="What happened?",
        language="English",
        transform_query=False,
    )

    retrieval.retrieve.assert_awaited_once_with(
        user_id=42,
        document_ids=[1],
        queries=["What happened?"],
        limit=5,
    )

    assert query_transformation is not None
    query_transformation.transform.assert_not_awaited()


@pytest.mark.asyncio
async def test_answer_rejects_requested_transformation_without_service() -> None:
    """Reject query transformation when no transformer is configured."""
    (
        service,
        retrieval,
        context_expansion,
        _,
        generation,
        _,
    ) = _build_service(
        with_query_transformation=False,
    )

    with pytest.raises(
        RuntimeError,
        match=("Query transformation requested but service is not configured"),
    ):
        await service.answer(
            user_id=42,
            document_ids=[1],
            query="Question",
            language="English",
            transform_query=True,
        )

    retrieval.retrieve.assert_not_awaited()
    context_expansion.expand.assert_not_awaited()
    generation.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_answer_uses_transformed_retrieval_queries() -> None:
    """Use transformed queries for retrieval while preserving the original question."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        query_transformation,
    ) = _build_service()

    assert query_transformation is not None

    query_transformation.transform.return_value = QueryTransformationResult(
        original_query="What happened?",
        retrieval_queries=[
            "What happened?",
            "focused retrieval query",
        ],
        used_fallback=False,
        attempt_count=1,
    )

    _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    await service.answer(
        user_id=42,
        document_ids=[1],
        query="What happened?",
        language="English",
        limit=7,
        transform_query=True,
    )

    query_transformation.transform.assert_awaited_once_with("What happened?")

    retrieval.retrieve.assert_awaited_once_with(
        user_id=42,
        document_ids=[1],
        queries=[
            "What happened?",
            "focused retrieval query",
        ],
        limit=7,
    )


@pytest.mark.asyncio
async def test_answer_expands_retrieved_anchor_chunks() -> None:
    """Expand the chunks selected directly by semantic retrieval."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    _, chunk = _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    await service.answer(
        user_id=99,
        document_ids=[1],
        query="Question",
        language="English",
        neighbor_window=2,
    )

    context_expansion.expand.assert_awaited_once_with(
        user_id=99,
        retrieved_chunks=[
            chunk,
        ],
        neighbor_window=2,
    )


@pytest.mark.asyncio
async def test_answer_requests_context_budget_for_original_question() -> None:
    """Calculate available context using the original question and output language."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    await service.answer(
        user_id=42,
        document_ids=[1],
        query="Original question",
        language="Portuguese",
    )

    generation.get_available_context_characters.assert_called_once_with(
        query="Original question",
        language="Portuguese",
    )


@pytest.mark.asyncio
async def test_answer_builds_context_using_available_character_budget() -> None:
    """Pass expanded chunks and generation budget to the context builder."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    _, chunk = _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    generation.get_available_context_characters.return_value = 1234

    await service.answer(
        user_id=42,
        document_ids=[1],
        query="Question",
        language="English",
    )

    context_builder.build_result.assert_called_once_with(
        [
            chunk,
        ],
        max_characters=1234,
    )


@pytest.mark.asyncio
async def test_answer_generates_from_original_query_and_built_context() -> None:
    """Generate the final answer from the original question and bounded context."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    context_builder.build_result.return_value = ContextBuildResult(
        context="final bounded context",
        included_chunks=context_builder.build_result.return_value.included_chunks,
    )

    await service.answer(
        user_id=42,
        document_ids=[1],
        query="Original question",
        language="Portuguese",
    )

    generation.generate.assert_awaited_once_with(
        query="Original question",
        context="final bounded context",
        language="Portuguese",
    )


@pytest.mark.asyncio
async def test_answer_returns_generated_answer() -> None:
    """Expose the exact grounded answer returned by generation."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    generation.generate.return_value = "Final grounded answer"

    result = await service.answer(
        user_id=42,
        document_ids=[1],
        query="Question",
        language="English",
    )

    assert isinstance(
        result,
        RagResult,
    )
    assert result.answer == "Final grounded answer"


@pytest.mark.asyncio
async def test_answer_maps_retrieval_hit_metadata() -> None:
    """Expose retrieval ranking diagnostics with source document metadata."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=1,
        title="Sherlock Holmes",
    )

    chunk = _build_chunk(
        chunk_id=10,
        document_id=1,
        chunk_index=21,
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            RetrievalHit(
                chunk=chunk,
                score=0.75,
                best_similarity=0.91,
                matched_query_indexes=(
                    1,
                    3,
                ),
            )
        ],
        documents=[
            document,
        ],
        retrieval_queries=[
            "original",
            "focused one",
            "focused two",
        ],
        strategy="multi_query_rrf",
    )

    context_expansion.expand.return_value = [
        chunk,
    ]

    generation.get_available_context_characters.return_value = 500

    context_builder.build_result.return_value = ContextBuildResult(
        context="context",
        included_chunks=(chunk,),
    )

    generation.generate.return_value = "answer"

    result = await service.answer(
        user_id=42,
        document_ids=[1],
        query="Question",
        language="English",
    )

    assert result.retrieval_hits == [
        RagRetrievalHit(
            document_id=1,
            document_title="Sherlock Holmes",
            chunk_index=21,
            score=0.75,
            best_similarity=0.91,
            matched_query_indexes=(
                1,
                3,
            ),
        )
    ]

    assert result.retrieval_strategy == "multi_query_rrf"

    assert result.retrieval_queries == [
        "original",
        "focused one",
        "focused two",
    ]


@pytest.mark.asyncio
async def test_answer_marks_retrieved_chunks_as_anchors() -> None:
    """Identify directly retrieved locations as anchor context chunks."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=1,
        title="Document",
    )

    anchor = _build_chunk(
        chunk_id=10,
        document_id=1,
        chunk_index=5,
    )

    neighbor = _build_chunk(
        chunk_id=11,
        document_id=1,
        chunk_index=6,
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(anchor),
        ],
        documents=[
            document,
        ],
        retrieval_queries=[
            "query",
        ],
        strategy="dense",
    )

    context_expansion.expand.return_value = [
        anchor,
        neighbor,
    ]

    generation.get_available_context_characters.return_value = 500

    context_builder.build_result.return_value = ContextBuildResult(
        context="context",
        included_chunks=(
            anchor,
            neighbor,
        ),
    )

    generation.generate.return_value = "answer"

    result = await service.answer(
        user_id=42,
        document_ids=[1],
        query="Question",
        language="English",
    )

    assert result.context_chunks == [
        RagContextChunk(
            document_id=1,
            document_title="Document",
            chunk_index=5,
            role="anchor",
        ),
        RagContextChunk(
            document_id=1,
            document_title="Document",
            chunk_index=6,
            role="neighbor",
        ),
    ]


@pytest.mark.asyncio
async def test_answer_detects_anchor_by_document_and_chunk_location() -> None:
    """Classify matching locations as anchors even when objects differ."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=1,
        title="Document",
    )

    retrieved_anchor = _build_chunk(
        chunk_id=10,
        document_id=1,
        chunk_index=5,
    )

    expanded_copy = _build_chunk(
        chunk_id=999,
        document_id=1,
        chunk_index=5,
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(retrieved_anchor),
        ],
        documents=[
            document,
        ],
        retrieval_queries=[
            "query",
        ],
        strategy="dense",
    )

    context_expansion.expand.return_value = [
        expanded_copy,
    ]

    generation.get_available_context_characters.return_value = 500

    context_builder.build_result.return_value = ContextBuildResult(
        context="context",
        included_chunks=(expanded_copy,),
    )

    generation.generate.return_value = "answer"

    result = await service.answer(
        user_id=42,
        document_ids=[1],
        query="Question",
        language="English",
    )

    assert result.context_chunks[0].role == "anchor"


@pytest.mark.asyncio
async def test_answer_exposes_all_expanded_context_candidates() -> None:
    """Expose expanded candidates even when the prompt budget excludes some."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=1,
        title="Document",
    )

    first = _build_chunk(
        chunk_id=1,
        document_id=1,
        chunk_index=1,
    )
    second = _build_chunk(
        chunk_id=2,
        document_id=1,
        chunk_index=2,
    )
    third = _build_chunk(
        chunk_id=3,
        document_id=1,
        chunk_index=3,
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(second),
        ],
        documents=[
            document,
        ],
        retrieval_queries=[
            "query",
        ],
        strategy="dense",
    )

    context_expansion.expand.return_value = [
        first,
        second,
        third,
    ]

    generation.get_available_context_characters.return_value = 100

    context_builder.build_result.return_value = ContextBuildResult(
        context="bounded context",
        included_chunks=(
            first,
            second,
        ),
    )

    generation.generate.return_value = "answer"

    result = await service.answer(
        user_id=42,
        document_ids=[1],
        query="Question",
        language="English",
    )

    assert len(result.context_chunks) == 3
    assert result.context_chunks_count == 2


@pytest.mark.asyncio
async def test_answer_sources_include_only_documents_used_in_final_context() -> None:
    """Report sources only for chunks that actually fit the prompt budget."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    first_document = _build_document(
        document_id=1,
        title="First document",
    )
    second_document = _build_document(
        document_id=2,
        title="Second document",
    )

    first_chunk = _build_chunk(
        chunk_id=1,
        document_id=1,
        chunk_index=0,
    )
    second_chunk = _build_chunk(
        chunk_id=2,
        document_id=2,
        chunk_index=0,
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(first_chunk),
            _build_hit(second_chunk),
        ],
        documents=[
            first_document,
            second_document,
        ],
        retrieval_queries=[
            "query",
        ],
        strategy="dense",
    )

    context_expansion.expand.return_value = [
        first_chunk,
        second_chunk,
    ]

    generation.get_available_context_characters.return_value = 100

    context_builder.build_result.return_value = ContextBuildResult(
        context="first document only",
        included_chunks=(first_chunk,),
    )

    generation.generate.return_value = "answer"

    result = await service.answer(
        user_id=42,
        document_ids=[
            1,
            2,
        ],
        query="Question",
        language="English",
    )

    assert result.sources == [
        "First document",
    ]


@pytest.mark.asyncio
async def test_answer_deduplicates_sources_preserving_context_order() -> None:
    """List each source document once in final included-context order."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    first_document = _build_document(
        document_id=1,
        title="First document",
    )
    second_document = _build_document(
        document_id=2,
        title="Second document",
    )

    first_document_chunk_one = _build_chunk(
        chunk_id=1,
        document_id=1,
        chunk_index=1,
    )
    first_document_chunk_two = _build_chunk(
        chunk_id=2,
        document_id=1,
        chunk_index=2,
    )
    second_document_chunk = _build_chunk(
        chunk_id=3,
        document_id=2,
        chunk_index=1,
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(first_document_chunk_one),
        ],
        documents=[
            first_document,
            second_document,
        ],
        retrieval_queries=[
            "query",
        ],
        strategy="dense",
    )

    context_expansion.expand.return_value = [
        first_document_chunk_one,
        first_document_chunk_two,
        second_document_chunk,
    ]

    generation.get_available_context_characters.return_value = 500

    context_builder.build_result.return_value = ContextBuildResult(
        context="context",
        included_chunks=(
            first_document_chunk_one,
            first_document_chunk_two,
            second_document_chunk,
        ),
    )

    generation.generate.return_value = "answer"

    result = await service.answer(
        user_id=42,
        document_ids=[
            1,
            2,
        ],
        query="Question",
        language="English",
    )

    assert result.sources == [
        "First document",
        "Second document",
    ]


@pytest.mark.asyncio
async def test_answer_sources_follow_included_chunk_order() -> None:
    """Order sources according to their first appearance in final context."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    first_document = _build_document(
        document_id=1,
        title="First",
    )
    second_document = _build_document(
        document_id=2,
        title="Second",
    )

    first_chunk = _build_chunk(
        chunk_id=1,
        document_id=1,
        chunk_index=0,
    )
    second_chunk = _build_chunk(
        chunk_id=2,
        document_id=2,
        chunk_index=0,
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(first_chunk),
        ],
        documents=[
            first_document,
            second_document,
        ],
        retrieval_queries=[
            "query",
        ],
        strategy="dense",
    )

    context_expansion.expand.return_value = [
        first_chunk,
        second_chunk,
    ]

    generation.get_available_context_characters.return_value = 500

    context_builder.build_result.return_value = ContextBuildResult(
        context="context",
        included_chunks=(
            second_chunk,
            first_chunk,
        ),
    )

    generation.generate.return_value = "answer"

    result = await service.answer(
        user_id=42,
        document_ids=[
            1,
            2,
        ],
        query="Question",
        language="English",
    )

    assert result.sources == [
        "Second",
        "First",
    ]


@pytest.mark.asyncio
async def test_answer_reports_context_character_count() -> None:
    """Report the exact length of the final context sent to generation."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    context_builder.build_result.return_value = ContextBuildResult(
        context="1234567890",
        included_chunks=context_builder.build_result.return_value.included_chunks,
    )

    result = await service.answer(
        user_id=42,
        document_ids=[1],
        query="Question",
        language="English",
    )

    assert result.context_characters == 10


@pytest.mark.asyncio
async def test_answer_reports_maximum_context_character_budget() -> None:
    """Expose the generation context budget used during context assembly."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    generation.get_available_context_characters.return_value = 4321

    result = await service.answer(
        user_id=42,
        document_ids=[1],
        query="Question",
        language="English",
    )

    assert result.max_context_characters == 4321


@pytest.mark.asyncio
async def test_answer_reports_number_of_chunks_actually_included() -> None:
    """Count only context chunks that fit into the final generation context."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=1,
        title="Document",
    )

    chunks = [
        _build_chunk(
            chunk_id=index,
            document_id=1,
            chunk_index=index,
        )
        for index in range(1, 4)
    ]

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(chunks[1]),
        ],
        documents=[
            document,
        ],
        retrieval_queries=[
            "query",
        ],
        strategy="dense",
    )

    context_expansion.expand.return_value = chunks

    generation.get_available_context_characters.return_value = 500

    context_builder.build_result.return_value = ContextBuildResult(
        context="context",
        included_chunks=(
            chunks[0],
            chunks[1],
        ),
    )

    generation.generate.return_value = "answer"

    result = await service.answer(
        user_id=42,
        document_ids=[1],
        query="Question",
        language="English",
    )

    assert result.context_chunks_count == 2
    assert len(result.context_chunks) == 3


@pytest.mark.asyncio
async def test_answer_rejects_context_chunk_with_unknown_document() -> None:
    """Reject expanded context metadata referencing an unvalidated document."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    known_document = _build_document(
        document_id=1,
        title="Known",
    )

    anchor = _build_chunk(
        chunk_id=1,
        document_id=1,
        chunk_index=0,
    )

    unknown_chunk = _build_chunk(
        chunk_id=2,
        document_id=999,
        chunk_index=0,
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(anchor),
        ],
        documents=[
            known_document,
        ],
        retrieval_queries=[
            "query",
        ],
        strategy="dense",
    )

    context_expansion.expand.return_value = [
        anchor,
        unknown_chunk,
    ]

    generation.get_available_context_characters.return_value = 500

    context_builder.build_result.return_value = ContextBuildResult(
        context="context",
        included_chunks=(anchor,),
    )

    generation.generate.return_value = "answer"

    with pytest.raises(
        RuntimeError,
        match="Context chunk references an unknown document",
    ):
        await service.answer(
            user_id=42,
            document_ids=[1],
            query="Question",
            language="English",
        )


@pytest.mark.asyncio
async def test_answer_rejects_retrieval_hit_with_unknown_document() -> None:
    """Reject retrieval metadata referencing an unvalidated source document."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    known_document = _build_document(
        document_id=1,
        title="Known",
    )

    known_context_chunk = _build_chunk(
        chunk_id=1,
        document_id=1,
        chunk_index=0,
    )

    unknown_retrieved_chunk = _build_chunk(
        chunk_id=2,
        document_id=999,
        chunk_index=0,
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(unknown_retrieved_chunk),
        ],
        documents=[
            known_document,
        ],
        retrieval_queries=[
            "query",
        ],
        strategy="dense",
    )

    context_expansion.expand.return_value = [
        known_context_chunk,
    ]

    generation.get_available_context_characters.return_value = 500

    context_builder.build_result.return_value = ContextBuildResult(
        context="context",
        included_chunks=(known_context_chunk,),
    )

    generation.generate.return_value = "answer"

    with pytest.raises(
        RuntimeError,
        match="Retrieved chunk references an unknown document",
    ):
        await service.answer(
            user_id=42,
            document_ids=[1],
            query="Question",
            language="English",
        )


@pytest.mark.asyncio
async def test_answer_propagates_retrieval_failure() -> None:
    """Propagate retrieval failures without continuing the RAG pipeline."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    retrieval_error = ValueError("All selected documents must be ready")

    retrieval.retrieve.side_effect = retrieval_error

    with pytest.raises(
        ValueError,
        match="All selected documents must be ready",
    ) as exc_info:
        await service.answer(
            user_id=42,
            document_ids=[1],
            query="Question",
            language="English",
        )

    assert exc_info.value is retrieval_error

    context_expansion.expand.assert_not_awaited()
    context_builder.build_result.assert_not_called()
    generation.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_answer_propagates_context_expansion_failure() -> None:
    """Propagate expansion failures before context construction or generation."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=1,
        title="Document",
    )
    chunk = _build_chunk(
        chunk_id=1,
        document_id=1,
        chunk_index=0,
    )

    retrieval.retrieve.return_value = RetrievalResult(
        hits=[
            _build_hit(chunk),
        ],
        documents=[
            document,
        ],
        retrieval_queries=[
            "query",
        ],
        strategy="dense",
    )

    expansion_error = RuntimeError("context expansion failed")

    context_expansion.expand.side_effect = expansion_error

    with pytest.raises(
        RuntimeError,
        match="context expansion failed",
    ) as exc_info:
        await service.answer(
            user_id=42,
            document_ids=[1],
            query="Question",
            language="English",
        )

    assert exc_info.value is expansion_error

    context_builder.build_result.assert_not_called()
    generation.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_answer_propagates_context_builder_failure() -> None:
    """Propagate prompt-budget context construction failures."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    context_error = ValueError(
        "No complete context chunk fits within the context budget"
    )

    context_builder.build_result.side_effect = context_error

    with pytest.raises(
        ValueError,
        match="No complete context chunk fits within the context budget",
    ) as exc_info:
        await service.answer(
            user_id=42,
            document_ids=[1],
            query="Question",
            language="English",
        )

    assert exc_info.value is context_error
    generation.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_answer_propagates_generation_failure() -> None:
    """Propagate LLM generation failures unchanged."""
    (
        service,
        retrieval,
        context_expansion,
        context_builder,
        generation,
        _,
    ) = _build_service()

    _configure_successful_pipeline(
        retrieval=retrieval,
        context_expansion=context_expansion,
        context_builder=context_builder,
        generation=generation,
    )

    generation_error = RuntimeError("LLM provider unavailable")

    generation.generate.side_effect = generation_error

    with pytest.raises(
        RuntimeError,
        match="LLM provider unavailable",
    ) as exc_info:
        await service.answer(
            user_id=42,
            document_ids=[1],
            query="Question",
            language="English",
        )

    assert exc_info.value is generation_error
