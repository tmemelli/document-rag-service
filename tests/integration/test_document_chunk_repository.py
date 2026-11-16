"""
Integration tests for DocumentChunkRepository against PostgreSQL and pgvector.
"""

import math

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.models.document_chunk import DocumentChunk
from document_rag_service.models.user import User
from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.user_repository import (
    UserRepository,
)

EMBEDDING_DIMENSION = 512


def _embedding(
    *values: float,
) -> list[float]:
    """Build a deterministic embedding with the production vector dimension."""
    vector = [0.0] * EMBEDDING_DIMENSION

    for index, value in enumerate(values):
        vector[index] = value

    return vector


async def _create_user(
    session: AsyncSession,
    *,
    email: str,
) -> User:
    """Create a persisted integration-test user."""
    repository = UserRepository(session)

    return await repository.create(
        email=email,
        password_hash="test-password-hash",
    )


async def _create_document(
    session: AsyncSession,
    *,
    user_id: int,
    title: str,
) -> Document:
    """Create one persisted document owned by a test user."""
    repository = DocumentRepository(session)

    return await repository.create(
        user_id=user_id,
        title=title,
        file_name=f"{title}.pdf",
        mime_type="application/pdf",
    )


async def _create_chunk(
    session: AsyncSession,
    *,
    document_id: int,
    chunk_index: int,
    content: str,
    embedding: list[float] | None = None,
    token_count: int = 10,
) -> DocumentChunk:
    """Create one persisted document chunk."""
    repository = DocumentChunkRepository(session)

    return await repository.create(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        embedding=embedding or _embedding(1.0),
        token_count=token_count,
    )


@pytest.mark.asyncio
async def test_create_persists_chunk(
    db_session: AsyncSession,
) -> None:
    """Persist a complete document chunk."""
    user = await _create_user(
        db_session,
        email="chunk-create@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Chunk create",
    )

    repository = DocumentChunkRepository(db_session)

    chunk = await repository.create(
        document_id=document.id,
        chunk_index=0,
        content="First chunk",
        embedding=_embedding(1.0),
        token_count=42,
    )

    assert chunk.id > 0
    assert chunk.document_id == document.id
    assert chunk.chunk_index == 0
    assert chunk.content == "First chunk"
    assert chunk.token_count == 42
    assert chunk.embedding is not None
    assert len(chunk.embedding) == EMBEDDING_DIMENSION
    assert chunk.created_at is not None


@pytest.mark.asyncio
async def test_list_by_document_id_returns_chunks_in_index_order(
    db_session: AsyncSession,
) -> None:
    """Order chunks by their natural document position."""
    user = await _create_user(
        db_session,
        email="chunk-order@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Ordered chunks",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=2,
        content="Third",
    )
    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="First",
    )
    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=1,
        content="Second",
    )

    repository = DocumentChunkRepository(db_session)

    result = await repository.list_by_document_id(document.id)

    assert [chunk.chunk_index for chunk in result] == [0, 1, 2]


@pytest.mark.asyncio
async def test_list_by_document_id_excludes_other_documents(
    db_session: AsyncSession,
) -> None:
    """Return chunks only from the requested document."""
    user = await _create_user(
        db_session,
        email="chunk-document-filter@example.com",
    )
    first_document = await _create_document(
        db_session,
        user_id=user.id,
        title="First document",
    )
    second_document = await _create_document(
        db_session,
        user_id=user.id,
        title="Second document",
    )

    expected = await _create_chunk(
        db_session,
        document_id=first_document.id,
        chunk_index=0,
        content="Expected",
    )
    await _create_chunk(
        db_session,
        document_id=second_document.id,
        chunk_index=0,
        content="Foreign document",
    )

    repository = DocumentChunkRepository(db_session)

    result = await repository.list_by_document_id(first_document.id)

    assert [chunk.id for chunk in result] == [expected.id]


@pytest.mark.asyncio
async def test_list_by_document_id_returns_empty_list_when_missing(
    db_session: AsyncSession,
) -> None:
    """Return an empty collection for an unknown document."""
    repository = DocumentChunkRepository(db_session)

    result = await repository.list_by_document_id(9_999_999_999)

    assert result == []


@pytest.mark.asyncio
async def test_list_by_locations_returns_requested_chunks(
    db_session: AsyncSession,
) -> None:
    """Retrieve chunks by document and chunk-index coordinates."""
    user = await _create_user(
        db_session,
        email="locations@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Locations",
    )

    first = await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="First",
    )
    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=1,
        content="Second",
    )
    third = await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=2,
        content="Third",
    )

    repository = DocumentChunkRepository(db_session)

    result = await repository.list_by_locations(
        user_id=user.id,
        locations=[
            (document.id, 0),
            (document.id, 2),
        ],
    )

    assert {chunk.id for chunk in result} == {
        first.id,
        third.id,
    }


@pytest.mark.asyncio
async def test_list_by_locations_enforces_user_ownership(
    db_session: AsyncSession,
) -> None:
    """Prevent retrieving another user's chunks by explicit location."""
    owner = await _create_user(
        db_session,
        email="locations-owner@example.com",
    )
    other_user = await _create_user(
        db_session,
        email="locations-other@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=owner.id,
        title="Private locations",
    )
    chunk = await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Private",
    )

    repository = DocumentChunkRepository(db_session)

    result = await repository.list_by_locations(
        user_id=other_user.id,
        locations=[
            (document.id, chunk.chunk_index),
        ],
    )

    assert result == []


@pytest.mark.asyncio
async def test_list_by_locations_returns_empty_list_for_empty_locations(
    db_session: AsyncSession,
) -> None:
    """Return immediately when no chunk locations are requested."""
    user = await _create_user(
        db_session,
        email="locations-empty@example.com",
    )

    repository = DocumentChunkRepository(db_session)

    result = await repository.list_by_locations(
        user_id=user.id,
        locations=[],
    )

    assert result == []


@pytest.mark.asyncio
async def test_search_similar_orders_chunks_by_cosine_distance(
    db_session: AsyncSession,
) -> None:
    """Use pgvector cosine distance to rank chunks semantically."""
    user = await _create_user(
        db_session,
        email="vector-order@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Vector ordering",
    )

    exact = await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Exact",
        embedding=_embedding(1.0, 0.0),
    )
    diagonal = await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=1,
        content="Diagonal",
        embedding=_embedding(1.0, 1.0),
    )
    orthogonal = await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=2,
        content="Orthogonal",
        embedding=_embedding(0.0, 1.0),
    )

    repository = DocumentChunkRepository(db_session)

    result = await repository.search_similar(
        user_id=user.id,
        document_ids=[document.id],
        query_embedding=_embedding(1.0, 0.0),
        limit=3,
    )

    assert [chunk.id for chunk, _ in result] == [
        exact.id,
        diagonal.id,
        orthogonal.id,
    ]

    distances = [distance for _, distance in result]

    assert distances[0] == pytest.approx(0.0)
    assert distances[1] == pytest.approx(
        1.0 - (1.0 / math.sqrt(2.0)),
        abs=1e-6,
    )
    assert distances[2] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_search_similar_respects_limit(
    db_session: AsyncSession,
) -> None:
    """Limit the number of pgvector retrieval candidates."""
    user = await _create_user(
        db_session,
        email="vector-limit@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Vector limit",
    )

    for chunk_index in range(4):
        await _create_chunk(
            db_session,
            document_id=document.id,
            chunk_index=chunk_index,
            content=f"Chunk {chunk_index}",
            embedding=_embedding(
                1.0,
                float(chunk_index + 1) / 10,
            ),
        )

    repository = DocumentChunkRepository(db_session)

    result = await repository.search_similar(
        user_id=user.id,
        document_ids=[document.id],
        query_embedding=_embedding(1.0, 0.0),
        limit=2,
    )

    assert len(result) == 2


@pytest.mark.asyncio
async def test_search_similar_filters_selected_document_ids(
    db_session: AsyncSession,
) -> None:
    """Search only inside explicitly selected documents."""
    user = await _create_user(
        db_session,
        email="vector-documents@example.com",
    )
    selected_document = await _create_document(
        db_session,
        user_id=user.id,
        title="Selected vector document",
    )
    excluded_document = await _create_document(
        db_session,
        user_id=user.id,
        title="Excluded vector document",
    )

    selected_chunk = await _create_chunk(
        db_session,
        document_id=selected_document.id,
        chunk_index=0,
        content="Selected",
        embedding=_embedding(0.8, 0.2),
    )
    await _create_chunk(
        db_session,
        document_id=excluded_document.id,
        chunk_index=0,
        content="Excluded but exact",
        embedding=_embedding(1.0, 0.0),
    )

    repository = DocumentChunkRepository(db_session)

    result = await repository.search_similar(
        user_id=user.id,
        document_ids=[selected_document.id],
        query_embedding=_embedding(1.0, 0.0),
        limit=5,
    )

    assert [chunk.id for chunk, _ in result] == [selected_chunk.id]


@pytest.mark.asyncio
async def test_search_similar_enforces_user_ownership(
    db_session: AsyncSession,
) -> None:
    """Prevent another user's chunks from appearing in vector search."""
    owner = await _create_user(
        db_session,
        email="vector-owner@example.com",
    )
    other_user = await _create_user(
        db_session,
        email="vector-other@example.com",
    )

    owner_document = await _create_document(
        db_session,
        user_id=owner.id,
        title="Owner vector document",
    )
    other_document = await _create_document(
        db_session,
        user_id=other_user.id,
        title="Other vector document",
    )

    owner_chunk = await _create_chunk(
        db_session,
        document_id=owner_document.id,
        chunk_index=0,
        content="Owner",
        embedding=_embedding(0.8, 0.2),
    )
    await _create_chunk(
        db_session,
        document_id=other_document.id,
        chunk_index=0,
        content="Other exact match",
        embedding=_embedding(1.0, 0.0),
    )

    repository = DocumentChunkRepository(db_session)

    result = await repository.search_similar(
        user_id=owner.id,
        document_ids=[
            owner_document.id,
            other_document.id,
        ],
        query_embedding=_embedding(1.0, 0.0),
        limit=5,
    )

    assert [chunk.id for chunk, _ in result] == [owner_chunk.id]


@pytest.mark.asyncio
async def test_search_similar_returns_empty_list_for_empty_document_ids(
    db_session: AsyncSession,
) -> None:
    """Avoid vector search when no documents are selected."""
    user = await _create_user(
        db_session,
        email="vector-empty-documents@example.com",
    )

    repository = DocumentChunkRepository(db_session)

    result = await repository.search_similar(
        user_id=user.id,
        document_ids=[],
        query_embedding=_embedding(1.0),
        limit=5,
    )

    assert result == []


@pytest.mark.asyncio
async def test_database_rejects_duplicate_chunk_index_per_document(
    db_session: AsyncSession,
) -> None:
    """Enforce unique chunk positions within each document."""
    user = await _create_user(
        db_session,
        email="duplicate-index@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Duplicate index",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="First",
    )

    repository = DocumentChunkRepository(db_session)

    async with db_session.begin_nested():
        with pytest.raises(IntegrityError):
            await repository.create(
                document_id=document.id,
                chunk_index=0,
                content="Duplicate",
                embedding=_embedding(1.0),
            )


@pytest.mark.asyncio
async def test_database_rejects_negative_chunk_index(
    db_session: AsyncSession,
) -> None:
    """Enforce the database chunk-index check constraint."""
    user = await _create_user(
        db_session,
        email="negative-index@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Negative index",
    )

    repository = DocumentChunkRepository(db_session)

    async with db_session.begin_nested():
        with pytest.raises(IntegrityError):
            await repository.create(
                document_id=document.id,
                chunk_index=-1,
                content="Invalid",
                embedding=_embedding(1.0),
            )


@pytest.mark.asyncio
async def test_database_rejects_negative_token_count(
    db_session: AsyncSession,
) -> None:
    """Enforce the database token-count check constraint."""
    user = await _create_user(
        db_session,
        email="negative-tokens@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Negative tokens",
    )

    repository = DocumentChunkRepository(db_session)

    async with db_session.begin_nested():
        with pytest.raises(IntegrityError):
            await repository.create(
                document_id=document.id,
                chunk_index=0,
                content="Invalid tokens",
                embedding=_embedding(1.0),
                token_count=-1,
            )
