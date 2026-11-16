"""
Integration tests for RetrievalService against PostgreSQL and pgvector.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.user_repository import UserRepository
from document_rag_service.services.embedding_service import (
    EmbeddingProvider,
    EmbeddingService,
)
from document_rag_service.services.retrieval_service import RetrievalService

EMBEDDING_DIMENSION = 512


def _vector(
    first: float,
    second: float,
) -> list[float]:
    """Build a deterministic production-dimension embedding."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[0] = first
    vector[1] = second
    return vector


class FakeRetrievalEmbeddingProvider:
    """Provide deterministic query embeddings for retrieval integration tests."""

    @property
    def model_name(self) -> str:
        """Return a stable fake model name."""
        return "fake-retrieval-integration-model"

    @property
    def dimension(self) -> int:
        """Return the production embedding dimension."""
        return EMBEDDING_DIMENSION

    @property
    def max_sequence_length(self) -> int:
        """Return a realistic maximum sequence length."""
        return 8192

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Return deterministic document embeddings."""
        return [_vector(1.0, 0.0) for _ in texts]

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Map known queries to deterministic search vectors."""
        normalized = text.strip().casefold()

        if normalized == "alpha":
            return _vector(1.0, 0.0)

        if normalized == "beta":
            return _vector(0.0, 1.0)

        return _vector(1.0, 0.0)

    def count_document_tokens(
        self,
        text: str,
    ) -> int:
        """Return a deterministic token count."""
        return len(text.split())

    async def aclose(self) -> None:
        """Release provider resources."""
        return None


async def _create_user(
    session: AsyncSession,
    *,
    email: str,
):
    """Create one persisted test user."""
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
    ready: bool = True,
) -> Document:
    """Create one document and optionally move it to ready state."""
    repository = DocumentRepository(session)

    document = await repository.create(
        user_id=user_id,
        title=title,
        file_name=f"{title}.pdf",
        mime_type="application/pdf",
    )

    if ready:
        await repository.update_status(
            document,
            "ready",
        )

    return document


async def _create_chunk(
    session: AsyncSession,
    *,
    document_id: int,
    chunk_index: int,
    content: str,
    embedding: list[float],
) -> None:
    """Create one persisted chunk with a real pgvector embedding."""
    repository = DocumentChunkRepository(session)

    await repository.create(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        embedding=embedding,
        token_count=len(content.split()),
    )


def _build_retrieval_service(
    session: AsyncSession,
    *,
    provider: EmbeddingProvider | None = None,
) -> RetrievalService:
    """Build RetrievalService with real repositories and fake embeddings."""
    embedding_service = EmbeddingService(provider or FakeRetrievalEmbeddingProvider())

    return RetrievalService(
        DocumentChunkRepository(session),
        DocumentRepository(session),
        embedding_service,
    )


@pytest.mark.asyncio
async def test_dense_retrieval_ranks_chunks_by_real_pgvector_similarity(
    db_session: AsyncSession,
) -> None:
    """Rank persisted chunks by cosine similarity in PostgreSQL."""
    user = await _create_user(
        db_session,
        email="retrieval-ranking@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Ranking document",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Exact alpha match",
        embedding=_vector(1.0, 0.0),
    )
    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=1,
        content="Partial alpha match",
        embedding=_vector(0.8, 0.6),
    )
    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=2,
        content="Orthogonal beta match",
        embedding=_vector(0.0, 1.0),
    )

    service = _build_retrieval_service(db_session)

    result = await service.retrieve(
        user_id=user.id,
        document_ids=[document.id],
        queries=["alpha"],
        limit=3,
        candidate_limit=3,
    )

    assert result.strategy == "dense"
    assert result.retrieval_queries == ["alpha"]
    assert len(result.hits) == 3

    assert [hit.chunk.chunk_index for hit in result.hits] == [
        0,
        1,
        2,
    ]

    assert result.hits[0].best_similarity > result.hits[1].best_similarity
    assert result.hits[1].best_similarity > result.hits[2].best_similarity

    assert all(hit.matched_query_indexes == (1,) for hit in result.hits)


@pytest.mark.asyncio
async def test_retrieval_only_searches_selected_documents(
    db_session: AsyncSession,
) -> None:
    """Exclude chunks from unselected documents owned by the same user."""
    user = await _create_user(
        db_session,
        email="retrieval-selected-documents@example.com",
    )

    selected_document = await _create_document(
        db_session,
        user_id=user.id,
        title="Selected",
    )
    unselected_document = await _create_document(
        db_session,
        user_id=user.id,
        title="Unselected",
    )

    await _create_chunk(
        db_session,
        document_id=selected_document.id,
        chunk_index=0,
        content="Selected document result",
        embedding=_vector(0.8, 0.6),
    )

    await _create_chunk(
        db_session,
        document_id=unselected_document.id,
        chunk_index=0,
        content="Better but unselected result",
        embedding=_vector(1.0, 0.0),
    )

    service = _build_retrieval_service(db_session)

    result = await service.retrieve(
        user_id=user.id,
        document_ids=[selected_document.id],
        queries=["alpha"],
        limit=5,
        candidate_limit=5,
    )

    assert len(result.hits) == 1
    assert result.hits[0].chunk.document_id == selected_document.id
    assert result.hits[0].chunk.content == "Selected document result"


@pytest.mark.asyncio
async def test_retrieval_rejects_document_owned_by_another_user(
    db_session: AsyncSession,
) -> None:
    """Prevent retrieval against another user's selected document."""
    owner = await _create_user(
        db_session,
        email="retrieval-owner@example.com",
    )
    other_user = await _create_user(
        db_session,
        email="retrieval-other@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=owner.id,
        title="Private retrieval document",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Private content",
        embedding=_vector(1.0, 0.0),
    )

    document_id = document.id
    other_user_id = other_user.id

    service = _build_retrieval_service(db_session)

    with pytest.raises(
        ValueError,
        match="One or more selected documents are invalid",
    ):
        await service.retrieve(
            user_id=other_user_id,
            document_ids=[document_id],
            queries=["alpha"],
            limit=5,
        )


@pytest.mark.asyncio
async def test_retrieval_rejects_document_that_is_not_ready(
    db_session: AsyncSession,
) -> None:
    """Require every selected document to be ready before retrieval."""
    user = await _create_user(
        db_session,
        email="retrieval-pending@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Pending retrieval document",
        ready=False,
    )

    service = _build_retrieval_service(db_session)

    with pytest.raises(
        ValueError,
        match="All selected documents must be ready",
    ):
        await service.retrieve(
            user_id=user.id,
            document_ids=[document.id],
            queries=["alpha"],
        )


@pytest.mark.asyncio
async def test_retrieval_respects_final_limit(
    db_session: AsyncSession,
) -> None:
    """Return no more than the requested number of final retrieval hits."""
    user = await _create_user(
        db_session,
        email="retrieval-limit@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Limit document",
    )

    embeddings = [
        _vector(1.0, 0.0),
        _vector(0.9, 0.1),
        _vector(0.8, 0.2),
        _vector(0.7, 0.3),
    ]

    for index, embedding in enumerate(embeddings):
        await _create_chunk(
            db_session,
            document_id=document.id,
            chunk_index=index,
            content=f"Chunk {index}",
            embedding=embedding,
        )

    service = _build_retrieval_service(db_session)

    result = await service.retrieve(
        user_id=user.id,
        document_ids=[document.id],
        queries=["alpha"],
        limit=2,
        candidate_limit=4,
    )

    assert len(result.hits) == 2

    assert [hit.chunk.chunk_index for hit in result.hits] == [
        0,
        1,
    ]


@pytest.mark.asyncio
async def test_multi_query_retrieval_uses_rrf_fusion(
    db_session: AsyncSession,
) -> None:
    """Fuse multiple pgvector rankings with reciprocal-rank fusion."""
    user = await _create_user(
        db_session,
        email="retrieval-rrf@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=user.id,
        title="RRF document",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Alpha-focused chunk",
        embedding=_vector(1.0, 0.0),
    )
    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=1,
        content="Balanced alpha beta chunk",
        embedding=_vector(0.70710678, 0.70710678),
    )
    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=2,
        content="Beta-focused chunk",
        embedding=_vector(0.0, 1.0),
    )

    service = _build_retrieval_service(db_session)

    result = await service.retrieve(
        user_id=user.id,
        document_ids=[document.id],
        queries=[
            "alpha",
            "beta",
        ],
        limit=3,
        candidate_limit=3,
        rrf_k=60,
    )

    assert result.strategy == "multi_query_rrf"
    assert result.retrieval_queries == [
        "alpha",
        "beta",
    ]

    assert len(result.hits) == 3

    assert {hit.chunk.chunk_index for hit in result.hits} == {
        0,
        1,
        2,
    }

    assert all(hit.matched_query_indexes == (1, 2) for hit in result.hits)

    assert all(hit.score > 0.0 for hit in result.hits)

    assert all(0.0 <= hit.best_similarity <= 1.0 for hit in result.hits)


@pytest.mark.asyncio
async def test_retrieval_deduplicates_selected_document_ids(
    db_session: AsyncSession,
) -> None:
    """Accept repeated selected ids without treating them as missing documents."""
    user = await _create_user(
        db_session,
        email="retrieval-duplicate-document-ids@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Duplicate selection document",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Retrievable content",
        embedding=_vector(1.0, 0.0),
    )

    service = _build_retrieval_service(db_session)

    result = await service.retrieve(
        user_id=user.id,
        document_ids=[
            document.id,
            document.id,
            document.id,
        ],
        queries=["alpha"],
        limit=5,
    )

    assert len(result.documents) == 1
    assert result.documents[0].id == document.id

    assert len(result.hits) == 1
    assert result.hits[0].chunk.document_id == document.id


@pytest.mark.asyncio
async def test_retrieval_normalizes_and_deduplicates_queries(
    db_session: AsyncSession,
) -> None:
    """Normalize repeated retrieval queries before searching pgvector."""
    user = await _create_user(
        db_session,
        email="retrieval-query-normalization@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Normalized queries",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Alpha content",
        embedding=_vector(1.0, 0.0),
    )

    service = _build_retrieval_service(db_session)

    result = await service.retrieve(
        user_id=user.id,
        document_ids=[document.id],
        queries=[
            " alpha ",
            "ALPHA",
            "Alpha",
        ],
        limit=5,
    )

    assert result.strategy == "dense"
    assert result.retrieval_queries == ["alpha"]
    assert len(result.hits) == 1
