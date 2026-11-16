"""
Integration tests for RagService against PostgreSQL and pgvector.
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
from document_rag_service.services.context_builder_service import (
    ContextBuilderService,
)
from document_rag_service.services.context_expansion_service import (
    ContextExpansionService,
)
from document_rag_service.services.embedding_service import (
    EmbeddingProvider,
    EmbeddingService,
)
from document_rag_service.services.generation_service import GenerationService
from document_rag_service.services.query_transformation_service import (
    QueryTransformationService,
)
from document_rag_service.services.rag_service import RagService
from document_rag_service.services.retrieval_service import RetrievalService

EMBEDDING_DIMENSION = 512


def _vector(
    first: float,
    second: float,
) -> list[float]:
    """Build a deterministic production-dimension vector."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[0] = first
    vector[1] = second
    return vector


class FakeEmbeddingProvider:
    """Deterministic embedding provider for RAG integration tests."""

    @property
    def model_name(self) -> str:
        """Return a stable fake model name."""
        return "fake-rag-embedding-model"

    @property
    def dimension(self) -> int:
        """Return production embedding dimension."""
        return EMBEDDING_DIMENSION

    @property
    def max_sequence_length(self) -> int:
        """Return a realistic sequence limit."""
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
        """Map retrieval queries to deterministic vectors."""
        normalized = text.strip().casefold()

        if "beta" in normalized:
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


class FakeGenerationProvider:
    """Capture the final generation prompt without external network calls."""

    def __init__(self) -> None:
        self.system_prompt: str | None = None
        self.prompt: str | None = None

    async def generate(
        self,
        *,
        system_prompt: str,
        prompt: str,
    ) -> str:
        """Capture generation input and return a deterministic answer."""
        self.system_prompt = system_prompt
        self.prompt = prompt

        return "Grounded integration answer"


class FakeQueryTransformationProvider:
    """Return deterministic focused retrieval queries."""

    async def generate(
        self,
        *,
        system_prompt: str,
        prompt: str,
    ) -> str:
        """Return two valid retrieval queries."""
        return '{"queries": ["alpha evidence", "beta evidence"]}'


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


async def _create_ready_document(
    session: AsyncSession,
    *,
    user_id: int,
    title: str,
) -> Document:
    """Create one ready document."""
    repository = DocumentRepository(session)

    document = await repository.create(
        user_id=user_id,
        title=title,
        file_name=f"{title}.pdf",
        mime_type="application/pdf",
    )

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
    """Persist one real pgvector-backed chunk."""
    repository = DocumentChunkRepository(session)

    await repository.create(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        embedding=embedding,
        token_count=len(content.split()),
    )


def _build_rag_service(
    session: AsyncSession,
    *,
    generation_provider: FakeGenerationProvider,
    embedding_provider: EmbeddingProvider | None = None,
    query_transformation_service: QueryTransformationService | None = None,
) -> RagService:
    """Build the complete RAG service using real repositories."""
    embedding_service = EmbeddingService(embedding_provider or FakeEmbeddingProvider())

    chunk_repository = DocumentChunkRepository(session)
    document_repository = DocumentRepository(session)

    retrieval_service = RetrievalService(
        chunk_repository,
        document_repository,
        embedding_service,
    )

    context_expansion_service = ContextExpansionService(chunk_repository)

    context_builder = ContextBuilderService()

    generation_service = GenerationService(
        generation_provider,
        max_prompt_characters=20_000,
    )

    return RagService(
        retrieval_service,
        context_expansion_service,
        context_builder,
        generation_service,
        query_transformation_service=query_transformation_service,
    )


@pytest.mark.asyncio
async def test_rag_answer_runs_complete_dense_pipeline(
    db_session: AsyncSession,
) -> None:
    """Run retrieval, context expansion, context building and generation."""
    user = await _create_user(
        db_session,
        email="rag-dense@example.com",
    )

    document = await _create_ready_document(
        db_session,
        user_id=user.id,
        title="Dense RAG Document",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Neighbor before the relevant evidence.",
        embedding=_vector(0.0, 1.0),
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=1,
        content="Alpha evidence is the primary relevant fact.",
        embedding=_vector(1.0, 0.0),
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=2,
        content="Neighbor after the relevant evidence.",
        embedding=_vector(0.0, 1.0),
    )

    generation_provider = FakeGenerationProvider()

    service = _build_rag_service(
        db_session,
        generation_provider=generation_provider,
    )

    result = await service.answer(
        user_id=user.id,
        document_ids=[document.id],
        query="alpha",
        language="pt-BR",
        limit=1,
        neighbor_window=1,
    )

    assert result.answer == "Grounded integration answer"
    assert result.sources == ["Dense RAG Document"]
    assert result.retrieval_strategy == "dense"
    assert result.retrieval_queries == ["alpha"]

    assert generation_provider.prompt is not None

    assert "Neighbor before the relevant evidence." in generation_provider.prompt
    assert "Alpha evidence is the primary relevant fact." in generation_provider.prompt
    assert "Neighbor after the relevant evidence." in generation_provider.prompt

    assert "Question:" in generation_provider.prompt
    assert "alpha" in generation_provider.prompt
    assert "Requested language:" in generation_provider.prompt
    assert "pt-BR" in generation_provider.prompt


@pytest.mark.asyncio
async def test_rag_answer_returns_sources_from_multiple_documents(
    db_session: AsyncSession,
) -> None:
    """Return unique source titles for retrieved documents."""
    user = await _create_user(
        db_session,
        email="rag-multiple-documents@example.com",
    )

    first_document = await _create_ready_document(
        db_session,
        user_id=user.id,
        title="First Source",
    )

    second_document = await _create_ready_document(
        db_session,
        user_id=user.id,
        title="Second Source",
    )

    await _create_chunk(
        db_session,
        document_id=first_document.id,
        chunk_index=0,
        content="First alpha evidence.",
        embedding=_vector(1.0, 0.0),
    )

    await _create_chunk(
        db_session,
        document_id=second_document.id,
        chunk_index=0,
        content="Second alpha evidence.",
        embedding=_vector(0.99, 0.01),
    )

    generation_provider = FakeGenerationProvider()

    service = _build_rag_service(
        db_session,
        generation_provider=generation_provider,
    )

    result = await service.answer(
        user_id=user.id,
        document_ids=[
            first_document.id,
            second_document.id,
        ],
        query="alpha",
        language="en",
        limit=2,
        neighbor_window=0,
    )

    assert result.answer == "Grounded integration answer"

    assert set(result.sources) == {
        "First Source",
        "Second Source",
    }

    assert len(result.sources) == 2


@pytest.mark.asyncio
async def test_rag_answer_expands_neighbor_context_from_real_database(
    db_session: AsyncSession,
) -> None:
    """Include neighboring chunks around the semantic retrieval anchor."""
    user = await _create_user(
        db_session,
        email="rag-neighbors@example.com",
    )

    document = await _create_ready_document(
        db_session,
        user_id=user.id,
        title="Neighbor Source",
    )

    contents = [
        "Chunk zero.",
        "Chunk one neighbor.",
        "Chunk two semantic anchor.",
        "Chunk three neighbor.",
        "Chunk four.",
    ]

    for index, content in enumerate(contents):
        embedding = _vector(1.0, 0.0) if index == 2 else _vector(0.0, 1.0)

        await _create_chunk(
            db_session,
            document_id=document.id,
            chunk_index=index,
            content=content,
            embedding=embedding,
        )

    generation_provider = FakeGenerationProvider()

    service = _build_rag_service(
        db_session,
        generation_provider=generation_provider,
    )

    await service.answer(
        user_id=user.id,
        document_ids=[document.id],
        query="alpha",
        language="en",
        limit=1,
        neighbor_window=1,
    )

    assert generation_provider.prompt is not None

    assert "Chunk one neighbor." in generation_provider.prompt
    assert "Chunk two semantic anchor." in generation_provider.prompt
    assert "Chunk three neighbor." in generation_provider.prompt

    assert "Chunk zero." not in generation_provider.prompt
    assert "Chunk four." not in generation_provider.prompt


@pytest.mark.asyncio
async def test_rag_answer_with_query_transformation_uses_multi_query_rrf(
    db_session: AsyncSession,
) -> None:
    """Run transformed multi-query retrieval through RRF and generation."""
    user = await _create_user(
        db_session,
        email="rag-transformed@example.com",
    )

    document = await _create_ready_document(
        db_session,
        user_id=user.id,
        title="Transformed Query Source",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Evidence strongly related to alpha.",
        embedding=_vector(1.0, 0.0),
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=1,
        content="Evidence strongly related to beta.",
        embedding=_vector(0.0, 1.0),
    )

    generation_provider = FakeGenerationProvider()

    transformation_service = QueryTransformationService(
        FakeQueryTransformationProvider(),
        max_generated_queries=2,
        max_attempts=2,
    )

    service = _build_rag_service(
        db_session,
        generation_provider=generation_provider,
        query_transformation_service=transformation_service,
    )

    result = await service.answer(
        user_id=user.id,
        document_ids=[document.id],
        query="Find alpha and beta evidence",
        language="en",
        limit=2,
        neighbor_window=0,
        transform_query=True,
    )

    assert result.answer == "Grounded integration answer"

    assert result.retrieval_strategy == "multi_query_rrf"

    assert result.retrieval_queries == [
        "Find alpha and beta evidence",
        "alpha evidence",
        "beta evidence",
    ]

    assert result.sources == [
        "Transformed Query Source",
    ]

    assert generation_provider.prompt is not None
    assert "Evidence strongly related to alpha." in generation_provider.prompt
    assert "Evidence strongly related to beta." in generation_provider.prompt


@pytest.mark.asyncio
async def test_rag_answer_rejects_query_transformation_without_service(
    db_session: AsyncSession,
) -> None:
    """Reject transformed retrieval when no transformer is configured."""
    user = await _create_user(
        db_session,
        email="rag-missing-transformer@example.com",
    )

    document = await _create_ready_document(
        db_session,
        user_id=user.id,
        title="Missing Transformer",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Alpha evidence.",
        embedding=_vector(1.0, 0.0),
    )

    generation_provider = FakeGenerationProvider()

    service = _build_rag_service(
        db_session,
        generation_provider=generation_provider,
    )

    with pytest.raises(
        RuntimeError,
        match="Query transformation requested but service is not configured",
    ):
        await service.answer(
            user_id=user.id,
            document_ids=[document.id],
            query="alpha",
            language="en",
            transform_query=True,
        )


@pytest.mark.asyncio
async def test_rag_answer_rejects_other_users_document(
    db_session: AsyncSession,
) -> None:
    """Preserve document ownership through the complete RAG pipeline."""
    owner = await _create_user(
        db_session,
        email="rag-owner@example.com",
    )

    other_user = await _create_user(
        db_session,
        email="rag-other@example.com",
    )

    document = await _create_ready_document(
        db_session,
        user_id=owner.id,
        title="Private RAG Source",
    )

    await _create_chunk(
        db_session,
        document_id=document.id,
        chunk_index=0,
        content="Private evidence.",
        embedding=_vector(1.0, 0.0),
    )

    document_id = document.id
    other_user_id = other_user.id

    generation_provider = FakeGenerationProvider()

    service = _build_rag_service(
        db_session,
        generation_provider=generation_provider,
    )

    with pytest.raises(
        ValueError,
        match="One or more selected documents are invalid",
    ):
        await service.answer(
            user_id=other_user_id,
            document_ids=[document_id],
            query="alpha",
            language="en",
        )
