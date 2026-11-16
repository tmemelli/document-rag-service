"""
Integration tests for IngestionService against PostgreSQL.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.document_status_history_repository import (
    DocumentStatusHistoryRepository,
)
from document_rag_service.repositories.error_event_repository import (
    ErrorEventRepository,
)
from document_rag_service.repositories.user_repository import UserRepository
from document_rag_service.services.chunking_service import ChunkingService
from document_rag_service.services.document_service import DocumentService
from document_rag_service.services.embedding_service import (
    EmbeddingProvider,
    EmbeddingService,
)
from document_rag_service.services.ingestion_service import IngestionService

EMBEDDING_DIMENSION = 512


def _rtf_content(text: str) -> bytes:
    """Build a minimal valid RTF document for integration tests."""
    escaped_text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

    return f"{{\\rtf1\\ansi {escaped_text}}}".encode()


class FakeEmbeddingProvider:
    """Deterministic embedding provider for integration tests."""

    @property
    def model_name(self) -> str:
        """Return a stable fake model name."""
        return "fake-integration-embedding-model"

    @property
    def dimension(self) -> int:
        """Return the production embedding dimension."""
        return EMBEDDING_DIMENSION

    @property
    def max_sequence_length(self) -> int:
        """Return a realistic maximum embedding sequence length."""
        return 8192

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Return deterministic non-zero document embeddings."""
        embeddings: list[list[float]] = []

        for index, _ in enumerate(texts):
            vector = [0.0] * EMBEDDING_DIMENSION
            vector[0] = 1.0
            vector[1] = float(index + 1) / 100.0
            embeddings.append(vector)

        return embeddings

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Return one deterministic query embedding."""
        vector = [0.0] * EMBEDDING_DIMENSION
        vector[0] = 1.0
        return vector

    def count_document_tokens(
        self,
        text: str,
    ) -> int:
        """Return a deterministic token count for integration tests."""
        return len(text.split())

    async def aclose(self) -> None:
        """Release provider resources."""
        return None


class FailingEmbeddingProvider(FakeEmbeddingProvider):
    """Embedding provider that always fails for document embeddings."""

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Simulate an embedding provider failure."""
        raise RuntimeError("Simulated embedding failure")


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


def _build_ingestion_service(
    session: AsyncSession,
    *,
    provider: EmbeddingProvider,
) -> IngestionService:
    """Build ingestion service with deterministic local dependencies."""
    embedding_service = EmbeddingService(provider)

    chunking_service = ChunkingService(
        chunk_size=80,
        chunk_overlap=20,
    )

    return IngestionService(
        session,
        embedding_service,
        chunking_service,
    )


@pytest.mark.asyncio
async def test_ingest_file_content_persists_chunks_embeddings_and_ready_status(
    db_session: AsyncSession,
) -> None:
    """Run the complete successful ingestion flow."""
    user = await _create_user(
        db_session,
        email="ingestion-success@example.com",
    )

    document_service = DocumentService(db_session)

    content = _rtf_content(
        "Retrieval augmented generation combines document retrieval with "
        "language models. Documents are divided into chunks before being "
        "embedded. Relevant chunks are retrieved later to ground answers."
    )

    document = await document_service.register_uploaded_document(
        user_id=user.id,
        title="Successful ingestion",
        file_name="success.rtf",
        mime_type="application/rtf",
        content=content,
    )
    document_id = document.id
    user_id = user.id

    ingestion_service = _build_ingestion_service(
        db_session,
        provider=FakeEmbeddingProvider(),
    )

    await ingestion_service.ingest_file_content(
        document_id=document_id,
        user_id=user_id,
        content=content,
        file_name="success.rtf",
    )

    document_repository = DocumentRepository(db_session)
    chunk_repository = DocumentChunkRepository(db_session)
    history_repository = DocumentStatusHistoryRepository(db_session)
    error_repository = ErrorEventRepository(db_session)

    persisted_document = await document_repository.get_by_id(
        document_id=document_id,
        user_id=user_id,
    )

    assert persisted_document is not None
    assert persisted_document.status == "ready"

    chunks = await chunk_repository.list_by_document_id(document_id)

    assert len(chunks) >= 2
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.token_count >= 0
        assert chunk.embedding is not None
        assert len(chunk.embedding) == EMBEDDING_DIMENSION

    history = await history_repository.list_by_document_id(document_id)

    assert [(event.from_status, event.to_status) for event in history] == [
        (None, "pending"),
        ("pending", "processing"),
        ("processing", "ready"),
    ]

    errors = await error_repository.list_by_document_id(document_id)

    assert errors == []


@pytest.mark.asyncio
async def test_ingestion_is_scoped_to_document_owner(
    db_session: AsyncSession,
) -> None:
    """Reject ingestion when the document belongs to another user."""
    owner = await _create_user(
        db_session,
        email="ingestion-owner@example.com",
    )
    other_user = await _create_user(
        db_session,
        email="ingestion-other@example.com",
    )

    document_service = DocumentService(db_session)

    content = _rtf_content("Owner-only document content.")

    document = await document_service.register_uploaded_document(
        user_id=owner.id,
        title="Private ingestion",
        file_name="private.rtf",
        mime_type="application/rtf",
        content=content,
    )
    document_id = document.id
    owner_id = owner.id
    other_user_id = other_user.id

    ingestion_service = _build_ingestion_service(
        db_session,
        provider=FakeEmbeddingProvider(),
    )

    with pytest.raises(ValueError):
        await ingestion_service.ingest_file_content(
            document_id=document_id,
            user_id=other_user_id,
            content=content,
            file_name="private.rtf",
        )

    document_repository = DocumentRepository(db_session)
    chunk_repository = DocumentChunkRepository(db_session)

    persisted_document = await document_repository.get_by_id(
        document_id=document_id,
        user_id=owner_id,
    )

    assert persisted_document is not None
    assert persisted_document.status == "pending"

    chunks = await chunk_repository.list_by_document_id(document_id)

    assert chunks == []


@pytest.mark.asyncio
async def test_ingestion_failure_marks_document_failed_and_records_error(
    db_session: AsyncSession,
) -> None:
    """Persist failed state, history and error event after provider failure."""
    user = await _create_user(
        db_session,
        email="ingestion-failure@example.com",
    )

    document_service = DocumentService(db_session)

    content = _rtf_content(
        "This document contains enough text to enter the ingestion pipeline "
        "before the embedding provider fails deliberately."
    )

    document = await document_service.register_uploaded_document(
        user_id=user.id,
        title="Failed ingestion",
        file_name="failure.rtf",
        mime_type="application/rtf",
        content=content,
    )
    document_id = document.id
    user_id = user.id

    ingestion_service = _build_ingestion_service(
        db_session,
        provider=FailingEmbeddingProvider(),
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated embedding failure",
    ):
        await ingestion_service.ingest_file_content(
            document_id=document_id,
            user_id=user_id,
            content=content,
            file_name="failure.rtf",
        )

    document_repository = DocumentRepository(db_session)
    chunk_repository = DocumentChunkRepository(db_session)
    history_repository = DocumentStatusHistoryRepository(db_session)
    error_repository = ErrorEventRepository(db_session)

    persisted_document = await document_repository.get_by_id(
        document_id=document_id,
        user_id=user_id,
    )

    assert persisted_document is not None
    assert persisted_document.status == "failed"

    chunks = await chunk_repository.list_by_document_id(document_id)

    assert chunks == []

    history = await history_repository.list_by_document_id(document_id)

    transitions = [(event.from_status, event.to_status) for event in history]

    assert transitions == [
        (None, "pending"),
        ("pending", "processing"),
        ("processing", "failed"),
    ]

    errors = await error_repository.list_by_document_id(document_id)

    assert len(errors) == 1

    error = errors[0]

    assert error.document_id == document_id
    assert error.error_type == "RuntimeError"
    assert error.error_code == "CHUNK_PERSISTENCE_FAILED"
    assert error.message == "Failed to persist document chunks."
    assert error.retryable is False


@pytest.mark.asyncio
async def test_failed_ingestion_rolls_back_partial_chunks(
    db_session: AsyncSession,
) -> None:
    """Prevent partial chunk persistence when ingestion fails."""
    user = await _create_user(
        db_session,
        email="ingestion-rollback@example.com",
    )

    document_service = DocumentService(db_session)

    content = _rtf_content(
        "First section contains enough characters for one chunk. "
        "Second section contains additional text that should create more "
        "chunks. The provider failure must leave no partial chunk rows."
    )

    document = await document_service.register_uploaded_document(
        user_id=user.id,
        title="Rollback ingestion",
        file_name="rollback.rtf",
        mime_type="application/rtf",
        content=content,
    )
    document_id = document.id
    user_id = user.id

    ingestion_service = _build_ingestion_service(
        db_session,
        provider=FailingEmbeddingProvider(),
    )

    with pytest.raises(RuntimeError):
        await ingestion_service.ingest_file_content(
            document_id=document_id,
            user_id=user_id,
            content=content,
            file_name="rollback.rtf",
        )

    chunk_repository = DocumentChunkRepository(db_session)

    chunks = await chunk_repository.list_by_document_id(document_id)

    assert chunks == []


@pytest.mark.asyncio
async def test_ingestion_persists_distinct_embeddings_for_multiple_chunks(
    db_session: AsyncSession,
) -> None:
    """Persist one embedding vector for every generated chunk."""
    user = await _create_user(
        db_session,
        email="ingestion-embeddings@example.com",
    )

    document_service = DocumentService(db_session)

    content = _rtf_content(
        "Alpha information about retrieval systems and embeddings. "
        "Beta information about chunking strategies and overlap. "
        "Gamma information about ranking and contextual generation."
    )

    document = await document_service.register_uploaded_document(
        user_id=user.id,
        title="Embedding ingestion",
        file_name="embeddings.rtf",
        mime_type="application/rtf",
        content=content,
    )
    document_id = document.id
    user_id = user.id

    ingestion_service = _build_ingestion_service(
        db_session,
        provider=FakeEmbeddingProvider(),
    )

    await ingestion_service.ingest_file_content(
        document_id=document_id,
        user_id=user_id,
        content=content,
        file_name="embeddings.rtf",
    )

    chunk_repository = DocumentChunkRepository(db_session)

    chunks = await chunk_repository.list_by_document_id(document_id)

    assert len(chunks) >= 2

    embeddings = [chunk.embedding for chunk in chunks]

    assert all(embedding is not None for embedding in embeddings)

    second_coordinates = [
        embedding[1] for embedding in embeddings if embedding is not None
    ]

    assert len(set(second_coordinates)) == len(second_coordinates)
