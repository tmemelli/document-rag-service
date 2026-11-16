"""
Integration tests for relational integrity and transaction behavior.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.models.document_chunk import DocumentChunk
from document_rag_service.models.document_status_history import (
    DocumentStatusHistory,
)
from document_rag_service.models.error_event import ErrorEvent
from document_rag_service.models.user import User
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
from document_rag_service.repositories.user_repository import (
    UserRepository,
)

EMBEDDING_DIMENSION = 512


async def _row_exists(
    session: AsyncSession,
    model: (
        type[Document]
        | type[DocumentChunk]
        | type[DocumentStatusHistory]
        | type[ErrorEvent]
        | type[User]
    ),
    row_id: int,
) -> bool:
    """Return whether a database row exists for the given primary key."""
    result = await session.scalar(select(model.id).where(model.id == row_id))

    return result is not None


def _embedding() -> list[float]:
    """Build a deterministic production-dimension embedding."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[0] = 1.0
    return vector


async def _create_user(
    session: AsyncSession,
    *,
    email: str,
) -> User:
    """Create one persisted test user."""
    repository = UserRepository(session)

    return await repository.create(
        email=email,
        password_hash="test-password-hash",
    )


async def _create_document_graph(
    session: AsyncSession,
    *,
    user_id: int,
    title: str,
) -> tuple[
    Document,
    DocumentChunk,
    DocumentStatusHistory,
    ErrorEvent,
]:
    """Create a document together with dependent database records."""
    document_repository = DocumentRepository(session)
    chunk_repository = DocumentChunkRepository(session)
    history_repository = DocumentStatusHistoryRepository(session)
    error_repository = ErrorEventRepository(session)

    document = await document_repository.create(
        user_id=user_id,
        title=title,
        file_name=f"{title}.pdf",
        mime_type="application/pdf",
    )

    chunk = await chunk_repository.create(
        document_id=document.id,
        chunk_index=0,
        content="Test chunk",
        embedding=_embedding(),
        token_count=10,
    )

    history = await history_repository.create(
        document_id=document.id,
        from_status=None,
        to_status="pending",
        component="document_service",
        operation="register_document",
    )

    error = await error_repository.create(
        document_id=document.id,
        component="ingestion",
        operation="persist_chunks",
        error_type="RuntimeError",
        message="Test error",
    )

    return document, chunk, history, error


@pytest.mark.asyncio
async def test_deleting_document_cascades_all_dependent_records(
    db_session: AsyncSession,
) -> None:
    """Delete chunks, status history and errors with their document."""
    user = await _create_user(
        db_session,
        email="document-cascade@example.com",
    )

    document, chunk, history, error = await _create_document_graph(
        db_session,
        user_id=user.id,
        title="Cascade document",
    )

    document_id = document.id
    chunk_id = chunk.id
    history_id = history.id
    error_id = error.id

    repository = DocumentRepository(db_session)

    await repository.delete(document)

    assert (
        await db_session.scalar(select(Document.id).where(Document.id == document_id))
        is None
    )

    assert (
        await db_session.scalar(
            select(DocumentChunk.id).where(DocumentChunk.id == chunk_id)
        )
        is None
    )

    assert (
        await db_session.scalar(
            select(DocumentStatusHistory.id).where(
                DocumentStatusHistory.id == history_id
            )
        )
        is None
    )

    assert (
        await db_session.scalar(select(ErrorEvent.id).where(ErrorEvent.id == error_id))
        is None
    )


@pytest.mark.asyncio
async def test_deleting_user_cascades_entire_document_graph(
    db_session: AsyncSession,
) -> None:
    """Delete documents and their dependents when the owner is deleted."""
    user = await _create_user(
        db_session,
        email="user-cascade@example.com",
    )

    document, chunk, history, error = await _create_document_graph(
        db_session,
        user_id=user.id,
        title="Owned cascade",
    )

    user_id = user.id
    document_id = document.id
    chunk_id = chunk.id
    history_id = history.id
    error_id = error.id

    await db_session.delete(user)
    await db_session.flush()

    assert await db_session.scalar(select(User.id).where(User.id == user_id)) is None

    assert (
        await db_session.scalar(select(Document.id).where(Document.id == document_id))
        is None
    )

    assert (
        await db_session.scalar(
            select(DocumentChunk.id).where(DocumentChunk.id == chunk_id)
        )
        is None
    )

    assert (
        await db_session.scalar(
            select(DocumentStatusHistory.id).where(
                DocumentStatusHistory.id == history_id
            )
        )
        is None
    )

    assert (
        await db_session.scalar(select(ErrorEvent.id).where(ErrorEvent.id == error_id))
        is None
    )


@pytest.mark.asyncio
async def test_document_delete_does_not_affect_other_documents(
    db_session: AsyncSession,
) -> None:
    """Keep unrelated document graphs intact during cascade deletion."""
    first_user = await _create_user(
        db_session,
        email="cascade-first@example.com",
    )
    second_user = await _create_user(
        db_session,
        email="cascade-second@example.com",
    )

    first_document, first_chunk, first_history, first_error = (
        await _create_document_graph(
            db_session,
            user_id=first_user.id,
            title="Delete this graph",
        )
    )

    second_document, second_chunk, second_history, second_error = (
        await _create_document_graph(
            db_session,
            user_id=second_user.id,
            title="Preserve this graph",
        )
    )

    first_ids = (
        first_document.id,
        first_chunk.id,
        first_history.id,
        first_error.id,
    )
    second_ids = (
        second_document.id,
        second_chunk.id,
        second_history.id,
        second_error.id,
    )

    repository = DocumentRepository(db_session)

    await repository.delete(first_document)

    assert (
        await db_session.scalar(select(Document.id).where(Document.id == first_ids[0]))
        is None
    )

    assert (
        await db_session.scalar(
            select(DocumentChunk.id).where(DocumentChunk.id == first_ids[1])
        )
        is None
    )

    assert (
        await db_session.scalar(
            select(DocumentStatusHistory.id).where(
                DocumentStatusHistory.id == first_ids[2]
            )
        )
        is None
    )

    assert (
        await db_session.scalar(
            select(ErrorEvent.id).where(ErrorEvent.id == first_ids[3])
        )
        is None
    )

    assert (
        await db_session.scalar(select(Document.id).where(Document.id == second_ids[0]))
        == second_ids[0]
    )

    assert (
        await db_session.scalar(
            select(DocumentChunk.id).where(DocumentChunk.id == second_ids[1])
        )
        == second_ids[1]
    )

    assert (
        await db_session.scalar(
            select(DocumentStatusHistory.id).where(
                DocumentStatusHistory.id == second_ids[2]
            )
        )
        == second_ids[2]
    )

    assert (
        await db_session.scalar(
            select(ErrorEvent.id).where(ErrorEvent.id == second_ids[3])
        )
        == second_ids[3]
    )


@pytest.mark.asyncio
async def test_nested_transaction_rollback_restores_document_state(
    db_session: AsyncSession,
) -> None:
    """Roll back a multi-table state transition atomically."""
    user = await _create_user(
        db_session,
        email="transaction-rollback@example.com",
    )

    document_repository = DocumentRepository(db_session)
    history_repository = DocumentStatusHistoryRepository(db_session)
    error_repository = ErrorEventRepository(db_session)

    document = await document_repository.create(
        user_id=user.id,
        title="Transactional document",
    )

    document_id = document.id

    savepoint = await db_session.begin_nested()

    await document_repository.update_status(
        document,
        "failed",
    )

    history = await history_repository.create(
        document_id=document.id,
        from_status="pending",
        to_status="failed",
        component="ingestion",
        operation="persist_chunks",
    )

    error = await error_repository.create(
        document_id=document.id,
        component="ingestion",
        operation="persist_chunks",
        error_type="RuntimeError",
        message="Simulated transactional failure",
    )

    history_id = history.id
    error_id = error.id

    await savepoint.rollback()

    db_session.expire(document)

    restored_document = await document_repository.get_by_id(
        document_id=document_id,
        user_id=user.id,
    )

    assert restored_document is not None
    assert restored_document.status == "pending"

    assert (
        await db_session.get(
            DocumentStatusHistory,
            history_id,
        )
        is None
    )

    assert (
        await db_session.get(
            ErrorEvent,
            error_id,
        )
        is None
    )


@pytest.mark.asyncio
async def test_nested_transaction_commit_preserves_all_changes(
    db_session: AsyncSession,
) -> None:
    """Commit a multi-table state transition atomically."""
    user = await _create_user(
        db_session,
        email="transaction-commit@example.com",
    )

    document_repository = DocumentRepository(db_session)
    history_repository = DocumentStatusHistoryRepository(db_session)
    error_repository = ErrorEventRepository(db_session)

    document = await document_repository.create(
        user_id=user.id,
        title="Committed transaction",
    )

    document_id = document.id

    savepoint = await db_session.begin_nested()

    await document_repository.update_status(
        document,
        "failed",
    )

    history = await history_repository.create(
        document_id=document.id,
        from_status="pending",
        to_status="failed",
        component="ingestion",
        operation="persist_chunks",
    )

    error = await error_repository.create(
        document_id=document.id,
        component="ingestion",
        operation="persist_chunks",
        error_type="RuntimeError",
        message="Persisted transactional failure",
    )

    history_id = history.id
    error_id = error.id

    await savepoint.commit()

    db_session.expire(document)

    persisted_document = await document_repository.get_by_id(
        document_id=document_id,
        user_id=user.id,
    )

    assert persisted_document is not None
    assert persisted_document.status == "failed"

    assert (
        await db_session.get(
            DocumentStatusHistory,
            history_id,
        )
        is not None
    )

    assert (
        await db_session.get(
            ErrorEvent,
            error_id,
        )
        is not None
    )


@pytest.mark.asyncio
async def test_database_contains_no_cross_document_orphans_after_delete(
    db_session: AsyncSession,
) -> None:
    """Verify cascade deletion leaves no dependent rows behind."""
    user = await _create_user(
        db_session,
        email="orphans@example.com",
    )

    document, _, _, _ = await _create_document_graph(
        db_session,
        user_id=user.id,
        title="No orphan rows",
    )

    document_id = document.id

    repository = DocumentRepository(db_session)

    await repository.delete(document)

    chunks = await db_session.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )
    histories = await db_session.scalars(
        select(DocumentStatusHistory).where(
            DocumentStatusHistory.document_id == document_id
        )
    )
    errors = await db_session.scalars(
        select(ErrorEvent).where(ErrorEvent.document_id == document_id)
    )

    assert list(chunks.all()) == []
    assert list(histories.all()) == []
    assert list(errors.all()) == []
