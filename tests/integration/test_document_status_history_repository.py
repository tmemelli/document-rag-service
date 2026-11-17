"""
Integration tests for DocumentStatusHistoryRepository against PostgreSQL.
"""

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.models.user import User
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.document_status_history_repository import (
    DocumentStatusHistoryRepository,
)
from document_rag_service.repositories.user_repository import (
    UserRepository,
)


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
    """Create one persisted document."""
    repository = DocumentRepository(session)

    return await repository.create(
        user_id=user_id,
        title=title,
        file_name=f"{title}.pdf",
        mime_type="application/pdf",
    )


@pytest.mark.asyncio
async def test_create_persists_status_history(
    db_session: AsyncSession,
) -> None:
    """Persist a complete document status transition."""
    user = await _create_user(
        db_session,
        email="history-create@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="History create",
    )

    repository = DocumentStatusHistoryRepository(db_session)

    history = await repository.create(
        document_id=document.id,
        from_status="pending",
        to_status="processing",
        component="ingestion",
        operation="start_ingestion",
        attempt_number=2,
        details={
            "file_name": "history.pdf",
            "chunk_count": 12,
        },
    )

    assert history.id > 0
    assert history.document_id == document.id
    assert history.from_status == "pending"
    assert history.to_status == "processing"
    assert history.component == "ingestion"
    assert history.operation == "start_ingestion"
    assert history.attempt_number == 2
    assert history.details == {
        "file_name": "history.pdf",
        "chunk_count": 12,
    }
    assert history.created_at is not None


@pytest.mark.asyncio
async def test_create_accepts_initial_transition_from_none(
    db_session: AsyncSession,
) -> None:
    """Allow the first document transition to have no previous status."""
    user = await _create_user(
        db_session,
        email="history-initial@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Initial history",
    )

    repository = DocumentStatusHistoryRepository(db_session)

    history = await repository.create(
        document_id=document.id,
        from_status=None,
        to_status="pending",
        component="document_service",
        operation="register_document",
    )

    assert history.from_status is None
    assert history.to_status == "pending"
    assert history.attempt_number == 1
    assert history.details == {}


@pytest.mark.asyncio
async def test_get_latest_attempt_number_returns_highest_attempt(
    db_session: AsyncSession,
) -> None:
    """Return the highest attempt recorded for one document."""
    user = await _create_user(
        db_session,
        email="history-latest-attempt@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Latest attempt",
    )

    repository = DocumentStatusHistoryRepository(db_session)

    await repository.create(
        document_id=document.id,
        from_status=None,
        to_status="pending",
        component="document_service",
        operation="register_document",
        attempt_number=1,
    )
    await repository.create(
        document_id=document.id,
        from_status="failed",
        to_status="pending",
        component="document_service",
        operation="retry_uploaded_document",
        attempt_number=2,
    )

    assert await repository.get_latest_attempt_number(document.id) == 2


@pytest.mark.asyncio
async def test_get_latest_attempt_number_returns_none_without_history(
    db_session: AsyncSession,
) -> None:
    """Return None when a document has no status-history records."""
    assert (
        await DocumentStatusHistoryRepository(
            db_session
        ).get_latest_attempt_number(9_999_999_999)
        is None
    )


@pytest.mark.asyncio
async def test_list_by_document_id_returns_only_matching_history(
    db_session: AsyncSession,
) -> None:
    """Return status events only for the requested document."""
    user = await _create_user(
        db_session,
        email="history-filter@example.com",
    )

    first_document = await _create_document(
        db_session,
        user_id=user.id,
        title="First history document",
    )
    second_document = await _create_document(
        db_session,
        user_id=user.id,
        title="Second history document",
    )

    repository = DocumentStatusHistoryRepository(db_session)

    first = await repository.create(
        document_id=first_document.id,
        from_status=None,
        to_status="pending",
        component="document_service",
        operation="register_document",
    )
    second = await repository.create(
        document_id=first_document.id,
        from_status="pending",
        to_status="processing",
        component="ingestion",
        operation="start_ingestion",
    )

    await repository.create(
        document_id=second_document.id,
        from_status=None,
        to_status="pending",
        component="document_service",
        operation="register_document",
    )

    result = await repository.list_by_document_id(first_document.id)

    assert {history.id for history in result} == {
        first.id,
        second.id,
    }


@pytest.mark.asyncio
async def test_list_by_document_id_returns_empty_list_when_missing(
    db_session: AsyncSession,
) -> None:
    """Return an empty history for an unknown document."""
    repository = DocumentStatusHistoryRepository(db_session)

    result = await repository.list_by_document_id(9_999_999_999)

    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        ("invalid", "ready"),
        ("pending", "invalid"),
    ],
)
async def test_database_rejects_invalid_status_values(
    db_session: AsyncSession,
    from_status: str,
    to_status: str,
) -> None:
    """Enforce valid document status values in PostgreSQL."""
    user = await _create_user(
        db_session,
        email=f"history-invalid-{from_status}-{to_status}@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Invalid history status",
    )

    repository = DocumentStatusHistoryRepository(db_session)

    async with db_session.begin_nested():
        with pytest.raises(IntegrityError):
            await repository.create(
                document_id=document.id,
                from_status=from_status,
                to_status=to_status,
                component="test",
                operation="invalid_status",
            )


@pytest.mark.asyncio
async def test_database_rejects_non_positive_attempt_number(
    db_session: AsyncSession,
) -> None:
    """Enforce positive history attempt numbers."""
    user = await _create_user(
        db_session,
        email="history-attempt@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Invalid history attempt",
    )

    repository = DocumentStatusHistoryRepository(db_session)

    async with db_session.begin_nested():
        with pytest.raises(IntegrityError):
            await repository.create(
                document_id=document.id,
                from_status="pending",
                to_status="processing",
                component="ingestion",
                operation="start_ingestion",
                attempt_number=0,
            )


@pytest.mark.asyncio
async def test_database_rejects_history_for_unknown_document(
    db_session: AsyncSession,
) -> None:
    """Enforce the document foreign-key relationship."""
    repository = DocumentStatusHistoryRepository(db_session)

    async with db_session.begin_nested():
        with pytest.raises(IntegrityError):
            await repository.create(
                document_id=9_999_999_999,
                from_status="pending",
                to_status="processing",
                component="ingestion",
                operation="start_ingestion",
            )
