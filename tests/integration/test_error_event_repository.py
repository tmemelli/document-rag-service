"""
Integration tests for ErrorEventRepository against PostgreSQL.
"""

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.models.user import User
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.error_event_repository import (
    ErrorEventRepository,
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
async def test_create_persists_complete_error_event(
    db_session: AsyncSession,
) -> None:
    """Persist complete structured error metadata."""
    user = await _create_user(
        db_session,
        email="error-create@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Error create",
    )

    repository = ErrorEventRepository(db_session)

    event = await repository.create(
        document_id=document.id,
        component="embedding",
        operation="embed_chunks",
        error_type="ProviderError",
        error_code="EMBEDDING_FAILED",
        message="Embedding provider failed",
        details={
            "chunk_index": 4,
            "provider": "cloudflare",
        },
        retryable=True,
        attempt_number=2,
        correlation_id="request-123",
    )

    assert event.id > 0
    assert event.document_id == document.id
    assert event.component == "embedding"
    assert event.operation == "embed_chunks"
    assert event.error_type == "ProviderError"
    assert event.error_code == "EMBEDDING_FAILED"
    assert event.message == "Embedding provider failed"
    assert event.details == {
        "chunk_index": 4,
        "provider": "cloudflare",
    }
    assert event.retryable is True
    assert event.attempt_number == 2
    assert event.correlation_id == "request-123"
    assert event.occurred_at is not None


@pytest.mark.asyncio
async def test_create_uses_default_optional_values(
    db_session: AsyncSession,
) -> None:
    """Persist repository defaults for optional error metadata."""
    repository = ErrorEventRepository(db_session)

    event = await repository.create(
        document_id=None,
        component="application",
        operation="startup",
        error_type="RuntimeError",
        message="Startup failure",
    )

    assert event.document_id is None
    assert event.error_code is None
    assert event.details == {}
    assert event.retryable is False
    assert event.attempt_number == 1
    assert event.correlation_id is None


@pytest.mark.asyncio
async def test_create_accepts_error_without_document(
    db_session: AsyncSession,
) -> None:
    """Allow application errors not associated with a document."""
    repository = ErrorEventRepository(db_session)

    event = await repository.create(
        document_id=None,
        component="llm",
        operation="generate",
        error_type="TimeoutError",
        message="Inference timed out",
        retryable=True,
    )

    assert event.id > 0
    assert event.document_id is None
    assert event.retryable is True


@pytest.mark.asyncio
async def test_list_by_document_id_returns_only_matching_errors(
    db_session: AsyncSession,
) -> None:
    """Return errors only for the requested document."""
    user = await _create_user(
        db_session,
        email="error-filter@example.com",
    )

    first_document = await _create_document(
        db_session,
        user_id=user.id,
        title="First error document",
    )
    second_document = await _create_document(
        db_session,
        user_id=user.id,
        title="Second error document",
    )

    repository = ErrorEventRepository(db_session)

    first = await repository.create(
        document_id=first_document.id,
        component="extraction",
        operation="extract_text",
        error_type="ValueError",
        message="First error",
    )
    second = await repository.create(
        document_id=first_document.id,
        component="embedding",
        operation="embed_chunks",
        error_type="RuntimeError",
        message="Second error",
    )

    await repository.create(
        document_id=second_document.id,
        component="ingestion",
        operation="persist_chunks",
        error_type="RuntimeError",
        message="Foreign error",
    )

    result = await repository.list_by_document_id(first_document.id)

    assert {event.id for event in result} == {
        first.id,
        second.id,
    }


@pytest.mark.asyncio
async def test_list_by_document_id_returns_empty_list_when_missing(
    db_session: AsyncSession,
) -> None:
    """Return an empty collection when no error events exist."""
    repository = ErrorEventRepository(db_session)

    result = await repository.list_by_document_id(9_999_999_999)

    assert result == []


@pytest.mark.asyncio
async def test_database_rejects_non_positive_attempt_number(
    db_session: AsyncSession,
) -> None:
    """Enforce positive error attempt numbers."""
    repository = ErrorEventRepository(db_session)

    async with db_session.begin_nested():
        with pytest.raises(IntegrityError):
            await repository.create(
                document_id=None,
                component="embedding",
                operation="embed_chunks",
                error_type="RuntimeError",
                message="Invalid attempt",
                attempt_number=0,
            )


@pytest.mark.asyncio
async def test_database_rejects_unknown_document_reference(
    db_session: AsyncSession,
) -> None:
    """Enforce the optional document foreign key when an id is supplied."""
    repository = ErrorEventRepository(db_session)

    async with db_session.begin_nested():
        with pytest.raises(IntegrityError):
            await repository.create(
                document_id=9_999_999_999,
                component="ingestion",
                operation="persist_chunks",
                error_type="RuntimeError",
                message="Unknown document",
            )
