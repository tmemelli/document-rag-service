"""
Integration tests for DocumentService against PostgreSQL.
"""

import hashlib

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.models.document_status_history import (
    DocumentStatusHistory,
)
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.document_status_history_repository import (
    DocumentStatusHistoryRepository,
)
from document_rag_service.repositories.user_repository import (
    UserRepository,
)
from document_rag_service.services.document_service import DocumentService


async def _create_user(
    session: AsyncSession,
    *,
    email: str,
):
    """Create one persisted owner for document-service tests."""
    repository = UserRepository(session)

    return await repository.create(
        email=email,
        password_hash="test-password-hash",
    )


@pytest.mark.asyncio
async def test_register_uploaded_document_persists_document_and_checksum(
    db_session: AsyncSession,
) -> None:
    """Persist uploaded document metadata and its SHA-256 checksum."""
    user = await _create_user(
        db_session,
        email="document-service-create@example.com",
    )

    content = b"integration document content"

    service = DocumentService(db_session)

    document = await service.register_uploaded_document(
        user_id=user.id,
        title="Integration document",
        file_name="integration.pdf",
        mime_type="application/pdf",
        content=content,
    )

    expected_checksum = hashlib.sha256(content).hexdigest()

    assert document.id > 0
    assert document.user_id == user.id
    assert document.title == "Integration document"
    assert document.source_type == "upload"
    assert document.file_name == "integration.pdf"
    assert document.mime_type == "application/pdf"
    assert document.checksum == expected_checksum
    assert document.status == "pending"

    repository = DocumentRepository(db_session)

    persisted = await repository.get_by_id(
        document_id=document.id,
        user_id=user.id,
    )

    assert persisted is not None
    assert persisted.checksum == expected_checksum


@pytest.mark.asyncio
async def test_register_uploaded_document_creates_initial_status_history(
    db_session: AsyncSession,
) -> None:
    """Create the initial None-to-pending status event atomically."""
    user = await _create_user(
        db_session,
        email="document-service-history@example.com",
    )

    service = DocumentService(db_session)

    document = await service.register_uploaded_document(
        user_id=user.id,
        title="History document",
        file_name="history.pdf",
        mime_type="application/pdf",
        content=b"history content",
    )

    history_repository = DocumentStatusHistoryRepository(db_session)

    history = await history_repository.list_by_document_id(document.id)

    assert len(history) == 1

    event = history[0]

    assert event.document_id == document.id
    assert event.from_status is None
    assert event.to_status == "pending"
    assert event.component == "document_service"
    assert event.operation == "register_document"


@pytest.mark.asyncio
async def test_register_uploaded_document_rejects_pending_duplicate_for_same_user(
    db_session: AsyncSession,
) -> None:
    """Reject duplicate content while the existing document is pending."""
    user = await _create_user(
        db_session,
        email="document-service-duplicate@example.com",
    )

    service = DocumentService(db_session)

    content = b"duplicate content"

    original = await service.register_uploaded_document(
        user_id=user.id,
        title="Original",
        file_name="original.pdf",
        mime_type="application/pdf",
        content=content,
    )

    with pytest.raises(
        ValueError,
        match=r"Document content is already being processed",
    ):
        await service.register_uploaded_document(
            user_id=user.id,
            title="Duplicate",
            file_name="duplicate.pdf",
            mime_type="application/pdf",
            content=content,
        )

    repository = DocumentRepository(db_session)

    documents = await repository.list_all(
        user_id=user.id,
    )

    assert len(documents) == 1
    assert documents[0].id == original.id


@pytest.mark.asyncio
async def test_register_uploaded_document_allows_same_content_for_different_users(
    db_session: AsyncSession,
) -> None:
    """Scope duplicate detection to each document owner."""
    first_user = await _create_user(
        db_session,
        email="document-service-first@example.com",
    )
    second_user = await _create_user(
        db_session,
        email="document-service-second@example.com",
    )

    service = DocumentService(db_session)

    content = b"shared content"

    first = await service.register_uploaded_document(
        user_id=first_user.id,
        title="First copy",
        file_name="first.pdf",
        mime_type="application/pdf",
        content=content,
    )

    second = await service.register_uploaded_document(
        user_id=second_user.id,
        title="Second copy",
        file_name="second.pdf",
        mime_type="application/pdf",
        content=content,
    )

    assert first.id != second.id
    assert first.checksum == second.checksum
    assert first.user_id == first_user.id
    assert second.user_id == second_user.id


@pytest.mark.asyncio
async def test_delete_document_removes_owned_document(
    db_session: AsyncSession,
) -> None:
    """Delete a document owned by the requesting user."""
    user = await _create_user(
        db_session,
        email="document-service-delete@example.com",
    )

    service = DocumentService(db_session)

    document = await service.register_uploaded_document(
        user_id=user.id,
        title="Delete document",
        file_name="delete.pdf",
        mime_type="application/pdf",
        content=b"delete me",
    )

    document_id = document.id

    await service.delete_document(
        document_id=document_id,
        user_id=user.id,
    )

    result = await db_session.scalar(
        select(Document.id).where(Document.id == document_id)
    )

    assert result is None


@pytest.mark.asyncio
async def test_delete_document_rejects_other_users_document(
    db_session: AsyncSession,
) -> None:
    """Prevent deleting a document owned by another user."""
    owner = await _create_user(
        db_session,
        email="document-service-owner@example.com",
    )
    other_user = await _create_user(
        db_session,
        email="document-service-other@example.com",
    )

    service = DocumentService(db_session)

    document = await service.register_uploaded_document(
        user_id=owner.id,
        title="Private document",
        file_name="private.pdf",
        mime_type="application/pdf",
        content=b"private content",
    )

    with pytest.raises(
        ValueError,
        match="Document not found",
    ):
        await service.delete_document(
            document_id=document.id,
            user_id=other_user.id,
        )

    repository = DocumentRepository(db_session)

    persisted = await repository.get_by_id(
        document_id=document.id,
        user_id=owner.id,
    )

    assert persisted is not None


@pytest.mark.asyncio
async def test_delete_documents_removes_all_selected_owned_documents(
    db_session: AsyncSession,
) -> None:
    """Delete multiple selected documents in one service operation."""
    user = await _create_user(
        db_session,
        email="document-service-bulk@example.com",
    )

    service = DocumentService(db_session)

    first = await service.register_uploaded_document(
        user_id=user.id,
        title="First bulk document",
        file_name="first.pdf",
        mime_type="application/pdf",
        content=b"first bulk content",
    )
    second = await service.register_uploaded_document(
        user_id=user.id,
        title="Second bulk document",
        file_name="second.pdf",
        mime_type="application/pdf",
        content=b"second bulk content",
    )

    await service.delete_documents(
        document_ids=[
            first.id,
            second.id,
        ],
        user_id=user.id,
    )

    remaining = await db_session.scalars(
        select(Document.id).where(
            Document.id.in_(
                [
                    first.id,
                    second.id,
                ]
            )
        )
    )

    assert list(remaining.all()) == []


@pytest.mark.asyncio
async def test_delete_documents_is_all_or_nothing_when_one_document_is_missing(
    db_session: AsyncSession,
) -> None:
    """Validate all requested ids before deleting any document."""
    user = await _create_user(
        db_session,
        email="document-service-bulk-atomic@example.com",
    )

    service = DocumentService(db_session)

    document = await service.register_uploaded_document(
        user_id=user.id,
        title="Keep on failure",
        file_name="keep.pdf",
        mime_type="application/pdf",
        content=b"keep this document",
    )

    with pytest.raises(
        ValueError,
        match="One or more documents not found",
    ):
        await service.delete_documents(
            document_ids=[
                document.id,
                9_999_999_999,
            ],
            user_id=user.id,
        )

    persisted = await db_session.scalar(
        select(Document.id).where(Document.id == document.id)
    )

    assert persisted == document.id


@pytest.mark.asyncio
async def test_delete_document_cascades_initial_history(
    db_session: AsyncSession,
) -> None:
    """Delete service-created status history with its document."""
    user = await _create_user(
        db_session,
        email="document-service-history-cascade@example.com",
    )

    service = DocumentService(db_session)

    document = await service.register_uploaded_document(
        user_id=user.id,
        title="Cascade history",
        file_name="cascade.pdf",
        mime_type="application/pdf",
        content=b"cascade history content",
    )

    document_id = document.id

    await service.delete_document(
        document_id=document_id,
        user_id=user.id,
    )

    remaining_history = await db_session.scalars(
        select(DocumentStatusHistory.id).where(
            DocumentStatusHistory.document_id == document_id
        )
    )

    assert list(remaining_history.all()) == []
