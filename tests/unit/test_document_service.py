"""
Tests for document lifecycle management and persistence coordination.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.services.document_service import DocumentService


def _build_document(
    *,
    document_id: int = 1,
    user_id: int = 42,
    title: str = "Test document",
    status: str = "pending",
    checksum: str | None = None,
) -> Document:
    """Create an in-memory document for service tests."""
    return Document(
        id=document_id,
        user_id=user_id,
        title=title,
        source_type="upload",
        file_name="document.pdf",
        mime_type="application/pdf",
        checksum=checksum,
        status=status,
    )


def _build_service() -> tuple[
    DocumentService,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    """Create a document service with isolated persistence test doubles."""
    session = MagicMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()

    document_repository = MagicMock()
    document_repository.get_by_checksum = AsyncMock()
    document_repository.create = AsyncMock()
    document_repository.get_by_id = AsyncMock()
    document_repository.update_status = AsyncMock()
    document_repository.claim_failed_for_retry = AsyncMock()
    document_repository.delete = AsyncMock()
    document_repository.list_by_ids = AsyncMock()

    status_history_repository = MagicMock()
    status_history_repository.create = AsyncMock()
    status_history_repository.get_latest_attempt_number = AsyncMock()

    checksum_service = MagicMock()
    checksum_service.sha256 = MagicMock()

    service = DocumentService(session)

    service._documents = document_repository
    service._status_history = status_history_repository
    service._checksum = checksum_service

    return (
        service,
        session,
        document_repository,
        status_history_repository,
        checksum_service,
    )


@pytest.mark.asyncio
async def test_register_uploaded_document_generates_checksum_from_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate a SHA-256 checksum before duplicate-content lookup."""
    (
        service,
        _,
        documents,
        _,
        checksum,
    ) = _build_service()

    checksum.sha256.return_value = "generated-checksum"
    documents.get_by_checksum.return_value = None

    register_document = AsyncMock(
        return_value=_build_document(checksum="generated-checksum")
    )

    monkeypatch.setattr(
        service,
        "register_document",
        register_document,
    )

    content = b"uploaded document bytes"

    await service.register_uploaded_document(
        user_id=42,
        title="My document",
        file_name="document.pdf",
        mime_type="application/pdf",
        content=content,
    )

    checksum.sha256.assert_called_once_with(content)

    documents.get_by_checksum.assert_awaited_once_with(
        checksum="generated-checksum",
        user_id=42,
    )


@pytest.mark.asyncio
async def test_register_uploaded_document_registers_new_upload() -> None:
    """Register a new upload when its content is not already owned by the user."""
    (
        service,
        _,
        documents,
        _,
        checksum,
    ) = _build_service()

    checksum.sha256.return_value = "generated-checksum"
    documents.get_by_checksum.return_value = None

    expected_document = _build_document(checksum="generated-checksum")

    register_document = AsyncMock(return_value=expected_document)

    service.register_document = register_document  # type: ignore[method-assign]

    result = await service.register_uploaded_document(
        user_id=42,
        title="My document",
        file_name="document.pdf",
        mime_type="application/pdf",
        content=b"content",
    )

    assert result is expected_document

    register_document.assert_awaited_once_with(
        user_id=42,
        title="My document",
        source_type="upload",
        file_name="document.pdf",
        mime_type="application/pdf",
        checksum="generated-checksum",
    )


@pytest.mark.asyncio
async def test_register_uploaded_document_allows_same_content_for_different_users() -> (
    None
):
    """Scope duplicate detection to the authenticated document owner."""
    (
        service,
        _,
        documents,
        _,
        checksum,
    ) = _build_service()

    checksum.sha256.return_value = "same-checksum"
    documents.get_by_checksum.return_value = None

    register_document = AsyncMock(
        return_value=_build_document(
            user_id=99,
            checksum="same-checksum",
        )
    )

    service.register_document = register_document  # type: ignore[method-assign]

    await service.register_uploaded_document(
        user_id=99,
        title="Document",
        file_name="document.pdf",
        mime_type="application/pdf",
        content=b"content",
    )

    documents.get_by_checksum.assert_awaited_once_with(
        checksum="same-checksum",
        user_id=99,
    )


@pytest.mark.asyncio
async def test_register_uploaded_document_rejects_ready_duplicate() -> None:
    """Reject uploaded content that already completed ingestion."""
    (
        service,
        _,
        documents,
        _,
        checksum,
    ) = _build_service()

    checksum.sha256.return_value = "duplicate-checksum"

    documents.get_by_checksum.return_value = _build_document(
        document_id=7,
        status="ready",
        checksum="duplicate-checksum",
    )

    with pytest.raises(
        ValueError,
        match=r"Document content already exists \(document_id=7\)",
    ):
        await service.register_uploaded_document(
            user_id=42,
            title="Duplicate",
            file_name="duplicate.pdf",
            mime_type="application/pdf",
            content=b"duplicate",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_value",
    [
        "pending",
        "processing",
    ],
)
async def test_register_uploaded_document_rejects_duplicate_being_processed(
    status_value: str,
) -> None:
    """Reject content whose existing document is pending or processing."""
    (
        service,
        _,
        documents,
        _,
        checksum,
    ) = _build_service()

    checksum.sha256.return_value = "duplicate-checksum"

    documents.get_by_checksum.return_value = _build_document(
        document_id=8,
        status=status_value,
        checksum="duplicate-checksum",
    )

    with pytest.raises(
        ValueError,
        match=(r"Document content is already being processed " r"\(document_id=8\)"),
    ):
        await service.register_uploaded_document(
            user_id=42,
            title="Duplicate",
            file_name="duplicate.pdf",
            mime_type="application/pdf",
            content=b"duplicate",
        )


@pytest.mark.asyncio
async def test_register_uploaded_document_retries_failed_duplicate() -> None:
    """Reserve a failed duplicate for a new ingestion attempt."""
    (
        service,
        session,
        documents,
        history,
        checksum,
    ) = _build_service()

    checksum.sha256.return_value = "failed-checksum"

    failed_document = _build_document(
        document_id=9,
        status="failed",
        checksum="failed-checksum",
    )
    claimed_document = _build_document(
        document_id=9,
        status="pending",
        checksum="failed-checksum",
    )

    documents.get_by_checksum.return_value = failed_document
    documents.claim_failed_for_retry.return_value = claimed_document
    history.get_latest_attempt_number.return_value = 1

    result = await service.register_uploaded_document(
        user_id=42,
        title="Retried document",
        file_name="retry.pdf",
        mime_type="application/pdf",
        content=b"failed",
    )

    assert result is claimed_document

    documents.claim_failed_for_retry.assert_awaited_once_with(
        document_id=9,
        user_id=42,
        title="Retried document",
        file_name="retry.pdf",
        mime_type="application/pdf",
    )
    history.create.assert_awaited_once_with(
        document_id=9,
        from_status="failed",
        to_status="pending",
        component="document_service",
        operation="retry_uploaded_document",
        attempt_number=2,
    )
    session.commit.assert_awaited_once_with()
    session.refresh.assert_awaited_once_with(claimed_document)


@pytest.mark.asyncio
async def test_register_uploaded_document_rejects_retry_claimed_concurrently() -> None:
    """Reject retry when another request claims the failed document first."""
    (
        service,
        _,
        documents,
        history,
        checksum,
    ) = _build_service()

    checksum.sha256.return_value = "failed-checksum"
    documents.get_by_checksum.return_value = _build_document(
        document_id=9,
        status="failed",
        checksum="failed-checksum",
    )
    documents.claim_failed_for_retry.return_value = None
    history.get_latest_attempt_number.return_value = 1

    with pytest.raises(
        ValueError,
        match=(r"Document content is already being processed \(document_id=9\)"),
    ):
        await service.register_uploaded_document(
            user_id=42,
            title="Retried document",
            file_name="retry.pdf",
            mime_type="application/pdf",
            content=b"failed",
        )

    history.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_register_document_creates_document_with_all_metadata() -> None:
    """Pass complete registration metadata to the document repository."""
    (
        service,
        _,
        documents,
        _,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=11,
    )
    documents.create.return_value = document

    await service.register_document(
        user_id=42,
        title="External document",
        source_type="url",
        source_url="https://example.com/document",
        file_name=None,
        mime_type="text/html",
        checksum="abc123",
    )

    documents.create.assert_awaited_once_with(
        user_id=42,
        title="External document",
        source_type="url",
        source_url="https://example.com/document",
        file_name=None,
        mime_type="text/html",
        checksum="abc123",
    )


@pytest.mark.asyncio
async def test_register_document_creates_initial_pending_status_history() -> None:
    """Record the initial pending state when a document is registered."""
    (
        service,
        _,
        documents,
        history,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=11,
    )
    documents.create.return_value = document

    await service.register_document(
        user_id=42,
        title="Document",
        source_type="upload",
    )

    history.create.assert_awaited_once_with(
        document_id=11,
        from_status=None,
        to_status="pending",
        component="document_service",
        operation="register_document",
    )


@pytest.mark.asyncio
async def test_register_document_commits_and_refreshes_document() -> None:
    """Commit registration and refresh the persisted entity before returning."""
    (
        service,
        session,
        documents,
        _,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=11,
    )
    documents.create.return_value = document

    result = await service.register_document(
        user_id=42,
        title="Document",
        source_type="upload",
    )

    session.commit.assert_awaited_once_with()
    session.refresh.assert_awaited_once_with(document)

    assert result is document


@pytest.mark.asyncio
async def test_register_document_does_not_commit_when_document_creation_fails() -> None:
    """Avoid committing when initial document persistence fails."""
    (
        service,
        session,
        documents,
        history,
        _,
    ) = _build_service()

    repository_error = RuntimeError("document persistence failed")

    documents.create.side_effect = repository_error

    with pytest.raises(
        RuntimeError,
        match="document persistence failed",
    ) as exc_info:
        await service.register_document(
            user_id=42,
            title="Document",
            source_type="upload",
        )

    assert exc_info.value is repository_error

    history.create.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_register_document_does_not_commit_when_history_creation_fails() -> None:
    """Avoid committing a document without its initial status-history record."""
    (
        service,
        session,
        documents,
        history,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=11,
    )
    documents.create.return_value = document

    history_error = RuntimeError("history persistence failed")
    history.create.side_effect = history_error

    with pytest.raises(
        RuntimeError,
        match="history persistence failed",
    ) as exc_info:
        await service.register_document(
            user_id=42,
            title="Document",
            source_type="upload",
        )

    assert exc_info.value is history_error

    session.commit.assert_not_awaited()
    session.refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_status_rejects_missing_document() -> None:
    """Reject status updates for documents not owned by the user."""
    (
        service,
        session,
        documents,
        history,
        _,
    ) = _build_service()

    documents.get_by_id.return_value = None

    with pytest.raises(
        ValueError,
        match="Document 99 not found",
    ):
        await service.update_status(
            document_id=99,
            user_id=42,
            status="ready",
        )

    documents.get_by_id.assert_awaited_once_with(
        document_id=99,
        user_id=42,
    )

    documents.update_status.assert_not_awaited()
    history.create.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_status_passes_document_and_new_status_to_repository() -> None:
    """Persist the requested new processing status."""
    (
        service,
        _,
        documents,
        _,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=12,
        status="processing",
    )

    documents.get_by_id.return_value = document

    await service.update_status(
        document_id=12,
        user_id=42,
        status="ready",
    )

    documents.update_status.assert_awaited_once_with(
        document=document,
        status="ready",
    )


@pytest.mark.asyncio
async def test_update_status_records_previous_and_new_status() -> None:
    """Record an accurate status transition in document history."""
    (
        service,
        _,
        documents,
        history,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=12,
        status="processing",
    )

    documents.get_by_id.return_value = document

    await service.update_status(
        document_id=12,
        user_id=42,
        status="ready",
    )

    history.create.assert_awaited_once_with(
        document_id=12,
        from_status="processing",
        to_status="ready",
        component="document_service",
        operation="update_status",
    )


@pytest.mark.asyncio
async def test_update_status_captures_previous_status_before_repository_mutation() -> (
    None
):
    """Preserve the original status even when repository mutation changes the entity."""

    (
        service,
        _,
        documents,
        history,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=12,
        status="pending",
    )

    documents.get_by_id.return_value = document

    async def mutate_status(
        *,
        document: Document,
        status: str,
    ) -> Document:
        document.status = status
        return document

    documents.update_status.side_effect = mutate_status

    await service.update_status(
        document_id=12,
        user_id=42,
        status="processing",
    )

    history.create.assert_awaited_once_with(
        document_id=12,
        from_status="pending",
        to_status="processing",
        component="document_service",
        operation="update_status",
    )


@pytest.mark.asyncio
async def test_update_status_commits_refreshes_and_returns_document() -> None:
    """Commit a status transition and return the refreshed document."""
    (
        service,
        session,
        documents,
        _,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=12,
        status="pending",
    )

    documents.get_by_id.return_value = document

    result = await service.update_status(
        document_id=12,
        user_id=42,
        status="processing",
    )

    session.commit.assert_awaited_once_with()
    session.refresh.assert_awaited_once_with(document)

    assert result is document


@pytest.mark.asyncio
async def test_update_status_does_not_commit_when_history_creation_fails() -> None:
    """Avoid committing a status change without its matching history event."""
    (
        service,
        session,
        documents,
        history,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=12,
        status="pending",
    )

    documents.get_by_id.return_value = document

    history.create.side_effect = RuntimeError("history failed")

    with pytest.raises(
        RuntimeError,
        match="history failed",
    ):
        await service.update_status(
            document_id=12,
            user_id=42,
            status="processing",
        )

    session.commit.assert_not_awaited()
    session.refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_document_rejects_missing_document() -> None:
    """Reject deletion when the document does not exist for the user."""
    (
        service,
        session,
        documents,
        _,
        _,
    ) = _build_service()

    documents.get_by_id.return_value = None

    with pytest.raises(
        ValueError,
        match="Document not found",
    ):
        await service.delete_document(
            document_id=100,
            user_id=42,
        )

    documents.delete.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_document_deletes_owned_document_and_commits() -> None:
    """Delete a user-owned document and commit the transaction."""
    (
        service,
        session,
        documents,
        _,
        _,
    ) = _build_service()

    document = _build_document(
        document_id=100,
    )

    documents.get_by_id.return_value = document

    await service.delete_document(
        document_id=100,
        user_id=42,
    )

    documents.get_by_id.assert_awaited_once_with(
        document_id=100,
        user_id=42,
    )
    documents.delete.assert_awaited_once_with(document)
    session.commit.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_delete_documents_rejects_empty_selection() -> None:
    """Reject bulk deletion when no document identifiers are supplied."""
    (
        service,
        session,
        documents,
        _,
        _,
    ) = _build_service()

    with pytest.raises(
        ValueError,
        match="At least one document must be selected",
    ):
        await service.delete_documents(
            document_ids=[],
            user_id=42,
        )

    documents.list_by_ids.assert_not_awaited()
    documents.delete.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_documents_deduplicates_identifiers_before_lookup() -> None:
    """Remove duplicate identifiers before validating bulk deletion."""
    (
        service,
        _,
        documents,
        _,
        _,
    ) = _build_service()

    first = _build_document(
        document_id=1,
    )
    second = _build_document(
        document_id=2,
    )
    third = _build_document(
        document_id=3,
    )

    documents.list_by_ids.return_value = [
        first,
        second,
        third,
    ]

    await service.delete_documents(
        document_ids=[
            1,
            2,
            1,
            3,
            2,
        ],
        user_id=42,
    )

    documents.list_by_ids.assert_awaited_once_with(
        user_id=42,
        document_ids=[
            1,
            2,
            3,
        ],
    )


@pytest.mark.asyncio
async def test_delete_documents_validates_all_documents_before_deleting_any() -> None:
    """Perform no deletion if any requested document is missing."""
    (
        service,
        session,
        documents,
        _,
        _,
    ) = _build_service()

    documents.list_by_ids.return_value = [
        _build_document(
            document_id=1,
        )
    ]

    with pytest.raises(
        ValueError,
        match="One or more documents not found",
    ):
        await service.delete_documents(
            document_ids=[
                1,
                2,
            ],
            user_id=42,
        )

    documents.delete.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_documents_deletes_every_validated_document() -> None:
    """Delete every validated document in a successful bulk operation."""
    (
        service,
        _,
        documents,
        _,
        _,
    ) = _build_service()

    first = _build_document(
        document_id=1,
    )
    second = _build_document(
        document_id=2,
    )
    third = _build_document(
        document_id=3,
    )

    documents.list_by_ids.return_value = [
        first,
        second,
        third,
    ]

    await service.delete_documents(
        document_ids=[
            1,
            2,
            3,
        ],
        user_id=42,
    )

    assert documents.delete.await_count == 3

    documents.delete.assert_any_await(first)
    documents.delete.assert_any_await(second)
    documents.delete.assert_any_await(third)


@pytest.mark.asyncio
async def test_delete_documents_commits_once_after_all_deletions() -> None:
    """Commit bulk deletion once after every document has been deleted."""
    (
        service,
        session,
        documents,
        _,
        _,
    ) = _build_service()

    documents.list_by_ids.return_value = [
        _build_document(
            document_id=1,
        ),
        _build_document(
            document_id=2,
        ),
    ]

    await service.delete_documents(
        document_ids=[
            1,
            2,
        ],
        user_id=42,
    )

    session.commit.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_delete_documents_propagates_repository_lookup_failure() -> None:
    """Propagate storage failures during bulk ownership validation."""
    (
        service,
        session,
        documents,
        _,
        _,
    ) = _build_service()

    repository_error = RuntimeError("database unavailable")

    documents.list_by_ids.side_effect = repository_error

    with pytest.raises(
        RuntimeError,
        match="database unavailable",
    ) as exc_info:
        await service.delete_documents(
            document_ids=[1, 2],
            user_id=42,
        )

    assert exc_info.value is repository_error

    documents.delete.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_documents_does_not_commit_when_deletion_fails() -> None:
    """Avoid committing a partially failed bulk deletion operation."""
    (
        service,
        session,
        documents,
        _,
        _,
    ) = _build_service()

    first = _build_document(
        document_id=1,
    )
    second = _build_document(
        document_id=2,
    )

    documents.list_by_ids.return_value = [
        first,
        second,
    ]

    documents.delete.side_effect = [
        None,
        RuntimeError("delete failed"),
    ]

    with pytest.raises(
        RuntimeError,
        match="delete failed",
    ):
        await service.delete_documents(
            document_ids=[1, 2],
            user_id=42,
        )

    session.commit.assert_not_awaited()
