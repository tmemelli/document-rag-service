"""
Integration tests for DocumentRepository against PostgreSQL.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.models.user import User
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.user_repository import (
    UserRepository,
)


async def _create_user(
    session: AsyncSession,
    *,
    email: str,
) -> User:
    """Create a database user required by document ownership."""
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
    checksum: str | None = None,
) -> Document:
    """Create one owned document for integration tests."""
    repository = DocumentRepository(session)

    return await repository.create(
        user_id=user_id,
        title=title,
        file_name=f"{title}.pdf",
        mime_type="application/pdf",
        checksum=checksum,
    )


@pytest.mark.asyncio
async def test_create_persists_document_with_defaults(
    db_session: AsyncSession,
) -> None:
    """Persist a document with database-generated fields and defaults."""
    user = await _create_user(
        db_session,
        email="create-document@example.com",
    )

    repository = DocumentRepository(db_session)

    document = await repository.create(
        user_id=user.id,
        title="Test document",
        file_name="test.pdf",
        mime_type="application/pdf",
        checksum="checksum-create",
    )

    assert document.id > 0
    assert document.user_id == user.id
    assert document.title == "Test document"
    assert document.source_type == "upload"
    assert document.file_name == "test.pdf"
    assert document.mime_type == "application/pdf"
    assert document.checksum == "checksum-create"
    assert document.status == "pending"
    assert document.processing_token is None
    assert document.processing_lease_expires_at is None
    assert document.created_at is not None
    assert document.updated_at is not None


@pytest.mark.asyncio
async def test_create_persists_url_document(
    db_session: AsyncSession,
) -> None:
    """Persist URL source metadata through the repository."""
    user = await _create_user(
        db_session,
        email="url-document@example.com",
    )

    repository = DocumentRepository(db_session)

    document = await repository.create(
        user_id=user.id,
        title="Remote document",
        source_type="url",
        source_url="https://example.com/document",
    )

    assert document.source_type == "url"
    assert document.source_url == "https://example.com/document"
    assert document.file_name is None
    assert document.mime_type is None


@pytest.mark.asyncio
async def test_get_by_id_returns_owned_document(
    db_session: AsyncSession,
) -> None:
    """Retrieve a document when it belongs to the requested user."""
    user = await _create_user(
        db_session,
        email="owned@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Owned",
    )

    repository = DocumentRepository(db_session)

    result = await repository.get_by_id(
        document_id=document.id,
        user_id=user.id,
    )

    assert result is not None
    assert result.id == document.id
    assert result.user_id == user.id


@pytest.mark.asyncio
async def test_get_by_id_hides_document_from_other_user(
    db_session: AsyncSession,
) -> None:
    """Enforce document ownership during identifier lookup."""
    owner = await _create_user(
        db_session,
        email="owner@example.com",
    )
    other_user = await _create_user(
        db_session,
        email="other@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=owner.id,
        title="Private",
    )

    repository = DocumentRepository(db_session)

    result = await repository.get_by_id(
        document_id=document.id,
        user_id=other_user.id,
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_unknown_document(
    db_session: AsyncSession,
) -> None:
    """Return None when the requested document does not exist."""
    user = await _create_user(
        db_session,
        email="missing-document@example.com",
    )

    repository = DocumentRepository(db_session)

    result = await repository.get_by_id(
        document_id=9_999_999_999,
        user_id=user.id,
    )

    assert result is None


@pytest.mark.asyncio
async def test_update_status_persists_new_status(
    db_session: AsyncSession,
) -> None:
    """Persist document status transitions."""
    user = await _create_user(
        db_session,
        email="status@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Status document",
    )

    repository = DocumentRepository(db_session)

    updated = await repository.update_status(
        document,
        "ready",
    )

    document_id = updated.id

    db_session.expire(updated)

    retrieved = await repository.get_by_id(
        document_id=document_id,
        user_id=user.id,
    )

    assert retrieved is not None
    assert retrieved.status == "ready"


@pytest.mark.asyncio
async def test_claim_failed_for_retry_is_atomic_and_refreshes_metadata(
    db_session: AsyncSession,
) -> None:
    """Claim one failed document once and move it back to pending."""
    user = await _create_user(
        db_session,
        email="retry-claim@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Original retry title",
        checksum="retry-checksum",
    )

    repository = DocumentRepository(db_session)
    await repository.update_status(document, "failed")

    claimed = await repository.claim_failed_for_retry(
        document_id=document.id,
        user_id=user.id,
        title="Retried title",
        file_name="retried.pdf",
        mime_type="application/pdf",
    )

    assert claimed is not None
    assert claimed.id == document.id
    assert claimed.status == "pending"
    assert claimed.title == "Retried title"
    assert claimed.file_name == "retried.pdf"
    assert claimed.mime_type == "application/pdf"

    second_claim = await repository.claim_failed_for_retry(
        document_id=document.id,
        user_id=user.id,
        title="Concurrent retry",
        file_name="concurrent.pdf",
        mime_type="application/pdf",
    )

    assert second_claim is None


@pytest.mark.asyncio
async def test_claim_pending_for_processing_sets_lease_atomically(
    db_session: AsyncSession,
) -> None:
    """Claim pending work once and persist its fencing token and lease."""
    user = await _create_user(
        db_session,
        email="processing-claim@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Processing claim",
    )

    repository = DocumentRepository(db_session)
    lease_expires_at = datetime.now(UTC) + timedelta(minutes=2)

    claimed = await repository.claim_pending_for_processing(
        document_id=document.id,
        user_id=user.id,
        processing_token="a" * 32,
        lease_expires_at=lease_expires_at,
    )

    assert claimed is not None
    assert claimed.status == "processing"
    assert claimed.processing_token == "a" * 32
    assert claimed.processing_lease_expires_at == lease_expires_at

    second_claim = await repository.claim_pending_for_processing(
        document_id=document.id,
        user_id=user.id,
        processing_token="b" * 32,
        lease_expires_at=lease_expires_at,
    )

    assert second_claim is None


@pytest.mark.asyncio
async def test_renew_processing_lease_requires_matching_token(
    db_session: AsyncSession,
) -> None:
    """Renew only the worker that still owns the current processing token."""
    user = await _create_user(
        db_session,
        email="lease-renew@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Lease renew",
    )

    repository = DocumentRepository(db_session)
    first_expiration = datetime.now(UTC) + timedelta(minutes=1)

    await repository.claim_pending_for_processing(
        document_id=document.id,
        user_id=user.id,
        processing_token="a" * 32,
        lease_expires_at=first_expiration,
    )

    renewed_expiration = datetime.now(UTC) + timedelta(minutes=2)

    assert await repository.renew_processing_lease(
        document_id=document.id,
        user_id=user.id,
        processing_token="a" * 32,
        lease_expires_at=renewed_expiration,
    )
    assert not await repository.renew_processing_lease(
        document_id=document.id,
        user_id=user.id,
        processing_token="b" * 32,
        lease_expires_at=renewed_expiration,
    )


@pytest.mark.asyncio
async def test_recover_interrupted_processing_reclaims_only_stale_lease(
    db_session: AsyncSession,
) -> None:
    """Recover expired processing while leaving a valid lease untouched."""
    user = await _create_user(
        db_session,
        email="lease-recovery@example.com",
    )
    stale_document = await _create_document(
        db_session,
        user_id=user.id,
        title="Stale processing",
    )
    active_document = await _create_document(
        db_session,
        user_id=user.id,
        title="Active processing",
    )

    repository = DocumentRepository(db_session)

    await repository.claim_pending_for_processing(
        document_id=stale_document.id,
        user_id=user.id,
        processing_token="a" * 32,
        lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    await repository.claim_pending_for_processing(
        document_id=active_document.id,
        user_id=user.id,
        processing_token="b" * 32,
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=2),
    )

    recovered = await repository.recover_interrupted_processing(
        document_id=stale_document.id,
        user_id=user.id,
    )
    active_recovery = await repository.recover_interrupted_processing(
        document_id=active_document.id,
        user_id=user.id,
    )

    assert recovered is not None
    assert recovered.status == "failed"
    assert recovered.processing_token is None
    assert recovered.processing_lease_expires_at is None
    assert active_recovery is None


@pytest.mark.asyncio
async def test_fail_processing_is_fenced_by_processing_token(
    db_session: AsyncSession,
) -> None:
    """Prevent stale workers from failing a processing attempt they do not own."""
    user = await _create_user(
        db_session,
        email="lease-failure@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Lease failure",
    )

    repository = DocumentRepository(db_session)

    await repository.claim_pending_for_processing(
        document_id=document.id,
        user_id=user.id,
        processing_token="a" * 32,
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=2),
    )

    stale_failure = await repository.fail_processing_if_owned(
        document_id=document.id,
        user_id=user.id,
        processing_token="b" * 32,
    )
    failed = await repository.fail_processing_if_owned(
        document_id=document.id,
        user_id=user.id,
        processing_token="a" * 32,
    )

    assert stale_failure is None
    assert failed is not None
    assert failed.status == "failed"
    assert failed.processing_token is None
    assert failed.processing_lease_expires_at is None


@pytest.mark.asyncio
async def test_complete_processing_is_fenced_by_processing_token(
    db_session: AsyncSession,
) -> None:
    """Prevent a stale worker from completing a document it no longer owns."""
    user = await _create_user(
        db_session,
        email="lease-completion@example.com",
    )
    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Lease completion",
    )

    repository = DocumentRepository(db_session)

    await repository.claim_pending_for_processing(
        document_id=document.id,
        user_id=user.id,
        processing_token="a" * 32,
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=2),
    )

    stale_completion = await repository.complete_processing(
        document_id=document.id,
        user_id=user.id,
        processing_token="b" * 32,
    )
    completed = await repository.complete_processing(
        document_id=document.id,
        user_id=user.id,
        processing_token="a" * 32,
    )

    assert stale_completion is None
    assert completed is not None
    assert completed.status == "ready"
    assert completed.processing_token is None
    assert completed.processing_lease_expires_at is None


@pytest.mark.asyncio
async def test_list_all_returns_only_users_documents(
    db_session: AsyncSession,
) -> None:
    """List only documents owned by the requested user."""
    first_user = await _create_user(
        db_session,
        email="list-first@example.com",
    )
    second_user = await _create_user(
        db_session,
        email="list-second@example.com",
    )

    first_document = await _create_document(
        db_session,
        user_id=first_user.id,
        title="First",
    )
    second_document = await _create_document(
        db_session,
        user_id=first_user.id,
        title="Second",
    )

    await _create_document(
        db_session,
        user_id=second_user.id,
        title="Foreign",
    )

    repository = DocumentRepository(db_session)

    result = await repository.list_all(
        user_id=first_user.id,
    )

    result_ids = {document.id for document in result}

    assert result_ids == {
        first_document.id,
        second_document.id,
    }


@pytest.mark.asyncio
async def test_list_all_returns_empty_list_when_user_has_no_documents(
    db_session: AsyncSession,
) -> None:
    """Return an empty collection when a user owns no documents."""
    user = await _create_user(
        db_session,
        email="empty-list@example.com",
    )

    repository = DocumentRepository(db_session)

    result = await repository.list_all(
        user_id=user.id,
    )

    assert result == []


@pytest.mark.asyncio
async def test_list_by_ids_returns_selected_owned_documents(
    db_session: AsyncSession,
) -> None:
    """Return only selected documents owned by the user."""
    user = await _create_user(
        db_session,
        email="selected@example.com",
    )

    first = await _create_document(
        db_session,
        user_id=user.id,
        title="Selected first",
    )
    second = await _create_document(
        db_session,
        user_id=user.id,
        title="Selected second",
    )
    third = await _create_document(
        db_session,
        user_id=user.id,
        title="Not selected",
    )

    repository = DocumentRepository(db_session)

    result = await repository.list_by_ids(
        user_id=user.id,
        document_ids=[
            first.id,
            second.id,
        ],
    )

    result_ids = {document.id for document in result}

    assert result_ids == {
        first.id,
        second.id,
    }
    assert third.id not in result_ids


@pytest.mark.asyncio
async def test_list_by_ids_excludes_other_users_documents(
    db_session: AsyncSession,
) -> None:
    """Prevent cross-user document access through bulk identifier lookup."""
    owner = await _create_user(
        db_session,
        email="bulk-owner@example.com",
    )
    other_user = await _create_user(
        db_session,
        email="bulk-other@example.com",
    )

    owned = await _create_document(
        db_session,
        user_id=owner.id,
        title="Owned bulk",
    )
    foreign = await _create_document(
        db_session,
        user_id=other_user.id,
        title="Foreign bulk",
    )

    repository = DocumentRepository(db_session)

    result = await repository.list_by_ids(
        user_id=owner.id,
        document_ids=[
            owned.id,
            foreign.id,
        ],
    )

    assert [document.id for document in result] == [owned.id]


@pytest.mark.asyncio
async def test_list_by_ids_returns_empty_list_for_empty_ids(
    db_session: AsyncSession,
) -> None:
    """Avoid issuing a meaningless lookup for an empty identifier list."""
    user = await _create_user(
        db_session,
        email="empty-ids@example.com",
    )

    repository = DocumentRepository(db_session)

    result = await repository.list_by_ids(
        user_id=user.id,
        document_ids=[],
    )

    assert result == []


@pytest.mark.asyncio
async def test_get_by_checksum_returns_owned_document(
    db_session: AsyncSession,
) -> None:
    """Retrieve an owned document by content checksum."""
    user = await _create_user(
        db_session,
        email="checksum@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Checksum",
        checksum="abc123",
    )

    repository = DocumentRepository(db_session)

    result = await repository.get_by_checksum(
        checksum="abc123",
        user_id=user.id,
    )

    assert result is not None
    assert result.id == document.id


@pytest.mark.asyncio
async def test_get_by_checksum_hides_other_users_document(
    db_session: AsyncSession,
) -> None:
    """Scope checksum lookups to document ownership."""
    owner = await _create_user(
        db_session,
        email="checksum-owner@example.com",
    )
    other_user = await _create_user(
        db_session,
        email="checksum-other@example.com",
    )

    await _create_document(
        db_session,
        user_id=owner.id,
        title="Private checksum",
        checksum="shared-checksum",
    )

    repository = DocumentRepository(db_session)

    result = await repository.get_by_checksum(
        checksum="shared-checksum",
        user_id=other_user.id,
    )

    assert result is None


@pytest.mark.asyncio
async def test_get_by_checksum_returns_none_when_missing(
    db_session: AsyncSession,
) -> None:
    """Return None when no owned document matches the checksum."""
    user = await _create_user(
        db_session,
        email="missing-checksum@example.com",
    )

    repository = DocumentRepository(db_session)

    result = await repository.get_by_checksum(
        checksum="does-not-exist",
        user_id=user.id,
    )

    assert result is None


@pytest.mark.asyncio
async def test_delete_removes_document(
    db_session: AsyncSession,
) -> None:
    """Delete a persisted document from PostgreSQL."""
    user = await _create_user(
        db_session,
        email="delete@example.com",
    )

    document = await _create_document(
        db_session,
        user_id=user.id,
        title="Delete me",
    )

    document_id = document.id

    repository = DocumentRepository(db_session)

    await repository.delete(document)

    result = await repository.get_by_id(
        document_id=document_id,
        user_id=user.id,
    )

    assert result is None
