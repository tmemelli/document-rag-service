"""Repository for document persistence and user-scoped lookup operations."""

from datetime import datetime

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document


class DocumentRepository:
    """
    Provide persistence operations for documents.

    The repository encapsulates SQLAlchemy queries for document creation,
    lookup, status updates, listing, checksum-based retrieval, and deletion.
    User-scoped queries always include the owning user identifier so documents
    cannot be retrieved across account boundaries.

    Transaction boundaries remain the responsibility of the calling service.
    Mutating operations flush changes so database state is available within
    the current transaction without committing it.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the document repository.

        Args:
            session: Asynchronous SQLAlchemy session used for persistence
                operations.
        """

        self._session = session

    async def create(
        self,
        *,
        user_id: int,
        title: str,
        source_type: str = "upload",
        source_url: str | None = None,
        file_name: str | None = None,
        mime_type: str | None = None,
        checksum: str | None = None,
    ) -> Document:
        """
        Create and persist a document.

        The new entity is flushed and refreshed so database-generated values,
        including its identifier and server-side defaults, are available to the
        caller without committing the surrounding transaction.

        Args:
            user_id: Identifier of the user who owns the document.
            title: Human-readable document title.
            source_type: Origin of the document, such as ``upload`` or ``url``.
            source_url: Optional source URL for externally sourced documents.
            file_name: Optional original uploaded file name.
            mime_type: Optional MIME type associated with the document.
            checksum: Optional checksum used to identify identical content.

        Returns:
            Newly created and refreshed document entity.
        """

        document = Document(
            user_id=user_id,
            title=title,
            source_type=source_type,
            source_url=source_url,
            file_name=file_name,
            mime_type=mime_type,
            checksum=checksum,
        )

        self._session.add(document)
        await self._session.flush()
        await self._session.refresh(document)

        return document

    async def get_by_id(
        self,
        *,
        document_id: int,
        user_id: int,
    ) -> Document | None:
        """
        Return a document by identifier while enforcing user ownership.

        Args:
            document_id: Identifier of the document to retrieve.
            user_id: Identifier of the user who must own the document.

        Returns:
            Matching user-owned document, or ``None`` when the document does
            not exist or belongs to another user.
        """

        statement = select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id,
        )

        return await self._session.scalar(statement)

    async def update_status(
        self,
        document: Document,
        status: str,
    ) -> Document:
        """
        Update and persist a document processing status.

        The change is flushed and the entity is refreshed so the caller sees
        the database-backed state immediately. The transaction is not committed
        by the repository.

        Args:
            document: Persisted document whose status should be updated.
            status: New processing status.

        Returns:
            Updated and refreshed document entity.
        """

        document.status = status

        if status != "processing":
            document.processing_token = None
            document.processing_lease_expires_at = None

        await self._session.flush()
        await self._session.refresh(document)

        return document

    async def list_all(
        self,
        *,
        user_id: int,
    ) -> list[Document]:
        """
        Return all documents owned by a user.

        Documents are ordered by creation time in descending order so the most
        recently registered documents appear first.

        Args:
            user_id: Identifier of the document owner.

        Returns:
            User-owned documents ordered from newest to oldest.
        """

        statement = (
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
        )

        result = await self._session.scalars(statement)

        return list(result.all())

    async def list_by_ids(
        self,
        *,
        document_ids: list[int],
        user_id: int,
    ) -> list[Document]:
        """
        Return user-owned documents matching a collection of identifiers.

        Ownership filtering is applied together with the identifier filter so
        documents belonging to other users are never included. An empty input
        list returns immediately without querying the database.

        Args:
            document_ids: Document identifiers to retrieve.
            user_id: Identifier of the user who must own the documents.

        Returns:
            Matching documents owned by the specified user.
        """

        if not document_ids:
            return []

        statement = select(Document).where(
            Document.user_id == user_id,
            Document.id.in_(document_ids),
        )

        result = await self._session.scalars(statement)

        return list(result.all())

    async def get_by_checksum(
        self,
        *,
        checksum: str,
        user_id: int,
    ) -> Document | None:
        """
        Return a user-owned document matching a content checksum.

        This lookup supports duplicate-content detection while keeping the
        comparison scoped to the owning user.

        Args:
            checksum: Content checksum to search for.
            user_id: Identifier of the user who must own the document.

        Returns:
            Matching user-owned document, or ``None`` when no matching checksum
            exists for the specified user.
        """

        statement = select(Document).where(
            Document.checksum == checksum,
            Document.user_id == user_id,
        )

        return await self._session.scalar(statement)

    async def claim_failed_for_retry(
        self,
        *,
        document_id: int,
        user_id: int,
        title: str,
        file_name: str,
        mime_type: str,
    ) -> Document | None:
        """
        Atomically reserve a failed document for a new ingestion attempt.

        The conditional update prevents two concurrent uploads from claiming
        the same failed document. A successful claim moves the document back
        to ``pending`` and refreshes upload metadata from the latest request.

        Args:
            document_id: Identifier of the failed document to retry.
            user_id: Identifier of the document owner.
            title: Title supplied for the new upload attempt.
            file_name: File name supplied for the new upload attempt.
            mime_type: MIME type supplied for the new upload attempt.

        Returns:
            The claimed document in ``pending`` state, or ``None`` when the
            document is no longer failed and therefore could not be claimed.
        """

        statement = (
            update(Document)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
                Document.status == "failed",
            )
            .values(
                title=title,
                file_name=file_name,
                mime_type=mime_type,
                status="pending",
                processing_token=None,
                processing_lease_expires_at=None,
            )
            .returning(Document)
        )

        return await self._session.scalar(statement)

    async def claim_pending_for_processing(
        self,
        *,
        document_id: int,
        user_id: int,
        processing_token: str,
        lease_expires_at: datetime,
    ) -> Document | None:
        """
        Atomically claim a pending document for one processing attempt.

        The conditional update prevents two ingestion workers from moving the
        same pending document into processing concurrently. A successful claim
        stores a fencing token and lease expiration used to distinguish active
        work from interrupted work.

        Args:
            document_id: Identifier of the pending document.
            user_id: Identifier of the document owner.
            processing_token: Unique token identifying the processing attempt.
            lease_expires_at: Expiration timestamp for the current lease.

        Returns:
            Claimed document in ``processing`` state, or ``None`` when the
            document is no longer pending.
        """

        statement = (
            update(Document)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
                Document.status == "pending",
            )
            .values(
                status="processing",
                processing_token=processing_token,
                processing_lease_expires_at=lease_expires_at,
            )
            .returning(Document)
        )

        return await self._session.scalar(statement)

    async def renew_processing_lease(
        self,
        *,
        document_id: int,
        user_id: int,
        processing_token: str,
        lease_expires_at: datetime,
    ) -> bool:
        """
        Extend the lease only while the same processing attempt still owns it.

        Returns:
            ``True`` when the lease was renewed, otherwise ``False``.
        """

        statement = (
            update(Document)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
                Document.status == "processing",
                Document.processing_token == processing_token,
            )
            .values(
                processing_lease_expires_at=lease_expires_at,
            )
            .returning(Document.id)
        )

        return await self._session.scalar(statement) is not None

    async def complete_processing(
        self,
        *,
        document_id: int,
        user_id: int,
        processing_token: str,
    ) -> Document | None:
        """
        Mark processing ready only when the caller still owns the active lease.

        The token check acts as a fencing guard. A worker whose lease was
        recovered by a newer attempt cannot commit a stale successful result.

        Returns:
            Ready document when ownership is still valid, otherwise ``None``.
        """

        statement = (
            update(Document)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
                Document.status == "processing",
                Document.processing_token == processing_token,
            )
            .values(
                status="ready",
                processing_token=None,
                processing_lease_expires_at=None,
            )
            .returning(Document)
        )

        return await self._session.scalar(statement)

    async def fail_processing_if_owned(
        self,
        *,
        document_id: int,
        user_id: int,
        processing_token: str,
    ) -> Document | None:
        """
        Mark processing failed only while the caller still owns its lease.

        Returns:
            Failed document when ownership is still valid, otherwise ``None``.
        """

        statement = (
            update(Document)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
                Document.status == "processing",
                Document.processing_token == processing_token,
            )
            .values(
                status="failed",
                processing_token=None,
                processing_lease_expires_at=None,
            )
            .returning(Document)
        )

        return await self._session.scalar(statement)

    async def recover_interrupted_processing(
        self,
        *,
        document_id: int,
        user_id: int,
    ) -> Document | None:
        """
        Atomically recover a processing document whose lease is stale or absent.

        A missing lease supports recovery of documents created before lease
        tracking existed. An expired lease indicates that the worker stopped
        renewing its heartbeat. Active leases are never reclaimed.

        Returns:
            Document transitioned to ``failed`` when recovery succeeds, or
            ``None`` when the document remains actively processing or changed
            state concurrently.
        """

        statement = (
            update(Document)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
                Document.status == "processing",
                or_(
                    Document.processing_lease_expires_at.is_(None),
                    Document.processing_lease_expires_at <= func.now(),
                ),
            )
            .values(
                status="failed",
                processing_token=None,
                processing_lease_expires_at=None,
            )
            .returning(Document)
        )

        return await self._session.scalar(statement)

    async def delete(
        self,
        document: Document,
    ) -> None:
        """
        Delete a document from the current database session.

        The deletion is flushed immediately so database-side effects are
        reflected within the active transaction. Committing or rolling back
        remains the responsibility of the calling service.

        Args:
            document: Persisted document entity to delete.
        """

        await self._session.delete(document)
        await self._session.flush()
