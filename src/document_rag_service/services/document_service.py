"""Document lifecycle management service."""

from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.repositories.document_repository import DocumentRepository
from document_rag_service.repositories.document_status_history_repository import (
    DocumentStatusHistoryRepository,
)
from document_rag_service.services.checksum_service import ChecksumService


class DocumentService:
    """
    Coordinate document registration, status tracking, and deletion.

    The service owns document-level business rules while repositories remain
    responsible for persistence operations. Database commits are performed
    here so document mutations and their related status history records can be
    persisted atomically.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the document service.

        Args:
            session: Asynchronous SQLAlchemy session used by the service and
                its repositories.
        """

        self._session = session
        self._documents = DocumentRepository(session)
        self._status_history = DocumentStatusHistoryRepository(session)
        self._checksum = ChecksumService()

    async def register_uploaded_document(
        self,
        *,
        user_id: int,
        title: str,
        file_name: str,
        mime_type: str,
        content: bytes,
    ) -> Document:
        """
        Register an uploaded document after checking for duplicate content.

        A SHA-256 checksum is generated from the uploaded bytes and compared
        only against documents owned by the same user. Existing documents are
        rejected according to their current ingestion status.

        Args:
            user_id: Identifier of the user who owns the document.
            title: Human-readable document title.
            file_name: Original uploaded file name.
            mime_type: MIME type associated with the uploaded file.
            content: Raw uploaded file content.

        Returns:
            Newly registered document in the initial pending state, or the
            existing failed document reserved for a new ingestion attempt.

        Raises:
            ValueError: If the same document content is already ready or is
                currently being processed.
        """

        checksum = self._checksum.sha256(content)

        existing_document = await self._documents.get_by_checksum(
            checksum=checksum,
            user_id=user_id,
        )

        if existing_document is not None:
            if existing_document.status == "ready":
                raise ValueError(
                    f"Document content already exists "
                    f"(document_id={existing_document.id})"
                )

            if existing_document.status in {"pending", "processing"}:
                raise ValueError(
                    f"Document content is already being processed "
                    f"(document_id={existing_document.id})"
                )

            if existing_document.status == "failed":
                latest_attempt = (
                    await self._status_history.get_latest_attempt_number(
                        existing_document.id
                    )
                    or 1
                )
                next_attempt = latest_attempt + 1

                claimed_document = await self._documents.claim_failed_for_retry(
                    document_id=existing_document.id,
                    user_id=user_id,
                    title=title,
                    file_name=file_name,
                    mime_type=mime_type,
                )

                if claimed_document is None:
                    raise ValueError(
                        f"Document content is already being processed "
                        f"(document_id={existing_document.id})"
                    )

                await self._status_history.create(
                    document_id=claimed_document.id,
                    from_status="failed",
                    to_status="pending",
                    component="document_service",
                    operation="retry_uploaded_document",
                    attempt_number=next_attempt,
                )

                await self._session.commit()
                await self._session.refresh(claimed_document)

                return claimed_document

        return await self.register_document(
            user_id=user_id,
            title=title,
            source_type="upload",
            file_name=file_name,
            mime_type=mime_type,
            checksum=checksum,
        )

    async def register_document(
        self,
        *,
        user_id: int,
        title: str,
        source_type: str,
        source_url: str | None = None,
        file_name: str | None = None,
        mime_type: str | None = None,
        checksum: str | None = None,
    ) -> Document:
        """
        Register a new document and record its initial status.

        The document and its initial ``pending`` status-history event are
        committed in the same transaction.

        Args:
            user_id: Identifier of the document owner.
            title: Human-readable document title.
            source_type: Origin of the document, such as ``upload`` or ``url``.
            source_url: Optional source URL for URL-based documents.
            file_name: Optional original uploaded file name.
            mime_type: Optional document MIME type.
            checksum: Optional SHA-256 checksum of the document content.

        Returns:
            Newly created document.

        Raises:
            SQLAlchemyError: If persistence of the document or status history
                fails.
        """

        document = await self._documents.create(
            user_id=user_id,
            title=title,
            source_type=source_type,
            source_url=source_url,
            file_name=file_name,
            mime_type=mime_type,
            checksum=checksum,
        )

        await self._status_history.create(
            document_id=document.id,
            from_status=None,
            to_status="pending",
            component="document_service",
            operation="register_document",
        )

        await self._session.commit()
        await self._session.refresh(document)

        return document

    async def update_status(
        self,
        document_id: int,
        user_id: int,
        status: str,
    ) -> Document:
        """
        Update a document status and record the transition.

        The current document status is captured before the update so the
        corresponding status-history event accurately records the transition.
        Both mutations are committed together.

        Args:
            document_id: Identifier of the document to update.
            user_id: Identifier of the document owner.
            status: New document status.

        Returns:
            Updated document.

        Raises:
            ValueError: If the document does not exist for the specified user.
        """

        document = await self._documents.get_by_id(
            document_id=document_id,
            user_id=user_id,
        )

        if document is None:
            raise ValueError(f"Document {document_id} not found")

        from_status = document.status

        await self._documents.update_status(
            document=document,
            status=status,
        )

        await self._status_history.create(
            document_id=document.id,
            from_status=from_status,
            to_status=status,
            component="document_service",
            operation="update_status",
        )

        await self._session.commit()
        await self._session.refresh(document)

        return document

    async def delete_document(
        self,
        *,
        document_id: int,
        user_id: int,
    ) -> None:
        """
        Delete a document owned by a specific user.

        Args:
            document_id: Identifier of the document to delete.
            user_id: Identifier of the document owner.

        Raises:
            ValueError: If the document does not exist for the user.
        """

        document = await self._documents.get_by_id(
            document_id=document_id,
            user_id=user_id,
        )

        if document is None:
            raise ValueError("Document not found")

        await self._documents.delete(document)
        await self._session.commit()

    async def delete_documents(
        self,
        *,
        document_ids: list[int],
        user_id: int,
    ) -> None:
        """
        Delete multiple documents owned by a specific user.

        Duplicate identifiers are removed before validation. All requested
        documents are validated before any deletion occurs so the operation
        behaves atomically from the service perspective.

        Args:
            document_ids: Identifiers of the documents to delete.
            user_id: Identifier of the document owner.

        Raises:
            ValueError: If no documents are selected or one or more requested
                documents do not exist for the specified user.
        """

        if not document_ids:
            raise ValueError("At least one document must be selected")

        unique_document_ids = list(dict.fromkeys(document_ids))

        documents = await self._documents.list_by_ids(
            user_id=user_id,
            document_ids=unique_document_ids,
        )

        if len(documents) != len(unique_document_ids):
            raise ValueError("One or more documents not found")

        for document in documents:
            await self._documents.delete(document)

        await self._session.commit()
