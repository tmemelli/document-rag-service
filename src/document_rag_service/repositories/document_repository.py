"""Repository for document persistence and user-scoped lookup operations."""

from sqlalchemy import select
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
