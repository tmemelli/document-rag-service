"""Repository for document processing status-history persistence and lookup."""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document_status_history import (
    DocumentStatusHistory,
)


class DocumentStatusHistoryRepository:
    """
    Provide persistence operations for document processing status history.

    Status-history records capture document state transitions together with the
    component and operation responsible for each change. The repository keeps
    persistence concerns isolated while leaving transaction boundaries to the
    calling service.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the document status-history repository.

        Args:
            session: Asynchronous SQLAlchemy session used for persistence
                operations.
        """

        self._session = session

    async def create(
        self,
        *,
        document_id: int,
        from_status: str | None,
        to_status: str,
        component: str,
        operation: str,
        attempt_number: int = 1,
        details: dict[str, Any] | None = None,
    ) -> DocumentStatusHistory:
        """
        Create and persist a document status transition.

        The transition is flushed and refreshed so database-generated values
        are immediately available without committing the surrounding
        transaction.

        Args:
            document_id: Identifier of the document whose status changed.
            from_status: Previous document status, or ``None`` when recording
                the initial state.
            to_status: New document status after the transition.
            component: Application component responsible for the transition.
            operation: Operation that caused the status change.
            attempt_number: Processing attempt associated with the transition.
            details: Optional structured metadata describing the transition.

        Returns:
            Newly created and refreshed status-history entity.
        """

        history = DocumentStatusHistory(
            document_id=document_id,
            from_status=from_status,
            to_status=to_status,
            component=component,
            operation=operation,
            attempt_number=attempt_number,
            details=details or {},
        )

        self._session.add(history)
        await self._session.flush()
        await self._session.refresh(history)

        return history

    async def get_latest_attempt_number(
        self,
        document_id: int,
    ) -> int | None:
        """
        Return the highest processing-attempt number recorded for a document.

        Args:
            document_id: Identifier of the document whose attempts should be
                inspected.

        Returns:
            Highest persisted attempt number, or ``None`` when the document
            has no status-history records.
        """

        statement = select(
            func.max(DocumentStatusHistory.attempt_number)
        ).where(
            DocumentStatusHistory.document_id == document_id
        )

        return await self._session.scalar(statement)

    async def list_by_document_id(
        self,
        document_id: int,
    ) -> list[DocumentStatusHistory]:
        """
        Return the complete processing status timeline for a document.

        Transitions are ordered by creation time and then by primary-key
        identifier to guarantee deterministic ordering when multiple records
        share the same timestamp.

        Args:
            document_id: Identifier of the document whose status history should
                be retrieved.

        Returns:
            Status-history records in deterministic chronological order.
        """

        statement = (
            select(DocumentStatusHistory)
            .where(DocumentStatusHistory.document_id == document_id)
            .order_by(
                DocumentStatusHistory.created_at,
                DocumentStatusHistory.id,
            )
        )

        result = await self._session.scalars(statement)

        return list(result.all())
