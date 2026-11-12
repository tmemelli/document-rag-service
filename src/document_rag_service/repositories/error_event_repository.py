"""Repository for structured application error-event persistence and lookup."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.error_event import ErrorEvent


class ErrorEventRepository:
    """
    Provide persistence operations for structured application error events.

    Error events capture operational failures in a queryable form, including
    the affected document, component, operation, error classification, retry
    metadata, diagnostic details, and optional correlation identifiers.

    The repository handles persistence only. Transaction boundaries remain the
    responsibility of the calling service so error records can participate in
    the same transaction as related application state changes when appropriate.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the error-event repository.

        Args:
            session: Asynchronous SQLAlchemy session used for persistence
                operations.
        """

        self._session = session

    async def create(
        self,
        *,
        document_id: int | None,
        component: str,
        operation: str,
        error_type: str,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
        attempt_number: int = 1,
        correlation_id: str | None = None,
    ) -> ErrorEvent:
        """
        Create and persist a structured application error event.

        The event is flushed and refreshed so database-generated values are
        available to the caller without committing the surrounding transaction.

        Args:
            document_id: Optional identifier of the document associated with
                the failure.
            component: Application component where the error occurred.
            operation: Operation being executed when the error occurred.
            error_type: Logical or technical classification of the error.
            message: Human-readable description of the failure.
            error_code: Optional stable error code for programmatic analysis.
            details: Optional structured diagnostic metadata associated with
                the failure.
            retryable: Whether the failed operation may be safely retried.
            attempt_number: Attempt number on which the error occurred.
            correlation_id: Optional identifier used to correlate the error
                with related operations or logs.

        Returns:
            Newly created and refreshed error-event entity.
        """

        error_event = ErrorEvent(
            document_id=document_id,
            component=component,
            operation=operation,
            error_type=error_type,
            error_code=error_code,
            message=message,
            details=details or {},
            retryable=retryable,
            attempt_number=attempt_number,
            correlation_id=correlation_id,
        )

        self._session.add(error_event)
        await self._session.flush()
        await self._session.refresh(error_event)

        return error_event

    async def list_by_document_id(
        self,
        document_id: int,
    ) -> list[ErrorEvent]:
        """
        Return all error events associated with a document.

        Events are ordered chronologically by occurrence time and then by
        primary-key identifier to provide deterministic ordering when multiple
        events share the same timestamp.

        Args:
            document_id: Identifier of the document whose errors should be
                retrieved.

        Returns:
            Error events associated with the document in deterministic
            chronological order.
        """

        statement = (
            select(ErrorEvent)
            .where(ErrorEvent.document_id == document_id)
            .order_by(
                ErrorEvent.occurred_at,
                ErrorEvent.id,
            )
        )

        result = await self._session.scalars(statement)

        return list(result.all())
