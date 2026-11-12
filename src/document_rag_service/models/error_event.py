"""SQLAlchemy model for structured application error events."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    String,
    Text,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from document_rag_service.db.base import Base


class ErrorEvent(Base):
    """
    Persist structured application errors for operational observability.

    Error events capture where a failure occurred, which operation was running,
    how the error was classified, whether it may be retried, and any structured
    diagnostic metadata available at the time of failure.

    Events may optionally be associated with a document and correlation
    identifier so failures can be traced across ingestion attempts and related
    application operations.
    """

    __tablename__ = "error_events"

    __table_args__ = (
        CheckConstraint(
            "attempt_number > 0",
            name="ck_error_events_attempt_number",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )

    document_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=True,
        index=True,
    )

    component: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    operation: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    error_type: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    retryable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=false(),
    )

    attempt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="1",
    )

    correlation_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
