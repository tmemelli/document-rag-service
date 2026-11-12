"""SQLAlchemy model for auditable document processing status transitions."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from document_rag_service.db.base import Base


class DocumentStatusHistory(Base):
    """
    Persist the processing-status timeline of an ingested document.

    Each record represents one transition in the document lifecycle and stores
    both the previous and resulting status together with the application
    component and operation responsible for the change.

    The model also supports processing-attempt tracking and structured metadata
    so status transitions remain auditable and useful for debugging ingestion
    failures or future retry workflows. Database constraints ensure that only
    valid document states and positive attempt numbers are persisted.
    """

    __tablename__ = "document_status_history"

    __table_args__ = (
        CheckConstraint(
            "from_status IS NULL OR "
            "from_status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_document_status_history_from_status",
        ),
        CheckConstraint(
            "to_status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_document_status_history_to_status",
        ),
        CheckConstraint(
            "attempt_number > 0",
            name="ck_document_status_history_attempt_number",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )

    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    from_status: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    to_status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    component: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    operation: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    attempt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="1",
    )

    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
