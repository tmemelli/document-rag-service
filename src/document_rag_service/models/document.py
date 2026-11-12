"""SQLAlchemy model for ingested documents and their processing lifecycle."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from document_rag_service.db.base import Base


class Document(Base):
    """
    Persist document metadata, ownership, source information, and processing state.

    Each document belongs to a single application user and represents one item
    available to the ingestion and RAG pipelines. The model tracks where the
    document originated, optional source metadata, duplicate-detection checksum,
    current processing status, and audit timestamps.

    Database constraints restrict source types and processing states to values
    supported by the application lifecycle.
    """

    __tablename__ = "documents"

    __table_args__ = (
        CheckConstraint(
            "source_type IN ('upload', 'url')",
            name="ck_documents_source_type",
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_documents_status",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(
            "users.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="upload",
    )

    source_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    file_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(127),
        nullable=True,
    )

    checksum: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="pending",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
