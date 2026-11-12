"""SQLAlchemy model for indexed document chunks and vector embeddings."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from document_rag_service.db.base import Base


class DocumentChunk(Base):
    """
    Persist an indexed document fragment and its semantic vector representation.

    Each chunk belongs to one ingested document, preserves its position through
    a zero-based ``chunk_index``, and stores both the normalized text content
    and the token count used during ingestion.

    The embedding column uses pgvector with the fixed dimension expected by the
    retrieval pipeline, enabling cosine-distance similarity search directly in
    PostgreSQL. Database constraints protect chunk ordering, token-count
    validity, and uniqueness of each chunk position within a document.
    """

    __tablename__ = "document_chunks"

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_index",
        ),
        CheckConstraint(
            "chunk_index >= 0",
            name="ck_document_chunks_chunk_index",
        ),
        CheckConstraint(
            "token_count >= 0",
            name="ck_document_chunks_token_count",
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

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    token_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0",
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(512),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
