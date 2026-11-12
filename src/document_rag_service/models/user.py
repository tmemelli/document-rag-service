"""SQLAlchemy model for application users and authentication account state."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Identity,
    String,
    Text,
    func,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column

from document_rag_service.db.base import Base


class User(Base):
    """
    Persist application user credentials and account state.

    Each user owns documents and authenticates with a unique email address and
    securely stored password hash. Account activity and audit timestamps are
    maintained at the database level through server-side defaults and update
    hooks.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )

    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
        unique=True,
    )

    password_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=true(),
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
