"""SQLAlchemy declarative base for application ORM models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Serve as the declarative base for all SQLAlchemy ORM models.

    Application models inherit from this class so they share a common metadata
    registry used by SQLAlchemy and Alembic for schema definition and migration
    discovery.
    """
