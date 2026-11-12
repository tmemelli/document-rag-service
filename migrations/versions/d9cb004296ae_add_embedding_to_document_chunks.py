"""
Add vector embeddings to document chunks.

Revision ID: d9cb004296ae
Revises: fd0ffc3cd1f1
Create Date: 2026-09-28 20:53:39.503595
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "d9cb004296ae"
down_revision: str | Sequence[str] | None = "fd0ffc3cd1f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """
    Enable pgvector and add the initial embedding column.
    """

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "document_chunks",
        sa.Column(
            "embedding",
            Vector(384),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """
    Remove the embedding column.

    The vector extension is intentionally preserved because PostgreSQL
    extensions are database-level resources that may be used elsewhere.
    """

    op.drop_column(
        "document_chunks",
        "embedding",
    )
