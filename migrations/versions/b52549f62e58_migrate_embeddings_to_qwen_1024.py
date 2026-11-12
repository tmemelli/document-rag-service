"""migrate embeddings to qwen 1024

Revision ID: b52549f62e58
Revises: 3a5c98e4cc4d
Create Date: 2026-11-02 13:40:52.111717

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "b52549f62e58"
down_revision: str | Sequence[str] | None = "3a5c98e4cc4d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column(
        "document_chunks",
        "embedding",
    )

    op.add_column(
        "document_chunks",
        sa.Column(
            "embedding",
            Vector(1024),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "document_chunks",
        "embedding",
    )

    op.add_column(
        "document_chunks",
        sa.Column(
            "embedding",
            Vector(384),
            nullable=False,
        ),
    )
