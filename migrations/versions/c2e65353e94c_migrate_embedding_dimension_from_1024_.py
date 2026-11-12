"""migrate embedding dimension from 1024 to 512

Revision ID: c2e65353e94c
Revises: b52549f62e58
Create Date: 2026-11-02 15:55:27.479619

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "c2e65353e94c"
down_revision: str | Sequence[str] | None = "b52549f62e58"
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
            Vector(512),
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
            Vector(1024),
            nullable=False,
        ),
    )
