"""add document processing leases

Revision ID: 4f8c1a2d9b73
Revises: bc395eb78bce
Create Date: 2026-11-18 10:55:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4f8c1a2d9b73"
down_revision: str | Sequence[str] | None = "bc395eb78bce"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add lease metadata used to recover interrupted document ingestion."""

    op.add_column(
        "documents",
        sa.Column(
            "processing_token",
            sa.String(length=32),
            nullable=True,
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "processing_lease_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    op.create_check_constraint(
        "ck_documents_processing_lease_status",
        "documents",
        (
            "status = 'processing' OR "
            "(processing_token IS NULL AND "
            "processing_lease_expires_at IS NULL)"
        ),
    )
    op.create_check_constraint(
        "ck_documents_processing_lease_pair",
        "documents",
        (
            "(processing_token IS NULL AND "
            "processing_lease_expires_at IS NULL) OR "
            "(processing_token IS NOT NULL AND "
            "processing_lease_expires_at IS NOT NULL)"
        ),
    )


def downgrade() -> None:
    """Remove document processing-lease metadata."""

    op.drop_constraint(
        "ck_documents_processing_lease_pair",
        "documents",
        type_="check",
    )
    op.drop_constraint(
        "ck_documents_processing_lease_status",
        "documents",
        type_="check",
    )
    op.drop_column(
        "documents",
        "processing_lease_expires_at",
    )
    op.drop_column(
        "documents",
        "processing_token",
    )
