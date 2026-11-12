"""cascade document dependent records on delete

Revision ID: bc395eb78bce
Revises: c2e65353e94c
Create Date: 2026-11-10 14:04:53.101470

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bc395eb78bce"
down_revision: str | Sequence[str] | None = "c2e65353e94c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""

    op.alter_column(
        "document_status_history",
        "document_id",
        existing_type=sa.BIGINT(),
        nullable=False,
    )

    op.drop_constraint(
        "document_status_history_document_id_fkey",
        "document_status_history",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "document_status_history_document_id_fkey",
        "document_status_history",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.drop_constraint(
        "error_events_document_id_fkey",
        "error_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "error_events_document_id_fkey",
        "error_events",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_constraint(
        "error_events_document_id_fkey",
        "error_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "error_events_document_id_fkey",
        "error_events",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.drop_constraint(
        "document_status_history_document_id_fkey",
        "document_status_history",
        type_="foreignkey",
    )

    op.alter_column(
        "document_status_history",
        "document_id",
        existing_type=sa.BIGINT(),
        nullable=True,
    )

    op.create_foreign_key(
        "document_status_history_document_id_fkey",
        "document_status_history",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="SET NULL",
    )
