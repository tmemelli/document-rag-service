"""remove user email normalization constraint

Revision ID: 3a5c98e4cc4d
Revises: e090211e13b8
Create Date: 2026-10-01 08:44:06.077019

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3a5c98e4cc4d"
down_revision: str | Sequence[str] | None = "e090211e13b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint(
        "ck_users_email_normalized",
        "users",
        type_="check",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.create_check_constraint(
        "ck_users_email_normalized",
        "users",
        "email = lower(btrim(email))",
    )
