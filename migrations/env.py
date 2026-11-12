"""
Configure the Alembic migration environment for the Document RAG database.

The migration environment reuses the application's validated database
configuration and SQLAlchemy metadata so schema migrations and autogeneration
remain aligned with the runtime models.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

import document_rag_service.models  # noqa: F401
from document_rag_service.core.settings import get_settings
from document_rag_service.db.base import Base
from document_rag_service.db.session import create_database_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

settings = get_settings()
database_url = create_database_url(settings)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations without establishing a live database connection.

    Alembic renders SQL directly from the application's configured database URL
    and SQLAlchemy metadata. Literal values are embedded in generated SQL so
    the output can be executed independently.

    The same type and server-default comparison rules used by online
    migrations are enabled for consistent autogeneration behavior.
    """

    context.configure(
        url=database_url.render_as_string(
            hide_password=False,
        ),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={
            "paramstyle": "named",
        },
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations against a live database connection.

    A synchronous SQLAlchemy engine is created specifically for Alembic using
    the same PostgreSQL connection URL as the asynchronous application runtime.
    ``NullPool`` prevents migration execution from maintaining a persistent
    connection pool.

    The engine is always disposed after migration execution completes.
    """

    connectable = create_engine(
        database_url,
        poolclass=pool.NullPool,
    )

    try:
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                compare_server_default=True,
            )

            with context.begin_transaction():
                context.run_migrations()
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
