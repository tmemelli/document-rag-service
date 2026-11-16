"""
Shared PostgreSQL fixtures for integration tests.
"""

from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.core.settings import get_settings
from document_rag_service.db.session import (
    create_database_engine,
    create_session_factory,
)

TEST_DATABASE_NAME = "document_rag_test_db"
LOCAL_DATABASE_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
}


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Provide an isolated transaction against the local integration database."""
    settings = get_settings()

    if settings.postgres_host not in LOCAL_DATABASE_HOSTS:
        raise RuntimeError(
            "Integration tests may only run against a local PostgreSQL database"
        )

    test_settings = settings.model_copy(
        update={
            "postgres_db": TEST_DATABASE_NAME,
        }
    )

    engine = create_database_engine(test_settings)
    session_factory = create_session_factory(engine)

    connection = await engine.connect()
    transaction = await connection.begin()

    session = session_factory(
        bind=connection,
        join_transaction_mode="create_savepoint",
    )

    try:
        yield session
    finally:
        await session.close()

        if transaction.is_active:
            await transaction.rollback()

        await connection.close()
        await engine.dispose()
