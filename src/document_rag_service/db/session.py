"""
Database engine, connection URL, and asynchronous session factory construction.
"""

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from document_rag_service.core.settings import Settings

AsyncSessionFactory = async_sessionmaker[AsyncSession]


def create_database_url(
    settings: Settings,
) -> URL:
    """
    Build the PostgreSQL connection URL from validated application settings.

    Connection details are assembled through SQLAlchemy's ``URL`` object rather
    than manual string interpolation, keeping credential handling explicit and
    ensuring the configured PostgreSQL SSL mode is propagated consistently.

    Args:
        settings: Validated application settings containing database connection
            parameters and credentials.

    Returns:
        SQLAlchemy URL configured for PostgreSQL through the asynchronous
        psycopg driver.
    """

    return URL.create(
        drivername="postgresql+psycopg",
        username=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        query={"sslmode": settings.postgres_sslmode},
    )


def create_database_engine(
    settings: Settings,
) -> AsyncEngine:
    """
    Create the application's asynchronous SQLAlchemy database engine.

    The engine is configured with connection pre-ping so stale pooled
    connections are detected before use, improving resilience for long-lived
    application processes and externally hosted PostgreSQL databases.

    Args:
        settings: Validated application settings used to construct the database
            connection URL.

    Returns:
        Configured asynchronous SQLAlchemy engine.
    """

    database_url = create_database_url(settings)

    return create_async_engine(
        database_url,
        pool_pre_ping=True,
    )


def create_session_factory(
    engine: AsyncEngine,
) -> AsyncSessionFactory:
    """
    Create the asynchronous SQLAlchemy session factory.

    Sessions produced by the factory remain usable after commits because
    ``expire_on_commit`` is disabled. Transaction boundaries are controlled by
    the service layer rather than by the session factory itself.

    Args:
        engine: Asynchronous database engine used by generated sessions.

    Returns:
        Configured asynchronous session factory producing ``AsyncSession``
        instances.
    """

    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
