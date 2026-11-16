"""
Tests for database URL, engine, and async session factory construction.
"""

from unittest.mock import MagicMock

from pydantic import SecretStr
from sqlalchemy import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
)

from document_rag_service.core.settings import Settings
from document_rag_service.db import session as session_module
from document_rag_service.db.session import (
    create_database_engine,
    create_database_url,
    create_session_factory,
)


def _build_settings() -> MagicMock:
    """Create deterministic database settings for session tests."""
    settings = MagicMock(spec=Settings)

    settings.postgres_host = "db.example.com"
    settings.postgres_port = 5432
    settings.postgres_db = "document_rag_db"
    settings.postgres_user = "document_rag_user"
    settings.postgres_password = SecretStr("super-secret-password")
    settings.postgres_sslmode = "require"

    return settings


def test_create_database_url_returns_sqlalchemy_url() -> None:
    """Build a SQLAlchemy URL object for PostgreSQL."""
    settings = _build_settings()

    result = create_database_url(settings)

    assert isinstance(result, URL)


def test_create_database_url_uses_async_psycopg_driver() -> None:
    """Use the asynchronous psycopg SQLAlchemy driver."""
    settings = _build_settings()

    result = create_database_url(settings)

    assert result.drivername == "postgresql+psycopg"


def test_create_database_url_uses_database_credentials() -> None:
    """Populate the database URL from validated settings."""
    settings = _build_settings()

    result = create_database_url(settings)

    assert result.username == "document_rag_user"
    assert result.password == "super-secret-password"
    assert result.host == "db.example.com"
    assert result.port == 5432
    assert result.database == "document_rag_db"


def test_create_database_url_extracts_secret_password_value() -> None:
    """Use the underlying SecretStr value when building the connection URL."""
    settings = _build_settings()

    settings.postgres_password = SecretStr("actual-database-password")

    result = create_database_url(settings)

    assert result.password == "actual-database-password"


def test_create_database_url_includes_sslmode() -> None:
    """Forward the configured PostgreSQL SSL mode into the connection query."""
    settings = _build_settings()

    settings.postgres_sslmode = "verify-full"

    result = create_database_url(settings)

    assert result.query == {
        "sslmode": "verify-full",
    }


def test_create_database_url_supports_disable_sslmode() -> None:
    """Preserve an explicitly disabled PostgreSQL SSL mode."""
    settings = _build_settings()

    settings.postgres_sslmode = "disable"

    result = create_database_url(settings)

    assert result.query == {
        "sslmode": "disable",
    }


def test_create_database_url_does_not_expose_password_in_string() -> None:
    """Keep credentials hidden when the SQLAlchemy URL is rendered normally."""
    settings = _build_settings()

    settings.postgres_password = SecretStr("sensitive-password")

    result = create_database_url(settings)

    rendered = str(result)

    assert "sensitive-password" not in rendered
    assert "***" in rendered


def test_create_database_engine_uses_generated_database_url(
    monkeypatch,
) -> None:
    """Create the asynchronous engine from the generated database URL."""
    settings = _build_settings()

    expected_url = URL.create(
        drivername="postgresql+psycopg",
        username="user",
        password="password",
        host="host",
        port=5432,
        database="database",
    )

    create_url = MagicMock(
        return_value=expected_url,
    )

    engine = MagicMock(spec=AsyncEngine)

    create_engine = MagicMock(
        return_value=engine,
    )

    monkeypatch.setattr(
        session_module,
        "create_database_url",
        create_url,
    )

    monkeypatch.setattr(
        session_module,
        "create_async_engine",
        create_engine,
    )

    result = create_database_engine(settings)

    create_url.assert_called_once_with(settings)

    create_engine.assert_called_once_with(
        expected_url,
        pool_pre_ping=True,
    )

    assert result is engine


def test_create_database_engine_enables_pool_pre_ping(
    monkeypatch,
) -> None:
    """Enable connection health checks before pooled connections are reused."""
    settings = _build_settings()

    engine = MagicMock(spec=AsyncEngine)

    create_engine = MagicMock(
        return_value=engine,
    )

    monkeypatch.setattr(
        session_module,
        "create_async_engine",
        create_engine,
    )

    create_database_engine(settings)

    assert create_engine.call_args.kwargs["pool_pre_ping"] is True


def test_create_session_factory_uses_engine(
    monkeypatch,
) -> None:
    """Bind generated asynchronous sessions to the supplied engine."""
    engine = MagicMock(spec=AsyncEngine)

    factory = MagicMock()

    async_sessionmaker = MagicMock(
        return_value=factory,
    )

    monkeypatch.setattr(
        session_module,
        "async_sessionmaker",
        async_sessionmaker,
    )

    result = create_session_factory(engine)

    async_sessionmaker.assert_called_once_with(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    assert result is factory


def test_create_session_factory_disables_expiration_on_commit(
    monkeypatch,
) -> None:
    """Keep ORM instances usable after transaction commits."""
    engine = MagicMock(spec=AsyncEngine)

    async_sessionmaker = MagicMock()

    monkeypatch.setattr(
        session_module,
        "async_sessionmaker",
        async_sessionmaker,
    )

    create_session_factory(engine)

    assert async_sessionmaker.call_args.kwargs["expire_on_commit"] is False


def test_create_session_factory_uses_async_session_class(
    monkeypatch,
) -> None:
    """Configure generated sessions with SQLAlchemy's AsyncSession class."""
    engine = MagicMock(spec=AsyncEngine)

    async_sessionmaker = MagicMock()

    monkeypatch.setattr(
        session_module,
        "async_sessionmaker",
        async_sessionmaker,
    )

    create_session_factory(engine)

    assert async_sessionmaker.call_args.kwargs["class_"] is AsyncSession
