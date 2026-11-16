"""
Integration tests for UserRepository against PostgreSQL.
"""

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.repositories.user_repository import (
    UserRepository,
)


@pytest.mark.asyncio
async def test_create_persists_user(
    db_session: AsyncSession,
) -> None:
    """Persist a user and populate database-generated fields."""
    repository = UserRepository(db_session)

    user = await repository.create(
        email="create@example.com",
        password_hash="hashed-password",
    )

    assert user.id > 0
    assert user.email == "create@example.com"
    assert user.password_hash == "hashed-password"
    assert user.is_active is True
    assert user.created_at is not None
    assert user.updated_at is not None


@pytest.mark.asyncio
async def test_create_user_is_retrievable_from_database(
    db_session: AsyncSession,
) -> None:
    """Verify a repository-created user exists in PostgreSQL."""
    repository = UserRepository(db_session)

    created = await repository.create(
        email="persisted@example.com",
        password_hash="hash",
    )

    user_id = created.id

    db_session.expire(created)

    retrieved = await repository.get_by_id(user_id)

    assert retrieved is not None
    assert retrieved.id == user_id
    assert retrieved.email == "persisted@example.com"


@pytest.mark.asyncio
async def test_get_by_id_returns_existing_user(
    db_session: AsyncSession,
) -> None:
    """Retrieve an existing user by database identifier."""
    repository = UserRepository(db_session)

    created = await repository.create(
        email="by-id@example.com",
        password_hash="hash",
    )

    result = await repository.get_by_id(created.id)

    assert result is not None
    assert result.id == created.id
    assert result.email == "by-id@example.com"


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_unknown_user(
    db_session: AsyncSession,
) -> None:
    """Return None when the requested identifier does not exist."""
    repository = UserRepository(db_session)

    result = await repository.get_by_id(9_999_999_999)

    assert result is None


@pytest.mark.asyncio
async def test_get_by_email_returns_existing_user(
    db_session: AsyncSession,
) -> None:
    """Retrieve an existing user by email address."""
    repository = UserRepository(db_session)

    created = await repository.create(
        email="lookup@example.com",
        password_hash="hash",
    )

    result = await repository.get_by_email("lookup@example.com")

    assert result is not None
    assert result.id == created.id
    assert result.email == "lookup@example.com"


@pytest.mark.asyncio
async def test_get_by_email_returns_none_for_unknown_email(
    db_session: AsyncSession,
) -> None:
    """Return None when no user has the requested email."""
    repository = UserRepository(db_session)

    result = await repository.get_by_email("missing@example.com")

    assert result is None


@pytest.mark.asyncio
async def test_database_enforces_unique_email(
    db_session: AsyncSession,
) -> None:
    """Verify PostgreSQL rejects duplicate user email addresses."""
    repository = UserRepository(db_session)

    await repository.create(
        email="duplicate@example.com",
        password_hash="first-hash",
    )

    async with db_session.begin_nested():
        with pytest.raises(IntegrityError):
            await repository.create(
                email="duplicate@example.com",
                password_hash="second-hash",
            )


@pytest.mark.asyncio
async def test_database_generates_distinct_user_ids(
    db_session: AsyncSession,
) -> None:
    """Generate a distinct identity value for every persisted user."""
    repository = UserRepository(db_session)

    first = await repository.create(
        email="first@example.com",
        password_hash="hash",
    )

    second = await repository.create(
        email="second@example.com",
        password_hash="hash",
    )

    assert first.id != second.id
    assert first.id > 0
    assert second.id > 0


@pytest.mark.asyncio
async def test_email_lookup_is_exact(
    db_session: AsyncSession,
) -> None:
    """Preserve PostgreSQL's exact email lookup semantics."""
    repository = UserRepository(db_session)

    await repository.create(
        email="CaseSensitive@example.com",
        password_hash="hash",
    )

    exact = await repository.get_by_email("CaseSensitive@example.com")

    different_case = await repository.get_by_email("casesensitive@example.com")

    assert exact is not None
    assert different_case is None
