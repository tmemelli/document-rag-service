"""
Integration tests for UserService against PostgreSQL.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.repositories.user_repository import UserRepository
from document_rag_service.services.user_service import UserService


@pytest.mark.asyncio
async def test_register_user_persists_normalized_user(
    db_session: AsyncSession,
) -> None:
    """Register and persist a normalized user with a hashed password."""
    service = UserService(db_session)

    user = await service.register_user(
        email="  Integration.User@Example.COM  ",
        password="strong-password-123",
    )

    assert user.id > 0
    assert user.email == "Integration.User@example.com"
    assert user.password_hash != "strong-password-123"
    assert user.is_active is True
    assert user.created_at is not None
    assert user.updated_at is not None

    repository = UserRepository(db_session)

    persisted = await repository.get_by_email("Integration.User@example.com")

    assert persisted is not None
    assert persisted.id == user.id


@pytest.mark.asyncio
async def test_register_user_rejects_duplicate_email(
    db_session: AsyncSession,
) -> None:
    """Reject registration when the normalized email already exists."""
    service = UserService(db_session)

    await service.register_user(
        email="duplicate@example.com",
        password="first-password",
    )

    with pytest.raises(
        ValueError,
        match="Email is already registered",
    ):
        await service.register_user(
            email="duplicate@example.com",
            password="second-password",
        )


@pytest.mark.asyncio
async def test_register_user_rejects_invalid_email(
    db_session: AsyncSession,
) -> None:
    """Reject syntactically invalid email addresses."""
    service = UserService(db_session)

    with pytest.raises(
        ValueError,
        match="Invalid email",
    ):
        await service.register_user(
            email="not-an-email",
            password="password",
        )


@pytest.mark.asyncio
async def test_register_user_rejects_empty_password(
    db_session: AsyncSession,
) -> None:
    """Reject empty passwords before persistence."""
    service = UserService(db_session)

    with pytest.raises(
        ValueError,
        match="Password cannot be empty",
    ):
        await service.register_user(
            email="empty-password@example.com",
            password="",
        )

    repository = UserRepository(db_session)

    persisted = await repository.get_by_email("empty-password@example.com")

    assert persisted is None


@pytest.mark.asyncio
async def test_authenticate_user_returns_persisted_user(
    db_session: AsyncSession,
) -> None:
    """Authenticate a persisted user with the correct password."""
    service = UserService(db_session)

    registered = await service.register_user(
        email="authenticate@example.com",
        password="correct-password",
    )

    authenticated = await service.authenticate_user(
        email="authenticate@example.com",
        password="correct-password",
    )

    assert authenticated.id == registered.id
    assert authenticated.email == "authenticate@example.com"


@pytest.mark.asyncio
async def test_authenticate_user_normalizes_email(
    db_session: AsyncSession,
) -> None:
    """Normalize email input during authentication."""
    service = UserService(db_session)

    registered = await service.register_user(
        email="Normalized.Login@example.com",
        password="correct-password",
    )

    authenticated = await service.authenticate_user(
        email="  Normalized.Login@EXAMPLE.COM  ",
        password="correct-password",
    )

    assert authenticated.id == registered.id


@pytest.mark.asyncio
async def test_authenticate_user_rejects_unknown_email(
    db_session: AsyncSession,
) -> None:
    """Reject authentication for a user that does not exist."""
    service = UserService(db_session)

    with pytest.raises(
        ValueError,
        match="Invalid email or password",
    ):
        await service.authenticate_user(
            email="missing@example.com",
            password="password",
        )


@pytest.mark.asyncio
async def test_authenticate_user_rejects_invalid_email(
    db_session: AsyncSession,
) -> None:
    """Expose the generic authentication error for malformed email input."""
    service = UserService(db_session)

    with pytest.raises(
        ValueError,
        match="Invalid email or password",
    ):
        await service.authenticate_user(
            email="invalid-email",
            password="password",
        )


@pytest.mark.asyncio
async def test_authenticate_user_rejects_wrong_password(
    db_session: AsyncSession,
) -> None:
    """Reject authentication when the password does not match."""
    service = UserService(db_session)

    await service.register_user(
        email="wrong-password@example.com",
        password="correct-password",
    )

    with pytest.raises(
        ValueError,
        match="Invalid email or password",
    ):
        await service.authenticate_user(
            email="wrong-password@example.com",
            password="wrong-password",
        )


@pytest.mark.asyncio
async def test_authenticate_user_rejects_inactive_user(
    db_session: AsyncSession,
) -> None:
    """Reject authentication for an inactive persisted account."""
    service = UserService(db_session)

    user = await service.register_user(
        email="inactive@example.com",
        password="correct-password",
    )

    user.is_active = False
    await db_session.flush()

    with pytest.raises(
        ValueError,
        match="User account is inactive",
    ):
        await service.authenticate_user(
            email="inactive@example.com",
            password="correct-password",
        )
