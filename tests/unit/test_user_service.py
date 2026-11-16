"""
Tests for user registration, authentication, and account deletion.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.models.user import User
from document_rag_service.services.user_service import UserService


def _build_user(
    *,
    user_id: int = 42,
    email: str = "user@example.com",
    password_hash: str = "stored-password-hash",
    is_active: bool = True,
) -> User:
    """Create an in-memory user for service tests."""
    user = User(
        email=email,
        password_hash=password_hash,
    )
    user.id = user_id
    user.is_active = is_active
    return user


def _build_document(
    *,
    document_id: int,
    user_id: int = 42,
) -> Document:
    """Create an in-memory document owned by a test user."""
    document = Document(
        user_id=user_id,
        title=f"Document {document_id}",
        source_type="upload",
        status="ready",
    )
    document.id = document_id
    return document


def _build_service() -> tuple[
    UserService,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    """Create a user service with isolated persistence and password doubles."""
    session = MagicMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()

    user_repository = MagicMock()
    user_repository.get_by_email = AsyncMock()
    user_repository.get_by_id = AsyncMock()
    user_repository.create = AsyncMock()
    user_repository.delete = AsyncMock()

    document_repository = MagicMock()
    document_repository.list_all = AsyncMock()
    document_repository.delete = AsyncMock()

    password_service = MagicMock()
    password_service.hash_password = AsyncMock()
    password_service.verify_password = AsyncMock()

    service = UserService(session)

    service._users = user_repository
    service._documents = document_repository
    service._passwords = password_service

    return (
        service,
        session,
        user_repository,
        document_repository,
        password_service,
    )


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        (
            "user@example.com",
            "user@example.com",
        ),
        (
            "  user@example.com  ",
            "user@example.com",
        ),
        (
            "USER@example.com",
            "USER@example.com",
        ),
    ],
)
def test_normalize_email_returns_valid_normalized_address(
    email: str,
    expected: str,
) -> None:
    """Validate and normalize syntactically valid email addresses."""
    service, _, _, _, _ = _build_service()

    result = service._normalize_email(email)

    assert result == expected


@pytest.mark.parametrize(
    "email",
    [
        "",
        " ",
        "not-an-email",
        "missing-at.example.com",
        "@example.com",
        "user@",
    ],
)
def test_normalize_email_rejects_invalid_address(
    email: str,
) -> None:
    """Reject syntactically invalid email addresses."""
    service, _, _, _, _ = _build_service()

    with pytest.raises(
        ValueError,
        match="Invalid email",
    ):
        service._normalize_email(email)


@pytest.mark.asyncio
async def test_register_user_normalizes_email_before_lookup() -> None:
    """Normalize the supplied email before checking for duplicates."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = None
    passwords.hash_password.return_value = "hashed-password"

    user = _build_user(
        email="user@example.com",
        password_hash="hashed-password",
    )

    users.create.return_value = user

    await service.register_user(
        email="  user@example.com  ",
        password="secret-password",
    )

    users.get_by_email.assert_awaited_once_with("user@example.com")


@pytest.mark.asyncio
async def test_register_user_rejects_empty_password_before_lookup() -> None:
    """Reject an empty password before database or hashing work begins."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    with pytest.raises(
        ValueError,
        match="Password cannot be empty",
    ):
        await service.register_user(
            email="user@example.com",
            password="",
        )

    users.get_by_email.assert_not_awaited()
    passwords.hash_password.assert_not_awaited()


@pytest.mark.asyncio
async def test_register_user_rejects_existing_email() -> None:
    """Reject registration when the normalized email already exists."""
    (
        service,
        session,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = _build_user()

    with pytest.raises(
        ValueError,
        match="Email is already registered",
    ):
        await service.register_user(
            email="user@example.com",
            password="secret-password",
        )

    passwords.hash_password.assert_not_awaited()
    users.create.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_register_user_hashes_plaintext_password() -> None:
    """Hash the plaintext password before persistence."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = None
    passwords.hash_password.return_value = "hashed-password"

    users.create.return_value = _build_user(
        password_hash="hashed-password",
    )

    await service.register_user(
        email="user@example.com",
        password="secret-password",
    )

    passwords.hash_password.assert_awaited_once_with("secret-password")


@pytest.mark.asyncio
async def test_register_user_persists_normalized_email_and_hash() -> None:
    """Persist the normalized email together with the generated password hash."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = None
    passwords.hash_password.return_value = "hashed-password"

    users.create.return_value = _build_user(
        email="user@example.com",
        password_hash="hashed-password",
    )

    await service.register_user(
        email="  user@example.com  ",
        password="secret-password",
    )

    users.create.assert_awaited_once_with(
        email="user@example.com",
        password_hash="hashed-password",
    )


@pytest.mark.asyncio
async def test_register_user_commits_refreshes_and_returns_user() -> None:
    """Commit registration and return the refreshed persisted user."""
    (
        service,
        session,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = None
    passwords.hash_password.return_value = "hashed-password"

    user = _build_user(
        password_hash="hashed-password",
    )

    users.create.return_value = user

    result = await service.register_user(
        email="user@example.com",
        password="secret-password",
    )

    session.commit.assert_awaited_once_with()
    session.refresh.assert_awaited_once_with(user)

    assert result is user


@pytest.mark.asyncio
async def test_register_user_rolls_back_integrity_error() -> None:
    """Rollback a concurrent duplicate-email database constraint failure."""
    (
        service,
        session,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = None
    passwords.hash_password.return_value = "hashed-password"

    integrity_error = IntegrityError(
        statement="INSERT",
        params={},
        orig=Exception("duplicate email"),
    )

    users.create.side_effect = integrity_error

    with pytest.raises(
        ValueError,
        match="Email is already registered",
    ) as exc_info:
        await service.register_user(
            email="user@example.com",
            password="secret-password",
        )

    session.rollback.assert_awaited_once_with()
    session.commit.assert_not_awaited()

    assert exc_info.value.__cause__ is integrity_error


@pytest.mark.asyncio
async def test_register_user_does_not_rollback_non_integrity_failure() -> None:
    """Propagate unrelated persistence failures without translating them."""
    (
        service,
        session,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = None
    passwords.hash_password.return_value = "hashed-password"

    repository_error = RuntimeError("database unavailable")

    users.create.side_effect = repository_error

    with pytest.raises(
        RuntimeError,
        match="database unavailable",
    ) as exc_info:
        await service.register_user(
            email="user@example.com",
            password="secret-password",
        )

    assert exc_info.value is repository_error

    session.rollback.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_authenticate_user_maps_invalid_email_to_generic_error() -> None:
    """Hide email syntax details behind the generic credential error."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    with pytest.raises(
        ValueError,
        match="Invalid email or password",
    ):
        await service.authenticate_user(
            email="not-an-email",
            password="secret-password",
        )

    users.get_by_email.assert_not_awaited()
    passwords.verify_password.assert_not_awaited()


@pytest.mark.asyncio
async def test_authenticate_user_normalizes_email_before_lookup() -> None:
    """Normalize the supplied email before account lookup."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    user = _build_user()
    users.get_by_email.return_value = user
    passwords.verify_password.return_value = True

    await service.authenticate_user(
        email="  user@example.com  ",
        password="secret-password",
    )

    users.get_by_email.assert_awaited_once_with("user@example.com")


@pytest.mark.asyncio
async def test_authenticate_user_rejects_unknown_account() -> None:
    """Return the generic credential error when no account exists."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = None

    with pytest.raises(
        ValueError,
        match="Invalid email or password",
    ):
        await service.authenticate_user(
            email="user@example.com",
            password="secret-password",
        )

    passwords.verify_password.assert_not_awaited()


@pytest.mark.asyncio
async def test_authenticate_user_verifies_password_against_stored_hash() -> None:
    """Verify plaintext credentials using the user's stored password hash."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    user = _build_user(
        password_hash="stored-hash",
    )

    users.get_by_email.return_value = user
    passwords.verify_password.return_value = True

    await service.authenticate_user(
        email="user@example.com",
        password="secret-password",
    )

    passwords.verify_password.assert_awaited_once_with(
        password="secret-password",
        password_hash="stored-hash",
    )


@pytest.mark.asyncio
async def test_authenticate_user_rejects_incorrect_password() -> None:
    """Return the generic credential error for an incorrect password."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = _build_user()
    passwords.verify_password.return_value = False

    with pytest.raises(
        ValueError,
        match="Invalid email or password",
    ):
        await service.authenticate_user(
            email="user@example.com",
            password="incorrect-password",
        )


@pytest.mark.asyncio
async def test_authenticate_user_checks_activity_after_password_verification() -> None:
    """Verify credentials before disclosing that an account is inactive."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = _build_user(
        is_active=False,
    )

    passwords.verify_password.return_value = True

    with pytest.raises(
        ValueError,
        match="User account is inactive",
    ):
        await service.authenticate_user(
            email="user@example.com",
            password="secret-password",
        )

    passwords.verify_password.assert_awaited_once()


@pytest.mark.asyncio
async def test_authenticate_user_does_not_check_activity_when_password_is_wrong() -> (
    None
):
    """Reject bad credentials before exposing inactive-account state."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    users.get_by_email.return_value = _build_user(
        is_active=False,
    )

    passwords.verify_password.return_value = False

    with pytest.raises(
        ValueError,
        match="Invalid email or password",
    ):
        await service.authenticate_user(
            email="user@example.com",
            password="incorrect-password",
        )


@pytest.mark.asyncio
async def test_authenticate_user_returns_active_authenticated_user() -> None:
    """Return the active user when both credentials are valid."""
    (
        service,
        _,
        users,
        _,
        passwords,
    ) = _build_service()

    user = _build_user(
        is_active=True,
    )

    users.get_by_email.return_value = user
    passwords.verify_password.return_value = True

    result = await service.authenticate_user(
        email="user@example.com",
        password="secret-password",
    )

    assert result is user


@pytest.mark.asyncio
async def test_delete_account_rejects_missing_user() -> None:
    """Reject account deletion when the user does not exist."""
    (
        service,
        session,
        users,
        documents,
        passwords,
    ) = _build_service()

    users.get_by_id.return_value = None

    with pytest.raises(
        ValueError,
        match="User not found",
    ):
        await service.delete_account(
            user_id=42,
            password="secret-password",
        )

    passwords.verify_password.assert_not_awaited()
    documents.list_all.assert_not_awaited()
    users.delete.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_account_rejects_inactive_user_before_password_check() -> None:
    """Reject inactive accounts before executing password verification."""
    (
        service,
        session,
        users,
        documents,
        passwords,
    ) = _build_service()

    users.get_by_id.return_value = _build_user(
        is_active=False,
    )

    with pytest.raises(
        ValueError,
        match="User account is inactive",
    ):
        await service.delete_account(
            user_id=42,
            password="secret-password",
        )

    passwords.verify_password.assert_not_awaited()
    documents.list_all.assert_not_awaited()
    users.delete.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_account_verifies_current_password() -> None:
    """Confirm the destructive action using the user's current password."""
    (
        service,
        _,
        users,
        documents,
        passwords,
    ) = _build_service()

    user = _build_user(
        password_hash="stored-hash",
    )

    users.get_by_id.return_value = user
    passwords.verify_password.return_value = True
    documents.list_all.return_value = []

    await service.delete_account(
        user_id=42,
        password="secret-password",
    )

    passwords.verify_password.assert_awaited_once_with(
        password="secret-password",
        password_hash="stored-hash",
    )


@pytest.mark.asyncio
async def test_delete_account_rejects_invalid_password_before_deletion() -> None:
    """Perform no destructive work when password confirmation fails."""
    (
        service,
        session,
        users,
        documents,
        passwords,
    ) = _build_service()

    users.get_by_id.return_value = _build_user()
    passwords.verify_password.return_value = False

    with pytest.raises(
        ValueError,
        match="Invalid password",
    ):
        await service.delete_account(
            user_id=42,
            password="incorrect-password",
        )

    documents.list_all.assert_not_awaited()
    documents.delete.assert_not_awaited()
    users.delete.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_account_lists_documents_using_persisted_user_id() -> None:
    """Load every document owned by the authenticated account."""
    (
        service,
        _,
        users,
        documents,
        passwords,
    ) = _build_service()

    user = _build_user(
        user_id=99,
    )

    users.get_by_id.return_value = user
    passwords.verify_password.return_value = True
    documents.list_all.return_value = []

    await service.delete_account(
        user_id=99,
        password="secret-password",
    )

    documents.list_all.assert_awaited_once_with(
        user_id=99,
    )


@pytest.mark.asyncio
async def test_delete_account_deletes_all_owned_documents() -> None:
    """Delete every owned document before deleting the user account."""
    (
        service,
        _,
        users,
        documents,
        passwords,
    ) = _build_service()

    user = _build_user()

    first = _build_document(
        document_id=1,
    )
    second = _build_document(
        document_id=2,
    )
    third = _build_document(
        document_id=3,
    )

    users.get_by_id.return_value = user
    passwords.verify_password.return_value = True

    documents.list_all.return_value = [
        first,
        second,
        third,
    ]

    await service.delete_account(
        user_id=42,
        password="secret-password",
    )

    assert documents.delete.await_count == 3

    documents.delete.assert_any_await(first)
    documents.delete.assert_any_await(second)
    documents.delete.assert_any_await(third)


@pytest.mark.asyncio
async def test_delete_account_deletes_user_after_documents() -> None:
    """Delete the account entity after removing all owned documents."""
    (
        service,
        _,
        users,
        documents,
        passwords,
    ) = _build_service()

    user = _build_user()
    document = _build_document(
        document_id=1,
    )

    users.get_by_id.return_value = user
    passwords.verify_password.return_value = True
    documents.list_all.return_value = [
        document,
    ]

    await service.delete_account(
        user_id=42,
        password="secret-password",
    )

    documents.delete.assert_awaited_once_with(document)

    users.delete.assert_awaited_once_with(user)


@pytest.mark.asyncio
async def test_delete_account_handles_user_without_documents() -> None:
    """Delete an account successfully when it owns no documents."""
    (
        service,
        session,
        users,
        documents,
        passwords,
    ) = _build_service()

    user = _build_user()

    users.get_by_id.return_value = user
    passwords.verify_password.return_value = True
    documents.list_all.return_value = []

    await service.delete_account(
        user_id=42,
        password="secret-password",
    )

    documents.delete.assert_not_awaited()
    users.delete.assert_awaited_once_with(user)
    session.commit.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_delete_account_commits_once_after_complete_deletion() -> None:
    """Commit once after documents and user have all been deleted."""
    (
        service,
        session,
        users,
        documents,
        passwords,
    ) = _build_service()

    user = _build_user()

    users.get_by_id.return_value = user
    passwords.verify_password.return_value = True

    documents.list_all.return_value = [
        _build_document(document_id=1),
        _build_document(document_id=2),
    ]

    await service.delete_account(
        user_id=42,
        password="secret-password",
    )

    session.commit.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_delete_account_does_not_commit_when_document_deletion_fails() -> None:
    """Avoid committing a partially failed account-deletion workflow."""
    (
        service,
        session,
        users,
        documents,
        passwords,
    ) = _build_service()

    user = _build_user()

    first = _build_document(
        document_id=1,
    )
    second = _build_document(
        document_id=2,
    )

    users.get_by_id.return_value = user
    passwords.verify_password.return_value = True

    documents.list_all.return_value = [
        first,
        second,
    ]

    documents.delete.side_effect = [
        None,
        RuntimeError("document deletion failed"),
    ]

    with pytest.raises(
        RuntimeError,
        match="document deletion failed",
    ):
        await service.delete_account(
            user_id=42,
            password="secret-password",
        )

    users.delete.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_account_does_not_commit_when_user_deletion_fails() -> None:
    """Avoid committing when final user deletion fails."""
    (
        service,
        session,
        users,
        documents,
        passwords,
    ) = _build_service()

    user = _build_user()

    users.get_by_id.return_value = user
    passwords.verify_password.return_value = True
    documents.list_all.return_value = []

    users.delete.side_effect = RuntimeError("user deletion failed")

    with pytest.raises(
        RuntimeError,
        match="user deletion failed",
    ):
        await service.delete_account(
            user_id=42,
            password="secret-password",
        )

    session.commit.assert_not_awaited()
