"""
Tests for asynchronous Argon2id password hashing and verification.
"""

from unittest.mock import AsyncMock

import pytest

from document_rag_service.services import password_service as password_service_module
from document_rag_service.services.password_service import PasswordService


@pytest.mark.asyncio
async def test_hash_password_returns_argon2id_hash() -> None:
    """Return an encoded Argon2id hash for a valid plaintext password."""
    service = PasswordService()

    password_hash = await service.hash_password("correct-horse-battery-staple")

    assert password_hash.startswith("$argon2id$")
    assert password_hash != "correct-horse-battery-staple"


@pytest.mark.asyncio
async def test_hash_password_uses_random_salt() -> None:
    """Produce distinct hashes for identical plaintext passwords."""
    service = PasswordService()
    password = "same-password"

    first_hash = await service.hash_password(password)
    second_hash = await service.hash_password(password)

    assert first_hash != second_hash

    assert await service.verify_password(
        password=password,
        password_hash=first_hash,
    )

    assert await service.verify_password(
        password=password,
        password_hash=second_hash,
    )


@pytest.mark.asyncio
async def test_hash_password_rejects_empty_password() -> None:
    """Reject an empty plaintext password before invoking Argon2."""
    service = PasswordService()

    with pytest.raises(
        ValueError,
        match="Password cannot be empty",
    ):
        await service.hash_password("")


@pytest.mark.asyncio
async def test_hash_password_offloads_argon2_work_to_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run CPU-bound password hashing through asyncio.to_thread."""
    service = PasswordService()

    to_thread = AsyncMock(
        return_value="encoded-hash",
    )

    monkeypatch.setattr(
        password_service_module.asyncio,
        "to_thread",
        to_thread,
    )

    result = await service.hash_password("secret-password")

    assert result == "encoded-hash"

    to_thread.assert_awaited_once_with(
        service._hasher.hash,
        "secret-password",
    )


@pytest.mark.asyncio
async def test_verify_password_returns_true_for_matching_password() -> None:
    """Accept the original password for its stored Argon2id hash."""
    service = PasswordService()

    password = "correct-password"

    password_hash = await service.hash_password(password)

    result = await service.verify_password(
        password=password,
        password_hash=password_hash,
    )

    assert result is True


@pytest.mark.asyncio
async def test_verify_password_returns_false_for_incorrect_password() -> None:
    """Reject a plaintext password that does not match the stored hash."""
    service = PasswordService()

    password_hash = await service.hash_password("correct-password")

    result = await service.verify_password(
        password="incorrect-password",
        password_hash=password_hash,
    )

    assert result is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "password_hash",
    [
        "not-an-argon2-hash",
        "$argon2id$invalid",
        "malformed-hash-value",
    ],
)
async def test_verify_password_returns_false_for_malformed_hash(
    password_hash: str,
) -> None:
    """Treat malformed stored hashes as authentication failures."""
    service = PasswordService()

    result = await service.verify_password(
        password="password",
        password_hash=password_hash,
    )

    assert result is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("password", "password_hash"),
    [
        ("", "stored-hash"),
        ("password", ""),
        ("", ""),
    ],
)
async def test_verify_password_returns_false_for_empty_input(
    password: str,
    password_hash: str,
) -> None:
    """Reject empty verification inputs without invoking Argon2."""
    service = PasswordService()

    result = await service.verify_password(
        password=password,
        password_hash=password_hash,
    )

    assert result is False


@pytest.mark.asyncio
async def test_verify_password_does_not_offload_when_input_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return early for empty input without scheduling worker-thread work."""
    service = PasswordService()

    to_thread = AsyncMock()

    monkeypatch.setattr(
        password_service_module.asyncio,
        "to_thread",
        to_thread,
    )

    result = await service.verify_password(
        password="",
        password_hash="stored-hash",
    )

    assert result is False
    to_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_password_offloads_argon2_work_to_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run CPU-bound password verification through asyncio.to_thread."""
    service = PasswordService()

    to_thread = AsyncMock(
        return_value=True,
    )

    monkeypatch.setattr(
        password_service_module.asyncio,
        "to_thread",
        to_thread,
    )

    result = await service.verify_password(
        password="secret-password",
        password_hash="encoded-hash",
    )

    assert result is True

    to_thread.assert_awaited_once_with(
        service._hasher.verify,
        "encoded-hash",
        "secret-password",
    )
