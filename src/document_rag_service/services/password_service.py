"""Password hashing and verification services using Argon2id."""

import asyncio

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from argon2.low_level import Type


class PasswordService:
    """
    Hash and verify application passwords using Argon2id.

    Argon2 operations are CPU-bound, so hashing and verification are executed
    in worker threads through ``asyncio.to_thread`` to avoid blocking the
    application's asynchronous event loop.

    The service centralizes password handling so authentication code does not
    depend directly on the underlying hashing library or its exception model.
    """

    def __init__(self) -> None:
        """
        Initialize the Argon2id password hasher.

        The library's maintained default cost parameters are used while the
        hashing algorithm is explicitly configured as Argon2id.
        """

        self._hasher = PasswordHasher(
            type=Type.ID,
        )

    async def hash_password(
        self,
        password: str,
    ) -> str:
        """
        Generate an Argon2id hash for a plaintext password.

        Hashing is offloaded to a worker thread because Argon2 is intentionally
        computationally expensive and should not block the async event loop.

        Args:
            password: Plaintext password to hash.

        Returns:
            Encoded Argon2id password hash containing the parameters, salt,
            and derived hash required for later verification.

        Raises:
            ValueError: If the password is empty.
        """

        if not password:
            raise ValueError("Password cannot be empty")

        return await asyncio.to_thread(
            self._hasher.hash,
            password,
        )

    async def verify_password(
        self,
        *,
        password: str,
        password_hash: str,
    ) -> bool:
        """
        Verify a plaintext password against a stored Argon2 hash.

        Verification is executed in a worker thread to keep the asynchronous
        event loop responsive. Incorrect passwords and malformed stored hashes
        are both treated as authentication failures rather than being exposed
        as application errors.

        Args:
            password: Plaintext password supplied by the user.
            password_hash: Encoded Argon2 password hash stored for the account.

        Returns:
            ``True`` when the password matches the stored hash; otherwise
            ``False``.
        """

        if not password or not password_hash:
            return False

        try:
            return await asyncio.to_thread(
                self._hasher.verify,
                password_hash,
                password,
            )
        except (InvalidHashError, VerificationError):
            return False
