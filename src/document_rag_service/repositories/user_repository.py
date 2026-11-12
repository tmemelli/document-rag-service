"""Repository for application user persistence and lookup operations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.user import User


class UserRepository:
    """
    Provide persistence operations for application users.

    The repository isolates SQLAlchemy access for user creation, lookup, and
    deletion while leaving transaction boundaries to the calling service.
    Mutating operations flush changes so generated identifiers and database
    state are available without committing the surrounding transaction.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the user repository.

        Args:
            session: Asynchronous SQLAlchemy session used for persistence
                operations.
        """

        self._session = session

    async def create(
        self,
        *,
        email: str,
        password_hash: str,
    ) -> User:
        """
        Create and persist a new application user.

        The user is added to the current session, flushed so database-generated
        values become available, and refreshed before being returned. The
        transaction is not committed by the repository.

        Args:
            email: Normalized email address associated with the user.
            password_hash: Secure hash of the user's password.

        Returns:
            Newly created and refreshed user entity.
        """

        user = User(
            email=email,
            password_hash=password_hash,
        )

        self._session.add(user)
        await self._session.flush()
        await self._session.refresh(user)

        return user

    async def get_by_id(
        self,
        user_id: int,
    ) -> User | None:
        """
        Return a user by primary-key identifier.

        Args:
            user_id: Identifier of the user to retrieve.

        Returns:
            Matching user entity, or ``None`` when no user exists with the
            provided identifier.
        """

        return await self._session.get(
            User,
            user_id,
        )

    async def get_by_email(
        self,
        email: str,
    ) -> User | None:
        """
        Return a user by email address.

        Args:
            email: Email address used to locate the user.

        Returns:
            Matching user entity, or ``None`` when the email is not registered.
        """

        statement = select(User).where(User.email == email)

        return await self._session.scalar(statement)

    async def delete(
        self,
        user: User,
    ) -> None:
        """
        Delete a user from the current database session.

        The deletion is flushed immediately so database-side effects, including
        configured cascades, are reflected within the transaction. Committing
        or rolling back the transaction remains the responsibility of the
        calling service.

        Args:
            user: Persisted user entity to delete.
        """

        await self._session.delete(user)
        await self._session.flush()
