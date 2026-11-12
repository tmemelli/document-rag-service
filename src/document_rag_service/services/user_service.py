"""User registration, authentication, and account lifecycle services."""

from email_validator import EmailNotValidError, validate_email
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.user import User
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.user_repository import UserRepository
from document_rag_service.services.password_service import PasswordService


class UserService:
    """
    Coordinate user registration, authentication, and account deletion.

    The service owns application-level user workflows while delegating
    persistence to repositories and password hashing or verification to
    ``PasswordService``.

    Email addresses are normalized consistently before lookup or persistence,
    and transaction boundaries for user mutations are controlled here rather
    than inside the repository layer.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the user service.

        Args:
            session: Database session used for user and document persistence.
        """

        self._session = session
        self._users = UserRepository(session)
        self._documents = DocumentRepository(session)
        self._passwords = PasswordService()

    def _normalize_email(
        self,
        email: str,
    ) -> str:
        """
        Validate and normalize an email address for consistent persistence.

        Deliverability checks are intentionally disabled because registration
        validates address syntax and normalization without performing external
        DNS or mailbox verification.

        Args:
            email: Email address supplied by the user.

        Returns:
            Validated and normalized email address.

        Raises:
            ValueError: If the supplied address is not syntactically valid.
        """

        try:
            email_info = validate_email(
                email.strip(),
                check_deliverability=False,
            )
        except EmailNotValidError as exc:
            raise ValueError("Invalid email") from exc

        return email_info.normalized

    async def register_user(
        self,
        *,
        email: str,
        password: str,
    ) -> User:
        """
        Register a new application user.

        The email address is normalized before duplicate detection and
        persistence. Password hashing is completed before the user record is
        created.

        A pre-insert lookup provides an immediate duplicate-email error, while
        the database unique constraint remains the authoritative protection
        against concurrent registration attempts. Any resulting
        ``IntegrityError`` is rolled back and translated into the same
        application-level duplicate-email error.

        Args:
            email: Email address for the new account.
            password: Plaintext password to hash and store securely.

        Returns:
            Persisted user entity after the registration transaction commits.

        Raises:
            ValueError: If the email is invalid, the password is empty, or the
                normalized email is already registered.
        """

        normalized_email = self._normalize_email(email)

        if not password:
            raise ValueError("Password cannot be empty")

        existing_user = await self._users.get_by_email(normalized_email)

        if existing_user is not None:
            raise ValueError("Email is already registered")

        password_hash = await self._passwords.hash_password(password)

        try:
            user = await self._users.create(
                email=normalized_email,
                password_hash=password_hash,
            )

            await self._session.commit()
            await self._session.refresh(user)

            return user

        except IntegrityError as exc:
            await self._session.rollback()

            raise ValueError("Email is already registered") from exc

    async def authenticate_user(
        self,
        *,
        email: str,
        password: str,
    ) -> User:
        """
        Authenticate a user with email and password credentials.

        Invalid email syntax, unknown accounts, and incorrect passwords are
        intentionally exposed through the same generic error so callers do not
        reveal which credential component failed.

        Account activity is checked only after the supplied credentials have
        been successfully verified.

        Args:
            email: Email address supplied during authentication.
            password: Plaintext password supplied during authentication.

        Returns:
            Authenticated active user entity.

        Raises:
            ValueError: If the credentials are invalid or the authenticated
                account is inactive.
        """

        try:
            normalized_email = self._normalize_email(email)
        except ValueError as exc:
            raise ValueError("Invalid email or password") from exc

        user = await self._users.get_by_email(normalized_email)

        if user is None:
            raise ValueError("Invalid email or password")

        password_is_valid = await self._passwords.verify_password(
            password=password,
            password_hash=user.password_hash,
        )

        if not password_is_valid:
            raise ValueError("Invalid email or password")

        if not user.is_active:
            raise ValueError("User account is inactive")

        return user

    async def delete_account(
        self,
        *,
        user_id: int,
        password: str,
    ) -> None:
        """
        Permanently delete a user account and all owned documents.

        The current password must be confirmed before any destructive operation
        occurs. Owned documents are deleted first so their dependent chunks,
        processing-history records, and error events are removed through
        database-level cascading foreign keys before the user itself is
        deleted.

        All deletions participate in the same service-level transaction and are
        committed only after the complete account-removal workflow succeeds.

        Args:
            user_id: Identifier of the user whose account will be deleted.
            password: Current password used to confirm the destructive action.

        Raises:
            ValueError: If the user does not exist, the account is inactive, or
                the supplied password is invalid.
        """

        user = await self._users.get_by_id(user_id)

        if user is None:
            raise ValueError("User not found")

        if not user.is_active:
            raise ValueError("User account is inactive")

        password_is_valid = await self._passwords.verify_password(
            password=password,
            password_hash=user.password_hash,
        )

        if not password_is_valid:
            raise ValueError("Invalid password")

        documents = await self._documents.list_all(
            user_id=user.id,
        )

        for document in documents:
            await self._documents.delete(document)

        await self._users.delete(user)

        await self._session.commit()
