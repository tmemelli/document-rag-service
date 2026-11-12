"""Authentication access-token creation and validation."""

from datetime import UTC, datetime, timedelta

import jwt
from jwt import InvalidTokenError


class AuthTokenService:
    """
    Create and validate signed JWT access tokens for authenticated users.

    The service centralizes token issuance and verification rules, including
    signing configuration, issuer and audience validation, expiration, token
    type enforcement, and extraction of the authenticated user identifier.
    """

    def __init__(
        self,
        *,
        secret: str,
        algorithm: str,
        issuer: str,
        audience: str,
        expire_minutes: int,
    ) -> None:
        """
        Initialize the authentication token service.

        Args:
            secret: Secret key used to sign and verify access tokens.
            algorithm: JWT signing algorithm accepted by the service.
            issuer: Issuer value written to and expected from access tokens.
            audience: Audience value written to and expected from access tokens.
            expire_minutes: Access-token lifetime in minutes.

        Raises:
            ValueError: If any required JWT configuration value is empty or the
                configured expiration is not positive.
        """

        if not secret:
            raise ValueError("JWT secret cannot be empty")

        if not algorithm:
            raise ValueError("JWT algorithm cannot be empty")

        if not issuer:
            raise ValueError("JWT issuer cannot be empty")

        if not audience:
            raise ValueError("JWT audience cannot be empty")

        if expire_minutes <= 0:
            raise ValueError("JWT expiration must be greater than zero")

        self._secret = secret
        self._algorithm = algorithm
        self._issuer = issuer
        self._audience = audience
        self._expire_minutes = expire_minutes

    def create_access_token(
        self,
        *,
        user_id: int,
    ) -> str:
        """
        Create a signed JWT access token for an authenticated user.

        The token contains the user identifier as its subject together with
        explicit token type, issuance time, expiration time, issuer, and
        audience claims.

        Args:
            user_id: Identifier of the authenticated user.

        Returns:
            Encoded and signed JWT access token.

        Raises:
            ValueError: If ``user_id`` is not a positive identifier.
        """

        if user_id <= 0:
            raise ValueError("User id must be greater than zero")

        now = datetime.now(UTC)
        expires_at = now + timedelta(
            minutes=self._expire_minutes,
        )

        payload = {
            "sub": str(user_id),
            "type": "access",
            "iat": now,
            "exp": expires_at,
            "iss": self._issuer,
            "aud": self._audience,
        }

        return jwt.encode(
            payload,
            self._secret,
            algorithm=self._algorithm,
        )

    def get_user_id(
        self,
        token: str,
    ) -> int:
        """
        Validate an access token and return its authenticated user identifier.

        Signature, algorithm, issuer, audience, expiration, and all required
        authentication claims are validated before the subject is accepted.
        The token must also explicitly identify itself as an access token.

        Args:
            token: Encoded JWT access token received from the client.

        Returns:
            Positive user identifier stored in the token subject.

        Raises:
            ValueError: If the token is empty, malformed, expired, improperly
                signed, missing required claims, has an unexpected issuer or
                audience, has the wrong token type, or contains an invalid
                user identifier.
        """

        if not token.strip():
            raise ValueError("Access token cannot be empty")

        try:
            payload = jwt.decode(
                token,
                self._secret,
                algorithms=[self._algorithm],
                audience=self._audience,
                issuer=self._issuer,
                options={
                    "require": [
                        "sub",
                        "type",
                        "iat",
                        "exp",
                        "iss",
                        "aud",
                    ]
                },
            )
        except InvalidTokenError as exc:
            raise ValueError("Invalid or expired access token") from exc

        if payload.get("type") != "access":
            raise ValueError("Invalid access token type")

        subject = payload.get("sub")

        if not isinstance(subject, str):
            raise ValueError("Access token subject is invalid")

        try:
            user_id = int(subject)
        except ValueError as exc:
            raise ValueError("Access token subject is invalid") from exc

        if user_id <= 0:
            raise ValueError("Access token subject is invalid")

        return user_id
