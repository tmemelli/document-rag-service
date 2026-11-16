"""
Tests for JWT access-token creation and validation.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest

from document_rag_service.services.auth_token_service import AuthTokenService

TEST_SECRET = (
    "test-secret-key-for-auth-token-service-tests-"
    "long-enough-for-all-supported-hmac-test-algorithms"
)
TEST_ALGORITHM = "HS256"
TEST_ISSUER = "document-rag-service-test"
TEST_AUDIENCE = "document-rag-client-test"
TEST_EXPIRE_MINUTES = 30


@pytest.fixture
def auth_token_service() -> AuthTokenService:
    """Return an authentication token service configured for isolated tests."""
    return AuthTokenService(
        secret=TEST_SECRET,
        algorithm=TEST_ALGORITHM,
        issuer=TEST_ISSUER,
        audience=TEST_AUDIENCE,
        expire_minutes=TEST_EXPIRE_MINUTES,
    )


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    """Return a valid access-token payload with optional claim overrides."""
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": "42",
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=TEST_EXPIRE_MINUTES),
        "iss": TEST_ISSUER,
        "aud": TEST_AUDIENCE,
    }
    payload.update(overrides)
    return payload


def _encode_token(
    payload: dict[str, Any],
    *,
    secret: str = TEST_SECRET,
    algorithm: str = TEST_ALGORITHM,
) -> str:
    """Encode a JWT used to exercise token-validation behavior."""
    return jwt.encode(
        payload,
        secret,
        algorithm=algorithm,
    )


def test_auth_token_service_accepts_valid_configuration() -> None:
    """Accept complete JWT configuration with a positive token lifetime."""
    service = AuthTokenService(
        secret=TEST_SECRET,
        algorithm=TEST_ALGORITHM,
        issuer=TEST_ISSUER,
        audience=TEST_AUDIENCE,
        expire_minutes=TEST_EXPIRE_MINUTES,
    )

    assert isinstance(service, AuthTokenService)


@pytest.mark.parametrize(
    ("field", "error_message"),
    [
        ("secret", "JWT secret cannot be empty"),
        ("algorithm", "JWT algorithm cannot be empty"),
        ("issuer", "JWT issuer cannot be empty"),
        ("audience", "JWT audience cannot be empty"),
    ],
)
def test_auth_token_service_rejects_empty_required_configuration(
    field: str,
    error_message: str,
) -> None:
    """Reject empty values for every required JWT string setting."""
    config: dict[str, Any] = {
        "secret": TEST_SECRET,
        "algorithm": TEST_ALGORITHM,
        "issuer": TEST_ISSUER,
        "audience": TEST_AUDIENCE,
        "expire_minutes": TEST_EXPIRE_MINUTES,
    }
    config[field] = ""

    with pytest.raises(ValueError, match=error_message):
        AuthTokenService(**config)


@pytest.mark.parametrize("expire_minutes", [0, -1, -30])
def test_auth_token_service_rejects_non_positive_expiration(
    expire_minutes: int,
) -> None:
    """Reject zero or negative access-token lifetimes."""
    with pytest.raises(
        ValueError,
        match="JWT expiration must be greater than zero",
    ):
        AuthTokenService(
            secret=TEST_SECRET,
            algorithm=TEST_ALGORITHM,
            issuer=TEST_ISSUER,
            audience=TEST_AUDIENCE,
            expire_minutes=expire_minutes,
        )


def test_create_access_token_returns_signed_jwt(
    auth_token_service: AuthTokenService,
) -> None:
    """Create a JWT that can be verified with the configured signing secret."""
    token = auth_token_service.create_access_token(user_id=42)

    assert isinstance(token, str)
    assert token.count(".") == 2

    payload = jwt.decode(
        token,
        TEST_SECRET,
        algorithms=[TEST_ALGORITHM],
        audience=TEST_AUDIENCE,
        issuer=TEST_ISSUER,
    )

    assert payload["sub"] == "42"


def test_create_access_token_contains_expected_authentication_claims(
    auth_token_service: AuthTokenService,
) -> None:
    """Include all claims required by the service authentication contract."""
    token = auth_token_service.create_access_token(user_id=123)

    payload = jwt.decode(
        token,
        TEST_SECRET,
        algorithms=[TEST_ALGORITHM],
        audience=TEST_AUDIENCE,
        issuer=TEST_ISSUER,
    )

    assert payload["sub"] == "123"
    assert payload["type"] == "access"
    assert payload["iss"] == TEST_ISSUER
    assert payload["aud"] == TEST_AUDIENCE
    assert isinstance(payload["iat"], int)
    assert isinstance(payload["exp"], int)


def test_create_access_token_uses_configured_expiration_lifetime(
    auth_token_service: AuthTokenService,
) -> None:
    """Set expiration exactly to the configured number of minutes after issue."""
    token = auth_token_service.create_access_token(user_id=42)

    payload = jwt.decode(
        token,
        TEST_SECRET,
        algorithms=[TEST_ALGORITHM],
        audience=TEST_AUDIENCE,
        issuer=TEST_ISSUER,
    )

    assert payload["exp"] - payload["iat"] == TEST_EXPIRE_MINUTES * 60


@pytest.mark.parametrize("user_id", [0, -1, -999])
def test_create_access_token_rejects_non_positive_user_id(
    auth_token_service: AuthTokenService,
    user_id: int,
) -> None:
    """Reject user identifiers that cannot represent authenticated users."""
    with pytest.raises(
        ValueError,
        match="User id must be greater than zero",
    ):
        auth_token_service.create_access_token(user_id=user_id)


def test_get_user_id_returns_subject_from_valid_access_token(
    auth_token_service: AuthTokenService,
) -> None:
    """Return the positive integer user identifier from a valid token."""
    token = auth_token_service.create_access_token(user_id=987)

    assert auth_token_service.get_user_id(token) == 987


@pytest.mark.parametrize("token", ["", " ", "   ", "\t", "\n"])
def test_get_user_id_rejects_empty_or_blank_token(
    auth_token_service: AuthTokenService,
    token: str,
) -> None:
    """Reject missing tokens and tokens containing only whitespace."""
    with pytest.raises(
        ValueError,
        match="Access token cannot be empty",
    ):
        auth_token_service.get_user_id(token)


@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        "invalid.jwt.token",
        "a.b",
    ],
)
def test_get_user_id_rejects_malformed_token(
    auth_token_service: AuthTokenService,
    token: str,
) -> None:
    """Reject strings that do not represent a valid signed JWT."""
    with pytest.raises(
        ValueError,
        match="Invalid or expired access token",
    ):
        auth_token_service.get_user_id(token)


def test_get_user_id_rejects_token_signed_with_different_secret(
    auth_token_service: AuthTokenService,
) -> None:
    """Reject tokens whose signature was created with another secret."""
    token = _encode_token(
        _valid_payload(),
        secret="another-secret-key-with-at-least-32-characters",
    )

    with pytest.raises(
        ValueError,
        match="Invalid or expired access token",
    ):
        auth_token_service.get_user_id(token)


def test_get_user_id_rejects_token_signed_with_unexpected_algorithm(
    auth_token_service: AuthTokenService,
) -> None:
    """Reject tokens signed with an algorithm outside the configured allowlist."""
    token = _encode_token(
        _valid_payload(),
        algorithm="HS384",
    )

    with pytest.raises(
        ValueError,
        match="Invalid or expired access token",
    ):
        auth_token_service.get_user_id(token)


def test_get_user_id_rejects_expired_token(
    auth_token_service: AuthTokenService,
) -> None:
    """Reject access tokens whose expiration time has already passed."""
    now = datetime.now(UTC)
    token = _encode_token(
        _valid_payload(
            iat=now - timedelta(minutes=31),
            exp=now - timedelta(minutes=1),
        )
    )

    with pytest.raises(
        ValueError,
        match="Invalid or expired access token",
    ):
        auth_token_service.get_user_id(token)


def test_get_user_id_rejects_unexpected_issuer(
    auth_token_service: AuthTokenService,
) -> None:
    """Reject otherwise valid tokens issued by another authority."""
    token = _encode_token(
        _valid_payload(iss="unexpected-issuer"),
    )

    with pytest.raises(
        ValueError,
        match="Invalid or expired access token",
    ):
        auth_token_service.get_user_id(token)


def test_get_user_id_rejects_unexpected_audience(
    auth_token_service: AuthTokenService,
) -> None:
    """Reject otherwise valid tokens intended for another audience."""
    token = _encode_token(
        _valid_payload(aud="unexpected-audience"),
    )

    with pytest.raises(
        ValueError,
        match="Invalid or expired access token",
    ):
        auth_token_service.get_user_id(token)


@pytest.mark.parametrize(
    "missing_claim",
    [
        "sub",
        "type",
        "iat",
        "exp",
        "iss",
        "aud",
    ],
)
def test_get_user_id_rejects_missing_required_claim(
    auth_token_service: AuthTokenService,
    missing_claim: str,
) -> None:
    """Reject tokens missing any claim required by the authentication contract."""
    payload = _valid_payload()
    payload.pop(missing_claim)
    token = _encode_token(payload)

    with pytest.raises(
        ValueError,
        match="Invalid or expired access token",
    ):
        auth_token_service.get_user_id(token)


@pytest.mark.parametrize("token_type", ["refresh", "bearer", "", 123])
def test_get_user_id_rejects_unexpected_token_type(
    auth_token_service: AuthTokenService,
    token_type: Any,
) -> None:
    """Reject signed tokens that are not explicitly access tokens."""
    token = _encode_token(
        _valid_payload(type=token_type),
    )

    with pytest.raises(
        ValueError,
        match="Invalid access token type",
    ):
        auth_token_service.get_user_id(token)


def test_get_user_id_rejects_null_token_type(
    auth_token_service: AuthTokenService,
) -> None:
    """Reject a token whose required type claim is null."""
    token = _encode_token(
        _valid_payload(type=None),
    )

    with pytest.raises(
        ValueError,
        match="Invalid or expired access token",
    ):
        auth_token_service.get_user_id(token)


@pytest.mark.parametrize("subject", ["not-an-integer", "42.5", "user-42"])
def test_get_user_id_rejects_non_numeric_string_subject(
    auth_token_service: AuthTokenService,
    subject: str,
) -> None:
    """Reject string subjects that cannot be converted to an integer user id."""
    token = _encode_token(
        _valid_payload(sub=subject),
    )

    with pytest.raises(
        ValueError,
        match="Access token subject is invalid",
    ):
        auth_token_service.get_user_id(token)


@pytest.mark.parametrize("subject", ["0", "-1", "-999"])
def test_get_user_id_rejects_non_positive_subject(
    auth_token_service: AuthTokenService,
    subject: str,
) -> None:
    """Reject numeric subjects that do not represent positive user identifiers."""
    token = _encode_token(
        _valid_payload(sub=subject),
    )

    with pytest.raises(
        ValueError,
        match="Access token subject is invalid",
    ):
        auth_token_service.get_user_id(token)


@pytest.mark.parametrize("subject", [42, 42.0, None, ["42"], {"id": 42}])
def test_get_user_id_rejects_non_string_subject(
    auth_token_service: AuthTokenService,
    subject: Any,
) -> None:
    """Reject subjects that violate JWT's string subject requirement."""
    token = _encode_token(
        _valid_payload(sub=subject),
    )

    with pytest.raises(
        ValueError,
        match="Invalid or expired access token",
    ):
        auth_token_service.get_user_id(token)
