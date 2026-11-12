"""Authentication request and response schemas exposed by the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    """
    Validate credentials supplied when creating a new user account.

    Attributes:
        email: Syntactically valid email address for the new account.
        password: Plaintext password accepted by the registration workflow,
            constrained to the supported password length.
    """

    email: EmailStr
    password: str = Field(
        min_length=8,
        max_length=128,
    )


class LoginRequest(BaseModel):
    """
    Validate credentials supplied for user authentication.

    Attributes:
        email: Syntactically valid email address associated with the account.
        password: Plaintext password supplied for authentication.
    """

    email: EmailStr
    password: str = Field(
        min_length=8,
        max_length=128,
    )


class UserResponse(BaseModel):
    """
    Represent the public API view of an authenticated user.

    The schema is configured to read values directly from ORM model
    attributes while exposing only non-sensitive account information.

    Attributes:
        id: Unique user identifier.
        email: Normalized account email address.
        is_active: Whether the account is currently active.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: int
    email: EmailStr
    is_active: bool


class AuthResponse(BaseModel):
    """
    Represent a successful registration or authentication response.

    Attributes:
        access_token: Bearer access token issued for the authenticated user.
        token_type: Authentication scheme associated with the token.
        user: Public representation of the authenticated user.
    """

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    user: UserResponse


class DeleteAccountRequest(BaseModel):
    """
    Validate password confirmation for permanent account deletion.

    Attributes:
        password: Current plaintext password used to authorize the destructive
            account-deletion operation.
    """

    password: str = Field(
        min_length=8,
        max_length=128,
    )
