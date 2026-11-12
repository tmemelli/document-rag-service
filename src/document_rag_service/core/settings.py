"""
Application settings, environment loading, and configuration validation.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Load and validate configuration for the Document RAG service.

    Configuration is sourced from environment variables and the local ``.env``
    file. Validation happens during settings construction so invalid database,
    authentication, embedding, chunking, or inference configuration fails
    before the application starts serving requests.

    Provider-specific validation is applied conditionally, allowing local and
    Cloudflare embedding backends to share the same configuration model without
    requiring credentials for an unused provider.
    """

    # Database
    postgres_host: str = Field(min_length=1)
    postgres_port: int = Field(default=5432, ge=1, le=65_535)
    postgres_db: str = Field(min_length=1)
    postgres_user: str = Field(min_length=1)
    postgres_password: SecretStr
    postgres_sslmode: Literal[
        "disable",
        "allow",
        "prefer",
        "require",
        "verify-ca",
        "verify-full",
    ] = "prefer"

    # HTTP API
    cors_allowed_origins: list[str] = Field(default_factory=list)

    document_upload_max_size_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1,
    )

    auth_register_rate_limit_per_minute: int = Field(
        default=3,
        ge=1,
    )
    auth_login_rate_limit_per_minute: int = Field(
        default=5,
        ge=1,
    )
    rag_rate_limit_per_minute: int = Field(
        default=5,
        ge=1,
    )

    # Chunking
    chunk_size: int = Field(default=1000, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)

    # Embeddings
    embedding_provider: Literal["local", "cloudflare"] = "local"
    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        min_length=1,
    )
    embedding_dimension: int = Field(default=512, ge=1)
    embedding_device: str = Field(default="cuda", min_length=1)

    # Cloudflare Workers AI
    cloudflare_account_id: str | None = None
    cloudflare_api_token: SecretStr | None = None
    cloudflare_embedding_model: str = Field(
        default="@cf/qwen/qwen3-embedding-0.6b",
        min_length=1,
    )
    cloudflare_embedding_max_sequence_length: int = Field(
        default=8192,
        ge=1,
    )
    cloudflare_embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=32,
    )
    cloudflare_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
    )
    cloudflare_max_attempts: int = Field(
        default=3,
        ge=1,
    )
    cloudflare_retry_base_delay_seconds: float = Field(
        default=1.0,
        gt=0,
    )

    # LLM Inference Service
    llm_inference_base_url: str = Field(
        default="https://llm-inference-service-g4os.onrender.com",
        min_length=1,
    )
    llm_inference_provider: str = Field(
        default="groq",
        min_length=1,
    )
    llm_inference_model: str = Field(
        default="openai/gpt-oss-120b",
        min_length=1,
    )
    llm_inference_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )
    llm_inference_max_tokens: int = Field(
        default=2000,
        ge=1,
    )
    llm_inference_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
    )
    llm_inference_max_prompt_characters: int = Field(
        default=20_000,
        ge=1,
    )

    # Query transformation
    query_transformation_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )
    query_transformation_max_tokens: int = Field(
        default=2000,
        ge=1,
    )
    query_transformation_max_generated_queries: int = Field(
        default=2,
        ge=1,
    )
    query_transformation_max_attempts: int = Field(
        default=2,
        ge=1,
    )

    # Authentication
    auth_jwt_secret: SecretStr
    auth_jwt_algorithm: Literal["HS256"] = "HS256"
    auth_jwt_issuer: str = "document-rag-service"
    auth_jwt_audience: str = "document-rag-frontend"
    auth_access_token_expire_minutes: int = 480

    @field_validator("auth_jwt_secret")
    @classmethod
    def validate_auth_jwt_secret(
        cls,
        value: SecretStr,
    ) -> SecretStr:
        """
        Validate the minimum length of the JWT signing secret.

        Args:
            value: Secret configured for JWT signing and verification.

        Returns:
            Validated JWT secret.

        Raises:
            ValueError: If the secret contains fewer than 32 characters.
        """

        if len(value.get_secret_value()) < 32:
            raise ValueError("AUTH_JWT_SECRET must contain at least 32 characters")

        return value

    @field_validator("auth_access_token_expire_minutes")
    @classmethod
    def validate_auth_access_token_expire_minutes(
        cls,
        value: int,
    ) -> int:
        """
        Validate the configured access-token lifetime.

        Args:
            value: Access-token lifetime expressed in minutes.

        Returns:
            Validated positive token lifetime.

        Raises:
            ValueError: If the configured lifetime is not greater than zero.
        """

        if value <= 0:
            raise ValueError(
                "AUTH_ACCESS_TOKEN_EXPIRE_MINUTES must be greater than zero"
            )

        return value

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        """
        Validate relationships and provider-specific configuration.

        Chunk overlap must remain smaller than the configured chunk size so the
        chunking algorithm always makes forward progress. Cloudflare account
        credentials are required only when the Cloudflare embedding provider is
        selected.

        Returns:
            Fully validated settings instance.

        Raises:
            ValueError: If chunking settings are inconsistent or required
                Cloudflare credentials are missing.
        """

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        if self.embedding_provider == "cloudflare":
            if (
                self.cloudflare_account_id is None
                or not self.cloudflare_account_id.strip()
            ):
                raise ValueError(
                    "CLOUDFLARE_ACCOUNT_ID is required when "
                    "EMBEDDING_PROVIDER=cloudflare."
                )

            if (
                self.cloudflare_api_token is None
                or not self.cloudflare_api_token.get_secret_value().strip()
            ):
                raise ValueError(
                    "CLOUDFLARE_API_TOKEN is required when "
                    "EMBEDDING_PROVIDER=cloudflare."
                )

        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return the process-wide cached application settings instance.

    Settings are constructed and validated only on the first call. Subsequent
    calls reuse the same immutable configuration source for the lifetime of the
    application process.

    Returns:
        Validated and cached application settings.
    """

    return Settings()  # pyright: ignore[reportCallIssue]
