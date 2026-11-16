"""
FastAPI dependencies for request-scoped and shared application resources.

This module connects the HTTP layer to application services and infrastructure.
Long-lived resources initialized by the FastAPI lifespan are retrieved from
``app.state``, while database-backed services are composed per request around a
request-scoped asynchronous SQLAlchemy session.
"""

from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.api.rate_limiter import RateLimiter
from document_rag_service.core.settings import get_settings
from document_rag_service.db.session import AsyncSessionFactory
from document_rag_service.models.user import User
from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.user_repository import UserRepository
from document_rag_service.services.auth_token_service import AuthTokenService
from document_rag_service.services.chunking_service import ChunkingService
from document_rag_service.services.context_builder_service import (
    ContextBuilderService,
)
from document_rag_service.services.context_expansion_service import (
    ContextExpansionService,
)
from document_rag_service.services.document_service import DocumentService
from document_rag_service.services.embedding_service import EmbeddingService
from document_rag_service.services.generation_service import GenerationService
from document_rag_service.services.ingestion_service import IngestionService
from document_rag_service.services.query_transformation_service import (
    QueryTransformationService,
)
from document_rag_service.services.rag_service import RagService
from document_rag_service.services.retrieval_service import RetrievalService
from document_rag_service.services.user_service import UserService

bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="access_token",
    description=("Paste the access_token returned by POST /v1/auth/login."),
)


def get_session_factory(
    request: Request,
) -> AsyncSessionFactory:
    """
    Return the application-wide asynchronous session factory.

    The session factory is initialized once during application startup and
    stored on ``app.state`` by the lifespan composition root.

    Args:
        request: Current FastAPI request.

    Returns:
        Shared asynchronous SQLAlchemy session factory.
    """

    return cast(
        AsyncSessionFactory,
        request.app.state.session_factory,
    )


async def get_database_session(
    session_factory: Annotated[
        AsyncSessionFactory,
        Depends(get_session_factory),
    ],
) -> AsyncIterator[AsyncSession]:
    """
    Provide one asynchronous database session per HTTP request.

    The session remains available for the duration of the dependency scope.
    Application services own normal transaction boundaries; if an unhandled
    exception escapes the request dependency graph, any outstanding transaction
    is rolled back before the session is released.

    Args:
        session_factory: Shared application session factory.

    Yields:
        Request-scoped asynchronous SQLAlchemy session.
    """

    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def get_embedding_service(
    request: Request,
) -> EmbeddingService:
    """
    Return the shared embedding service.

    The embedding service is created once during application startup so local
    embedding models or remote-provider clients are not recreated for every
    HTTP request.

    Args:
        request: Current FastAPI request.

    Returns:
        Application-wide embedding service.
    """

    return cast(
        EmbeddingService,
        request.app.state.embedding_service,
    )


def get_generation_service(
    request: Request,
) -> GenerationService:
    """
    Return the shared answer-generation service.

    Args:
        request: Current FastAPI request.

    Returns:
        Application-wide generation service initialized during startup.
    """

    return cast(
        GenerationService,
        request.app.state.generation_service,
    )


def get_query_transformation_service(
    request: Request,
) -> QueryTransformationService:
    """
    Return the shared query-transformation service.

    Args:
        request: Current FastAPI request.

    Returns:
        Application-wide query-transformation service initialized during
        startup.
    """

    return cast(
        QueryTransformationService,
        request.app.state.query_transformation_service,
    )


def get_rate_limiter(
    request: Request,
) -> RateLimiter:
    """
    Return the shared in-memory rate limiter.

    A single limiter instance is reused by requests handled by the application
    process so rate-limit buckets persist across individual requests.

    Args:
        request: Current FastAPI request.

    Returns:
        Application-wide rate limiter.
    """

    return cast(
        RateLimiter,
        request.app.state.rate_limiter,
    )


def get_auth_token_service(
    request: Request,
) -> AuthTokenService:
    """
    Return the shared authentication token service.

    Args:
        request: Current FastAPI request.

    Returns:
        Application-wide JWT access-token service.
    """

    return cast(
        AuthTokenService,
        request.app.state.auth_token_service,
    )


DatabaseSession = Annotated[
    AsyncSession,
    Depends(get_database_session),
]

EmbeddingServiceDependency = Annotated[
    EmbeddingService,
    Depends(get_embedding_service),
]

GenerationServiceDependency = Annotated[
    GenerationService,
    Depends(get_generation_service),
]

QueryTransformationServiceDependency = Annotated[
    QueryTransformationService,
    Depends(get_query_transformation_service),
]

AuthTokenServiceDependency = Annotated[
    AuthTokenService,
    Depends(get_auth_token_service),
]

RateLimiterDependency = Annotated[
    RateLimiter,
    Depends(get_rate_limiter),
]


def unauthorized_exception() -> HTTPException:
    """
    Create the standard HTTP 401 authentication error.

    Centralizing the response keeps missing, malformed, expired, and otherwise
    invalid Bearer credentials exposed through the same public API contract.

    Returns:
        HTTPException configured for Bearer authentication.
    """

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={
            "WWW-Authenticate": "Bearer",
        },
    )


async def get_current_user(
    session: DatabaseSession,
    token_service: AuthTokenServiceDependency,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> User:
    """
    Resolve the authenticated active user for the current request.

    The Bearer token is validated first. Its subject identifies the user,
    which is then loaded from the database so a previously issued token cannot
    authenticate an account that has since been deleted or deactivated.

    Args:
        session: Request-scoped database session.
        token_service: Shared access-token service.
        credentials: Bearer credentials extracted from the Authorization
            header.

    Returns:
        Authenticated active user.

    Raises:
        HTTPException: If credentials are missing or invalid, or if the user
            referenced by the token no longer exists or is inactive.
    """

    if credentials is None:
        raise unauthorized_exception()

    try:
        user_id = token_service.get_user_id(
            credentials.credentials,
        )
    except ValueError as exc:
        raise unauthorized_exception() from exc

    user_repository = UserRepository(session)

    user = await user_repository.get_by_id(
        user_id=user_id,
    )

    if user is None or not user.is_active:
        raise unauthorized_exception()

    return user


def get_chunking_service() -> ChunkingService:
    """
    Create a chunking service from validated application settings.

    Chunking is inexpensive and contains no long-lived external resources, so a
    configured service may safely be constructed through the request dependency
    graph.

    Returns:
        Chunking service configured with the application's chunk size and
        overlap.
    """

    settings = get_settings()

    return ChunkingService(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


ChunkingServiceDependency = Annotated[
    ChunkingService,
    Depends(get_chunking_service),
]


def get_document_service(
    session: DatabaseSession,
) -> DocumentService:
    """
    Create a request-scoped document service.

    Args:
        session: Request-scoped database session.

    Returns:
        Document service bound to the current request session.
    """

    return DocumentService(session)


DocumentServiceDependency = Annotated[
    DocumentService,
    Depends(get_document_service),
]


def get_ingestion_service(
    session: DatabaseSession,
    embedding_service: EmbeddingServiceDependency,
    chunking_service: ChunkingServiceDependency,
) -> IngestionService:
    """
    Create a request-scoped document-ingestion service.

    The service receives the current request session while reusing the shared
    embedding service initialized at application startup.

    Args:
        session: Request-scoped database session.
        embedding_service: Shared embedding service.
        chunking_service: Configured chunking service.

    Returns:
        Ingestion service bound to the current request session.
    """

    return IngestionService(
        session,
        embedding_service,
        chunking_service,
    )


def get_rag_service(
    session: DatabaseSession,
    embedding_service: EmbeddingServiceDependency,
    generation_service: GenerationServiceDependency,
    query_transformation_service: QueryTransformationServiceDependency,
) -> RagService:
    """
    Compose the request-scoped RAG orchestration graph.

    Repositories that depend on the database session are created per request,
    while long-lived embedding, generation, and query-transformation resources
    are reused from application state.

    Args:
        session: Request-scoped database session.
        embedding_service: Shared embedding service.
        generation_service: Shared answer-generation service.
        query_transformation_service: Shared query-transformation service.

    Returns:
        RAG service configured for the current request.
    """

    document_repository = DocumentRepository(session)
    chunk_repository = DocumentChunkRepository(session)

    retrieval_service = RetrievalService(
        chunk_repository,
        document_repository,
        embedding_service,
    )

    context_expansion_service = ContextExpansionService(
        chunk_repository,
    )

    context_builder = ContextBuilderService()

    return RagService(
        retrieval_service,
        context_expansion_service,
        context_builder,
        generation_service,
        query_transformation_service=query_transformation_service,
    )


def get_user_service(
    session: DatabaseSession,
) -> UserService:
    """
    Create a request-scoped user service.

    Args:
        session: Request-scoped database session.

    Returns:
        User service bound to the current database session.
    """

    return UserService(session)


UserServiceDependency = Annotated[
    UserService,
    Depends(get_user_service),
]

RagServiceDependency = Annotated[
    RagService,
    Depends(get_rag_service),
]

IngestionServiceDependency = Annotated[
    IngestionService,
    Depends(get_ingestion_service),
]

CurrentUser = Annotated[
    User,
    Depends(get_current_user),
]


def _get_client_ip(request: Request) -> str | None:
    """
    Resolve the client IP used by IP-based rate-limit buckets.

    The application deliberately trusts only ``request.client`` instead of
    reading forwarding headers directly. In production, trusted proxy headers
    should be resolved by the ASGI server before the request reaches the
    application. This prevents clients from bypassing rate limits by supplying
    a forged ``X-Forwarded-For`` header.

    Args:
        request: Current FastAPI request.

    Returns:
        Client IP address when one can be identified, otherwise ``None``.
    """

    if request.client is not None:
        return request.client.host

    return None


async def _enforce_rate_limit(
    *,
    rate_limiter: RateLimiter,
    key: str,
    limit: int,
) -> None:
    """
    Enforce one rate-limit bucket for the current operation.

    Args:
        rate_limiter: Shared in-memory rate limiter.
        key: Stable bucket identifier for the client, user, and operation.
        limit: Maximum requests accepted during the active sliding window.

    Raises:
        HTTPException: If the bucket is exhausted. The response includes a
            ``Retry-After`` header indicating when another request may be
            attempted.
    """

    exceeded = await rate_limiter.check(
        key=key,
        limit=limit,
    )

    if exceeded is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={
                "Retry-After": str(exceeded.retry_after_seconds),
            },
        )


async def enforce_register_rate_limit(
    request: Request,
    rate_limiter: RateLimiterDependency,
) -> None:
    """
    Enforce the per-client registration rate limit.

    Registration is keyed by client IP because no authenticated user exists
    before the account has been created.

    Args:
        request: Current FastAPI request.
        rate_limiter: Shared application rate limiter.

    Raises:
        HTTPException: If the client cannot be identified or its registration
            bucket has exceeded the configured limit.
    """

    client_ip = _get_client_ip(request)

    if client_ip is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to identify the request client.",
        )

    settings = get_settings()

    await _enforce_rate_limit(
        rate_limiter=rate_limiter,
        key=f"auth:register:{client_ip}",
        limit=settings.auth_register_rate_limit_per_minute,
    )


async def enforce_login_rate_limit(
    request: Request,
    rate_limiter: RateLimiterDependency,
) -> None:
    """
    Enforce the per-client authentication rate limit.

    Login attempts are keyed by client IP because authentication has not yet
    established a trusted user identity.

    Args:
        request: Current FastAPI request.
        rate_limiter: Shared application rate limiter.

    Raises:
        HTTPException: If the client cannot be identified or its login bucket
            has exceeded the configured limit.
    """

    client_ip = _get_client_ip(request)

    if client_ip is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to identify the request client.",
        )

    settings = get_settings()

    await _enforce_rate_limit(
        rate_limiter=rate_limiter,
        key=f"auth:login:{client_ip}",
        limit=settings.auth_login_rate_limit_per_minute,
    )


async def enforce_document_upload_rate_limit(
    current_user: CurrentUser,
    rate_limiter: RateLimiterDependency,
) -> None:
    """
    Enforce the per-user document-upload rate limit.

    Document ingestion performs parsing, chunking, embedding generation, and
    database writes, so the endpoint is protected independently from RAG
    queries to limit authenticated resource abuse.

    Args:
        current_user: Authenticated active user.
        rate_limiter: Shared application rate limiter.

    Raises:
        HTTPException: If the user's upload bucket has exceeded the configured
            limit.
    """

    settings = get_settings()

    await _enforce_rate_limit(
        rate_limiter=rate_limiter,
        key=f"documents:upload:user:{current_user.id}",
        limit=settings.document_upload_rate_limit_per_minute,
    )


async def enforce_rag_rate_limit(
    current_user: CurrentUser,
    rate_limiter: RateLimiterDependency,
) -> None:
    """
    Enforce the per-user RAG query rate limit.

    Unlike public authentication operations, RAG requests already have an
    authenticated identity and are therefore bucketed by user identifier rather
    than by client IP.

    Args:
        current_user: Authenticated active user.
        rate_limiter: Shared application rate limiter.

    Raises:
        HTTPException: If the user's RAG query bucket has exceeded the
            configured limit.
    """

    settings = get_settings()

    await _enforce_rate_limit(
        rate_limiter=rate_limiter,
        key=f"rag:query:user:{current_user.id}",
        limit=settings.rag_rate_limit_per_minute,
    )
