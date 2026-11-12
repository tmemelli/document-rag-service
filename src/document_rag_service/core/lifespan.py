"""
FastAPI application lifespan and long-lived resource composition.

This module acts as the application's composition root. It creates shared
infrastructure and service instances once during startup, exposes them through
``app.state`` for request-time dependency resolution, and releases owned
resources in a deterministic order during shutdown.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from document_rag_service.api.rate_limiter import RateLimiter
from document_rag_service.core.embedding_provider_factory import (
    create_embedding_provider,
)
from document_rag_service.core.settings import get_settings
from document_rag_service.db.session import (
    create_database_engine,
    create_session_factory,
)
from document_rag_service.providers.llm_inference_provider import (
    LLMInferenceProvider,
)
from document_rag_service.services.auth_token_service import AuthTokenService
from document_rag_service.services.embedding_service import EmbeddingService
from document_rag_service.services.generation_service import GenerationService
from document_rag_service.services.query_transformation_service import (
    QueryTransformationService,
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[None]:
    """
    Create, expose, and release long-lived application resources.

    The lifespan function is the application's composition root. It loads the
    validated configuration, creates shared infrastructure resources and
    application services, and stores request-independent dependencies on
    ``app.state`` so they can be reused across requests.

    Database engine and session-factory construction happen once at startup.
    The embedding provider is also created once, preventing repeated model or
    HTTP-client initialization. Generation and query-transformation providers
    share a single asynchronous HTTP client so connection pooling and inference
    service configuration are reused efficiently.

    Shutdown follows the reverse resource-lifecycle path: the shared LLM HTTP
    client is closed first, followed by the embedding service and finally the
    database engine.

    Args:
        app: FastAPI application instance whose state stores shared resources
            and services for request-time dependency resolution.

    Yields:
        Control to FastAPI while all application resources remain initialized
        and available.

    Raises:
        Exception: Propagates startup failures from configuration, database,
            embedding-provider, or service initialization so the application
            does not start in a partially initialized state.
    """

    settings = get_settings()
    rate_limiter = RateLimiter()

    auth_token_service = AuthTokenService(
        secret=settings.auth_jwt_secret.get_secret_value(),
        algorithm=settings.auth_jwt_algorithm,
        issuer=settings.auth_jwt_issuer,
        audience=settings.auth_jwt_audience,
        expire_minutes=settings.auth_access_token_expire_minutes,
    )

    database_engine = create_database_engine(settings)
    session_factory = create_session_factory(database_engine)

    embedding_service: EmbeddingService | None = None
    llm_http_client: httpx.AsyncClient | None = None

    try:
        embedding_provider = create_embedding_provider(settings)
        embedding_service = EmbeddingService(embedding_provider)

        llm_http_client = httpx.AsyncClient(
            base_url=f"{settings.llm_inference_base_url.rstrip('/')}/",
            timeout=settings.llm_inference_timeout_seconds,
        )

        generation_provider = LLMInferenceProvider(
            client=llm_http_client,
            provider=settings.llm_inference_provider,
            model=settings.llm_inference_model,
            temperature=settings.llm_inference_temperature,
            max_tokens=settings.llm_inference_max_tokens,
        )

        transformation_provider = LLMInferenceProvider(
            client=llm_http_client,
            provider=settings.llm_inference_provider,
            model=settings.llm_inference_model,
            temperature=settings.query_transformation_temperature,
            max_tokens=settings.query_transformation_max_tokens,
        )

        generation_service = GenerationService(
            generation_provider,
            max_prompt_characters=settings.llm_inference_max_prompt_characters,
        )

        query_transformation_service = QueryTransformationService(
            transformation_provider,
            max_generated_queries=settings.query_transformation_max_generated_queries,
            max_attempts=settings.query_transformation_max_attempts,
        )

        app.state.session_factory = session_factory
        app.state.embedding_service = embedding_service
        app.state.generation_service = generation_service
        app.state.query_transformation_service = query_transformation_service
        app.state.auth_token_service = auth_token_service
        app.state.rate_limiter = rate_limiter

        yield

    finally:
        if llm_http_client is not None:
            await llm_http_client.aclose()

        if embedding_service is not None:
            await embedding_service.aclose()

        await database_engine.dispose()
