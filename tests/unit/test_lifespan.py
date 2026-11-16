"""Tests for FastAPI lifespan resource composition and cleanup."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncEngine

from document_rag_service.core import lifespan as lifespan_module
from document_rag_service.core.lifespan import lifespan
from document_rag_service.core.settings import Settings


def _build_settings() -> MagicMock:
    """Create deterministic application settings for lifespan tests."""
    settings = MagicMock(spec=Settings)

    settings.auth_jwt_secret = SecretStr("a-very-long-test-jwt-secret-value")
    settings.auth_jwt_algorithm = "HS256"
    settings.auth_jwt_issuer = "document-rag-service"
    settings.auth_jwt_audience = "document-rag-api"
    settings.auth_access_token_expire_minutes = 30

    settings.llm_inference_base_url = "https://llm.example.com"
    settings.llm_inference_timeout_seconds = 120.0
    settings.llm_inference_provider = "groq"
    settings.llm_inference_model = "test-model"
    settings.llm_inference_temperature = 0.0
    settings.llm_inference_max_tokens = 2000
    settings.llm_inference_max_prompt_characters = 20_000

    settings.query_transformation_temperature = 0.0
    settings.query_transformation_max_tokens = 500
    settings.query_transformation_max_generated_queries = 2
    settings.query_transformation_max_attempts = 2

    return settings


def _configure_lifespan_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Replace lifespan infrastructure with deterministic test doubles."""
    settings = _build_settings()

    rate_limiter = MagicMock()
    rate_limiter_factory = MagicMock(
        return_value=rate_limiter,
    )

    auth_token_service = MagicMock()
    auth_token_factory = MagicMock(
        return_value=auth_token_service,
    )

    database_engine = MagicMock(spec=AsyncEngine)
    database_engine.dispose = AsyncMock()

    database_engine_factory = MagicMock(
        return_value=database_engine,
    )

    session_factory = MagicMock()

    session_factory_builder = MagicMock(
        return_value=session_factory,
    )

    embedding_provider = MagicMock()

    embedding_provider_factory = MagicMock(
        return_value=embedding_provider,
    )

    embedding_service = MagicMock()
    embedding_service.aclose = AsyncMock()

    embedding_service_factory = MagicMock(
        return_value=embedding_service,
    )

    llm_http_client = MagicMock()
    llm_http_client.aclose = AsyncMock()

    http_client_factory = MagicMock(
        return_value=llm_http_client,
    )

    generation_provider = MagicMock()
    transformation_provider = MagicMock()

    llm_provider_factory = MagicMock(
        side_effect=[
            generation_provider,
            transformation_provider,
        ]
    )

    generation_service = MagicMock()

    generation_service_factory = MagicMock(
        return_value=generation_service,
    )

    query_transformation_service = MagicMock()

    query_transformation_service_factory = MagicMock(
        return_value=query_transformation_service,
    )

    monkeypatch.setattr(
        lifespan_module,
        "get_settings",
        MagicMock(return_value=settings),
    )

    monkeypatch.setattr(
        lifespan_module,
        "RateLimiter",
        rate_limiter_factory,
    )

    monkeypatch.setattr(
        lifespan_module,
        "AuthTokenService",
        auth_token_factory,
    )

    monkeypatch.setattr(
        lifespan_module,
        "create_database_engine",
        database_engine_factory,
    )

    monkeypatch.setattr(
        lifespan_module,
        "create_session_factory",
        session_factory_builder,
    )

    monkeypatch.setattr(
        lifespan_module,
        "create_embedding_provider",
        embedding_provider_factory,
    )

    monkeypatch.setattr(
        lifespan_module,
        "EmbeddingService",
        embedding_service_factory,
    )

    monkeypatch.setattr(
        lifespan_module.httpx,
        "AsyncClient",
        http_client_factory,
    )

    monkeypatch.setattr(
        lifespan_module,
        "LLMInferenceProvider",
        llm_provider_factory,
    )

    monkeypatch.setattr(
        lifespan_module,
        "GenerationService",
        generation_service_factory,
    )

    monkeypatch.setattr(
        lifespan_module,
        "QueryTransformationService",
        query_transformation_service_factory,
    )

    return {
        "settings": settings,
        "rate_limiter": rate_limiter,
        "rate_limiter_factory": rate_limiter_factory,
        "auth_token_service": auth_token_service,
        "auth_token_factory": auth_token_factory,
        "database_engine": database_engine,
        "database_engine_factory": database_engine_factory,
        "session_factory": session_factory,
        "session_factory_builder": session_factory_builder,
        "embedding_provider": embedding_provider,
        "embedding_provider_factory": embedding_provider_factory,
        "embedding_service": embedding_service,
        "embedding_service_factory": embedding_service_factory,
        "llm_http_client": llm_http_client,
        "http_client_factory": http_client_factory,
        "generation_provider": generation_provider,
        "transformation_provider": transformation_provider,
        "llm_provider_factory": llm_provider_factory,
        "generation_service": generation_service,
        "generation_service_factory": generation_service_factory,
        "query_transformation_service": (query_transformation_service),
        "query_transformation_service_factory": (query_transformation_service_factory),
    }


@pytest.mark.asyncio
async def test_lifespan_creates_rate_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create one application-level rate limiter during startup."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        assert app.state.rate_limiter is dependencies["rate_limiter"]

    rate_limiter_factory = dependencies["rate_limiter_factory"]

    assert isinstance(
        rate_limiter_factory,
        MagicMock,
    )

    rate_limiter_factory.assert_called_once_with()


@pytest.mark.asyncio
async def test_lifespan_builds_auth_token_service_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build JWT authentication with validated application settings."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        pass

    auth_token_factory = dependencies["auth_token_factory"]

    assert isinstance(
        auth_token_factory,
        MagicMock,
    )

    auth_token_factory.assert_called_once_with(
        secret="a-very-long-test-jwt-secret-value",
        algorithm="HS256",
        issuer="document-rag-service",
        audience="document-rag-api",
        expire_minutes=30,
    )


@pytest.mark.asyncio
async def test_lifespan_builds_database_engine_and_session_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose the database engine and asynchronous session factory."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        assert app.state.session_factory is dependencies["session_factory"]

    database_engine_factory = dependencies["database_engine_factory"]
    session_factory_builder = dependencies["session_factory_builder"]

    assert isinstance(
        database_engine_factory,
        MagicMock,
    )
    assert isinstance(
        session_factory_builder,
        MagicMock,
    )

    database_engine_factory.assert_called_once_with(dependencies["settings"])

    session_factory_builder.assert_called_once_with(dependencies["database_engine"])


@pytest.mark.asyncio
async def test_lifespan_builds_embedding_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose the selected embedding provider into its service."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        assert app.state.embedding_service is dependencies["embedding_service"]

    embedding_provider_factory = dependencies["embedding_provider_factory"]
    embedding_service_factory = dependencies["embedding_service_factory"]

    assert isinstance(
        embedding_provider_factory,
        MagicMock,
    )
    assert isinstance(
        embedding_service_factory,
        MagicMock,
    )

    embedding_provider_factory.assert_called_once_with(dependencies["settings"])

    embedding_service_factory.assert_called_once_with(
        dependencies["embedding_provider"]
    )


@pytest.mark.asyncio
async def test_lifespan_creates_llm_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create one reusable asynchronous HTTP client for LLM requests."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        pass

    http_client_factory = dependencies["http_client_factory"]

    assert isinstance(
        http_client_factory,
        MagicMock,
    )

    http_client_factory.assert_called_once_with(
        base_url="https://llm.example.com/",
        timeout=120.0,
    )


@pytest.mark.asyncio
async def test_lifespan_normalizes_llm_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the shared LLM HTTP client receives one trailing slash."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    settings = dependencies["settings"]

    assert isinstance(
        settings,
        MagicMock,
    )

    settings.llm_inference_base_url = "https://llm.example.com///"

    app = FastAPI()

    async with lifespan(app):
        pass

    http_client_factory = dependencies["http_client_factory"]

    assert isinstance(
        http_client_factory,
        MagicMock,
    )

    assert (
        http_client_factory.call_args.kwargs["base_url"] == "https://llm.example.com/"
    )


@pytest.mark.asyncio
async def test_lifespan_builds_generation_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configure the generation provider from LLM settings."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        pass

    llm_provider_factory = dependencies["llm_provider_factory"]

    assert isinstance(
        llm_provider_factory,
        MagicMock,
    )

    assert llm_provider_factory.call_args_list[0] == call(
        client=dependencies["llm_http_client"],
        provider="groq",
        model="test-model",
        temperature=0.0,
        max_tokens=2000,
    )


@pytest.mark.asyncio
async def test_lifespan_builds_query_transformation_llm_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configure a separate LLM adapter for query transformation."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        pass

    llm_provider_factory = dependencies["llm_provider_factory"]

    assert isinstance(
        llm_provider_factory,
        MagicMock,
    )

    assert llm_provider_factory.call_args_list[1] == call(
        client=dependencies["llm_http_client"],
        provider="groq",
        model="test-model",
        temperature=0.0,
        max_tokens=500,
    )


@pytest.mark.asyncio
async def test_lifespan_reuses_same_http_client_for_both_llm_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Share one HTTP connection pool across LLM provider adapters."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        pass

    llm_provider_factory = dependencies["llm_provider_factory"]

    assert isinstance(
        llm_provider_factory,
        MagicMock,
    )

    first_client = llm_provider_factory.call_args_list[0].kwargs["client"]
    second_client = llm_provider_factory.call_args_list[1].kwargs["client"]

    assert first_client is dependencies["llm_http_client"]
    assert second_client is dependencies["llm_http_client"]


@pytest.mark.asyncio
async def test_lifespan_builds_generation_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose generation provider and prompt budget into GenerationService."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        assert app.state.generation_service is dependencies["generation_service"]

    generation_service_factory = dependencies["generation_service_factory"]

    assert isinstance(
        generation_service_factory,
        MagicMock,
    )

    generation_service_factory.assert_called_once_with(
        dependencies["generation_provider"],
        max_prompt_characters=20_000,
    )


@pytest.mark.asyncio
async def test_lifespan_builds_query_transformation_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compose transformation provider with configured query limits."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        assert (
            app.state.query_transformation_service
            is dependencies["query_transformation_service"]
        )

    factory = dependencies["query_transformation_service_factory"]

    assert isinstance(
        factory,
        MagicMock,
    )

    factory.assert_called_once_with(
        dependencies["transformation_provider"],
        max_generated_queries=2,
        max_attempts=2,
    )


@pytest.mark.asyncio
async def test_lifespan_exposes_auth_service_in_application_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the long-lived authentication service through app state."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        assert app.state.auth_token_service is dependencies["auth_token_service"]


@pytest.mark.asyncio
async def test_lifespan_closes_llm_http_client_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close the shared LLM HTTP client when the application shuts down."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        pass

    llm_http_client = dependencies["llm_http_client"]

    assert isinstance(
        llm_http_client,
        MagicMock,
    )

    llm_http_client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_lifespan_closes_embedding_service_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release embedding-provider resources during shutdown."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        pass

    embedding_service = dependencies["embedding_service"]

    assert isinstance(
        embedding_service,
        MagicMock,
    )

    embedding_service.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_lifespan_disposes_database_engine_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispose the database engine after application shutdown."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    async with lifespan(app):
        pass

    database_engine = dependencies["database_engine"]

    assert isinstance(
        database_engine,
        MagicMock,
    )

    database_engine.dispose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_lifespan_cleans_up_resources_when_application_body_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release resources even when the running application raises."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    app = FastAPI()

    with pytest.raises(
        RuntimeError,
        match="application failure",
    ):
        async with lifespan(app):
            raise RuntimeError("application failure")

    llm_http_client = dependencies["llm_http_client"]
    embedding_service = dependencies["embedding_service"]
    database_engine = dependencies["database_engine"]

    assert isinstance(
        llm_http_client,
        MagicMock,
    )
    assert isinstance(
        embedding_service,
        MagicMock,
    )
    assert isinstance(
        database_engine,
        MagicMock,
    )

    llm_http_client.aclose.assert_awaited_once_with()
    embedding_service.aclose.assert_awaited_once_with()
    database_engine.dispose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_lifespan_disposes_engine_when_embedding_provider_startup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispose database infrastructure after an early startup failure."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    startup_error = RuntimeError("embedding provider startup failed")

    embedding_provider_factory = dependencies["embedding_provider_factory"]

    assert isinstance(
        embedding_provider_factory,
        MagicMock,
    )

    embedding_provider_factory.side_effect = startup_error

    app = FastAPI()

    with pytest.raises(
        RuntimeError,
        match="embedding provider startup failed",
    ) as exc_info:
        async with lifespan(app):
            pass

    assert exc_info.value is startup_error

    database_engine = dependencies["database_engine"]
    embedding_service = dependencies["embedding_service"]
    llm_http_client = dependencies["llm_http_client"]

    assert isinstance(
        database_engine,
        MagicMock,
    )
    assert isinstance(
        embedding_service,
        MagicMock,
    )
    assert isinstance(
        llm_http_client,
        MagicMock,
    )

    database_engine.dispose.assert_awaited_once_with()

    embedding_service.aclose.assert_not_awaited()
    llm_http_client.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_lifespan_closes_embedding_service_when_http_client_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release embeddings when startup fails after their creation."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    startup_error = RuntimeError("HTTP client startup failed")

    http_client_factory = dependencies["http_client_factory"]

    assert isinstance(
        http_client_factory,
        MagicMock,
    )

    http_client_factory.side_effect = startup_error

    app = FastAPI()

    with pytest.raises(
        RuntimeError,
        match="HTTP client startup failed",
    ) as exc_info:
        async with lifespan(app):
            pass

    assert exc_info.value is startup_error

    embedding_service = dependencies["embedding_service"]
    database_engine = dependencies["database_engine"]

    assert isinstance(
        embedding_service,
        MagicMock,
    )
    assert isinstance(
        database_engine,
        MagicMock,
    )

    embedding_service.aclose.assert_awaited_once_with()
    database_engine.dispose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_lifespan_closes_all_created_resources_when_service_composition_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean up resources when service composition fails before startup completes."""
    dependencies = _configure_lifespan_dependencies(monkeypatch)

    startup_error = RuntimeError("generation service startup failed")

    generation_service_factory = dependencies["generation_service_factory"]

    assert isinstance(
        generation_service_factory,
        MagicMock,
    )

    generation_service_factory.side_effect = startup_error

    app = FastAPI()

    with pytest.raises(
        RuntimeError,
        match="generation service startup failed",
    ) as exc_info:
        async with lifespan(app):
            pass

    assert exc_info.value is startup_error

    llm_http_client = dependencies["llm_http_client"]
    embedding_service = dependencies["embedding_service"]
    database_engine = dependencies["database_engine"]

    assert isinstance(
        llm_http_client,
        MagicMock,
    )
    assert isinstance(
        embedding_service,
        MagicMock,
    )
    assert isinstance(
        database_engine,
        MagicMock,
    )

    llm_http_client.aclose.assert_awaited_once_with()
    embedding_service.aclose.assert_awaited_once_with()
    database_engine.dispose.assert_awaited_once_with()
