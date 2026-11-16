"""
HTTP integration tests for the RAG query route.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.api.dependencies import (
    enforce_login_rate_limit,
    enforce_rag_rate_limit,
    enforce_register_rate_limit,
    get_auth_token_service,
    get_database_session,
    get_embedding_service,
    get_generation_service,
    get_query_transformation_service,
)
from document_rag_service.api.routes import api_router
from document_rag_service.models.document import Document
from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.services.auth_token_service import AuthTokenService
from document_rag_service.services.embedding_service import EmbeddingService
from document_rag_service.services.query_transformation_service import (
    QueryTransformationService,
)

TEST_JWT_SECRET = "integration-test-secret-that-is-long-enough-for-jwt-signing"
EMBEDDING_DIMENSION = 512


def _vector(
    first: float,
    second: float,
) -> list[float]:
    """Build one deterministic production-dimension embedding."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[0] = first
    vector[1] = second
    return vector


class FakeEmbeddingProvider:
    """Provide deterministic query embeddings for HTTP RAG tests."""

    @property
    def model_name(self) -> str:
        """Return a stable model name."""
        return "fake-api-rag-embedding-model"

    @property
    def dimension(self) -> int:
        """Return the production embedding dimension."""
        return EMBEDDING_DIMENSION

    @property
    def max_sequence_length(self) -> int:
        """Return a realistic embedding sequence limit."""
        return 8192

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Return deterministic document embeddings."""
        return [_vector(1.0, 0.0) for _ in texts]

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Map known retrieval queries to deterministic vectors."""
        normalized = text.strip().casefold()

        if "beta" in normalized:
            return _vector(0.0, 1.0)

        return _vector(1.0, 0.0)

    def count_document_tokens(
        self,
        text: str,
    ) -> int:
        """Return a deterministic token count."""
        return len(text.split())

    async def aclose(self) -> None:
        """Release provider resources."""
        return None


class FakeGenerationService:
    """Capture final RAG context and return a deterministic answer."""

    def __init__(self) -> None:
        self.query: str | None = None
        self.context: str | None = None
        self.language: str | None = None

    def get_available_context_characters(
        self,
        *,
        query: str,
        language: str,
    ) -> int:
        """Return a deterministic context budget for RAG orchestration."""
        return 20_000

    async def generate(
        self,
        *,
        query: str,
        context: str,
        language: str,
    ) -> str:
        """Capture generation inputs and return a stable answer."""
        self.query = query
        self.context = context
        self.language = language

        return "HTTP grounded answer"


class FakeQueryTransformationProvider:
    """Return deterministic transformed retrieval queries."""

    async def generate(
        self,
        *,
        system_prompt: str,
        prompt: str,
    ) -> str:
        """Return focused alpha and beta retrieval queries."""
        return '{"queries": ["alpha evidence", "beta evidence"]}'


@dataclass(frozen=True)
class ApiTestApplication:
    """Hold the FastAPI app and test-observable fake services."""

    app: FastAPI
    generation_service: FakeGenerationService


def _build_test_application(
    db_session: AsyncSession,
) -> ApiTestApplication:
    """Build FastAPI using the real RAG composition with fake providers."""
    app = FastAPI()
    app.include_router(api_router)

    embedding_service = EmbeddingService(FakeEmbeddingProvider())

    generation_service = FakeGenerationService()

    query_transformation_service = QueryTransformationService(
        FakeQueryTransformationProvider(),
        max_generated_queries=2,
        max_attempts=2,
    )

    async def override_database_session() -> AsyncIterator[AsyncSession]:
        """Provide the transaction-scoped PostgreSQL integration session."""
        yield db_session

    def override_auth_token_service() -> AuthTokenService:
        """Provide deterministic JWT configuration."""
        return AuthTokenService(
            secret=TEST_JWT_SECRET,
            algorithm="HS256",
            issuer="document-rag-service-test",
            audience="document-rag-service-test",
            expire_minutes=30,
        )

    def override_embedding_service() -> EmbeddingService:
        """Provide deterministic embeddings without network calls."""
        return embedding_service

    def override_generation_service() -> FakeGenerationService:
        """Provide deterministic generation without an external LLM."""
        return generation_service

    def override_query_transformation_service() -> QueryTransformationService:
        """Provide deterministic query transformation."""
        return query_transformation_service

    async def bypass_rate_limit() -> None:
        """Disable rate limiting outside its dedicated tests."""
        return None

    app.dependency_overrides[get_database_session] = override_database_session
    app.dependency_overrides[get_auth_token_service] = override_auth_token_service
    app.dependency_overrides[get_embedding_service] = override_embedding_service
    app.dependency_overrides[get_generation_service] = override_generation_service
    app.dependency_overrides[get_query_transformation_service] = (
        override_query_transformation_service
    )
    app.dependency_overrides[enforce_register_rate_limit] = bypass_rate_limit
    app.dependency_overrides[enforce_login_rate_limit] = bypass_rate_limit
    app.dependency_overrides[enforce_rag_rate_limit] = bypass_rate_limit

    return ApiTestApplication(
        app=app,
        generation_service=generation_service,
    )


async def _build_client(
    application: ApiTestApplication,
) -> AsyncClient:
    """Build an HTTPX client connected directly to FastAPI."""
    return AsyncClient(
        transport=ASGITransport(app=application.app),
        base_url="http://testserver",
    )


async def _register_user(
    client: AsyncClient,
    *,
    email: str,
) -> tuple[int, str]:
    """Register one user and return its id and Bearer token."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 201

    payload = response.json()

    return (
        payload["user"]["id"],
        payload["access_token"],
    )


async def _create_ready_document(
    session: AsyncSession,
    *,
    user_id: int,
    title: str,
) -> Document:
    """Create one ready document directly in PostgreSQL."""
    repository = DocumentRepository(session)

    document = await repository.create(
        user_id=user_id,
        title=title,
        file_name=f"{title}.pdf",
        mime_type="application/pdf",
    )

    await repository.update_status(
        document,
        "ready",
    )

    return document


async def _create_chunk(
    session: AsyncSession,
    *,
    document_id: int,
    chunk_index: int,
    content: str,
    embedding: list[float],
) -> None:
    """Persist one real pgvector-backed chunk."""
    repository = DocumentChunkRepository(session)

    await repository.create(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        embedding=embedding,
        token_count=len(content.split()),
    )


@pytest.mark.asyncio
async def test_rag_query_runs_complete_http_pipeline(
    db_session: AsyncSession,
) -> None:
    """Run an authenticated RAG query through FastAPI and pgvector."""
    application = _build_test_application(db_session)

    async with await _build_client(application) as client:
        user_id, token = await _register_user(
            client,
            email="api-rag-query@example.com",
        )

        document = await _create_ready_document(
            db_session,
            user_id=user_id,
            title="HTTP RAG Source",
        )

        await _create_chunk(
            db_session,
            document_id=document.id,
            chunk_index=0,
            content="Neighbor information before the primary evidence.",
            embedding=_vector(0.0, 1.0),
        )

        await _create_chunk(
            db_session,
            document_id=document.id,
            chunk_index=1,
            content="Alpha evidence is the primary relevant information.",
            embedding=_vector(1.0, 0.0),
        )

        await _create_chunk(
            db_session,
            document_id=document.id,
            chunk_index=2,
            content="Neighbor information after the primary evidence.",
            embedding=_vector(0.0, 1.0),
        )

        response = await client.post(
            "/v1/rag/query",
            headers={
                "Authorization": f"Bearer {token}",
            },
            json={
                "document_ids": [document.id],
                "query": "alpha",
                "language": "en",
                "limit": 1,
                "neighbor_window": 1,
                "transform_query": False,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["answer"] == "HTTP grounded answer"
    assert payload["sources"] == ["HTTP RAG Source"]

    assert payload["retrieval_strategy"] == "dense"
    assert payload["retrieval_queries"] == ["alpha"]

    assert len(payload["retrieval_hits"]) == 1

    hit = payload["retrieval_hits"][0]

    assert hit["document_id"] == document.id
    assert hit["document_title"] == "HTTP RAG Source"
    assert hit["chunk_index"] == 1
    assert hit["best_similarity"] > 0.99

    assert payload["context_chunks_count"] == 3
    assert payload["context_characters"] > 0
    assert payload["max_context_characters"] > 0

    assert application.generation_service.query == "alpha"
    assert application.generation_service.language == "en"

    assert application.generation_service.context is not None

    assert (
        "Neighbor information before the primary evidence."
        in application.generation_service.context
    )
    assert (
        "Alpha evidence is the primary relevant information."
        in application.generation_service.context
    )
    assert (
        "Neighbor information after the primary evidence."
        in application.generation_service.context
    )


@pytest.mark.asyncio
async def test_rag_query_with_transformation_uses_multi_query_rrf(
    db_session: AsyncSession,
) -> None:
    """Expose transformed multi-query RRF retrieval through HTTP."""
    application = _build_test_application(db_session)

    async with await _build_client(application) as client:
        user_id, token = await _register_user(
            client,
            email="api-rag-transformed@example.com",
        )

        document = await _create_ready_document(
            db_session,
            user_id=user_id,
            title="HTTP Multi Query Source",
        )

        await _create_chunk(
            db_session,
            document_id=document.id,
            chunk_index=0,
            content="Strong alpha evidence.",
            embedding=_vector(1.0, 0.0),
        )

        await _create_chunk(
            db_session,
            document_id=document.id,
            chunk_index=1,
            content="Strong beta evidence.",
            embedding=_vector(0.0, 1.0),
        )

        response = await client.post(
            "/v1/rag/query",
            headers={
                "Authorization": f"Bearer {token}",
            },
            json={
                "document_ids": [document.id],
                "query": "Find alpha and beta evidence",
                "language": "en",
                "limit": 2,
                "neighbor_window": 0,
                "transform_query": True,
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["answer"] == "HTTP grounded answer"
    assert payload["retrieval_strategy"] == "multi_query_rrf"

    assert payload["retrieval_queries"] == [
        "Find alpha and beta evidence",
        "alpha evidence",
        "beta evidence",
    ]

    assert payload["sources"] == [
        "HTTP Multi Query Source",
    ]

    assert len(payload["retrieval_hits"]) == 2

    retrieved_chunk_indexes = {hit["chunk_index"] for hit in payload["retrieval_hits"]}

    assert retrieved_chunk_indexes == {
        0,
        1,
    }

    assert application.generation_service.context is not None

    assert "Strong alpha evidence." in application.generation_service.context
    assert "Strong beta evidence." in application.generation_service.context


@pytest.mark.asyncio
async def test_rag_query_hides_another_users_document(
    db_session: AsyncSession,
) -> None:
    """Return 404 when a user selects another user's document."""
    application = _build_test_application(db_session)

    async with await _build_client(application) as client:
        owner_id, _ = await _register_user(
            client,
            email="api-rag-owner@example.com",
        )

        _, other_token = await _register_user(
            client,
            email="api-rag-other@example.com",
        )

        document = await _create_ready_document(
            db_session,
            user_id=owner_id,
            title="Private HTTP RAG Source",
        )

        await _create_chunk(
            db_session,
            document_id=document.id,
            chunk_index=0,
            content="Private alpha evidence.",
            embedding=_vector(1.0, 0.0),
        )

        response = await client.post(
            "/v1/rag/query",
            headers={
                "Authorization": f"Bearer {other_token}",
            },
            json={
                "document_ids": [document.id],
                "query": "alpha",
                "language": "en",
                "limit": 1,
                "neighbor_window": 0,
                "transform_query": False,
            },
        )

    assert response.status_code == 404

    assert response.json() == {
        "detail": "One or more selected documents were not found",
    }


@pytest.mark.asyncio
async def test_rag_query_requires_authentication(
    db_session: AsyncSession,
) -> None:
    """Reject anonymous RAG queries."""
    application = _build_test_application(db_session)

    async with await _build_client(application) as client:
        response = await client.post(
            "/v1/rag/query",
            json={
                "document_ids": [1],
                "query": "alpha",
                "language": "en",
                "limit": 1,
                "neighbor_window": 0,
                "transform_query": False,
            },
        )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid authentication credentials",
    }


@pytest.mark.asyncio
async def test_rag_query_rejects_pending_document(
    db_session: AsyncSession,
) -> None:
    """Return HTTP 400 when a selected document is not ready."""
    application = _build_test_application(db_session)

    async with await _build_client(application) as client:
        user_id, token = await _register_user(
            client,
            email="api-rag-pending@example.com",
        )

        repository = DocumentRepository(db_session)

        document = await repository.create(
            user_id=user_id,
            title="Pending HTTP RAG Source",
            file_name="pending.pdf",
            mime_type="application/pdf",
        )

        response = await client.post(
            "/v1/rag/query",
            headers={
                "Authorization": f"Bearer {token}",
            },
            json={
                "document_ids": [document.id],
                "query": "alpha",
                "language": "en",
                "limit": 1,
                "neighbor_window": 0,
                "transform_query": False,
            },
        )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "All selected documents must be ready",
    }
