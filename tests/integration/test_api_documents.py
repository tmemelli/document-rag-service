"""
HTTP integration tests for document routes.
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

import document_rag_service.api.routes as routes_module
from document_rag_service.api.dependencies import (
    enforce_document_upload_rate_limit,
    enforce_login_rate_limit,
    enforce_register_rate_limit,
    get_auth_token_service,
    get_database_session,
    get_embedding_service,
    get_session_factory,
)
from document_rag_service.api.routes import api_router
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.error_event_repository import (
    ErrorEventRepository,
)
from document_rag_service.repositories.user_repository import UserRepository
from document_rag_service.services.auth_token_service import AuthTokenService
from document_rag_service.services.document_service import DocumentService
from document_rag_service.services.embedding_service import EmbeddingService

TEST_JWT_SECRET = "integration-test-secret-that-is-long-enough-for-jwt-signing"
EMBEDDING_DIMENSION = 512


def _rtf_content(text: str) -> bytes:
    """Build a minimal valid RTF document."""
    escaped_text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

    return f"{{\\rtf1\\ansi {escaped_text}}}".encode()


class FakeEmbeddingProvider:
    """Deterministic embedding provider for document HTTP tests."""

    @property
    def model_name(self) -> str:
        """Return a stable fake model name."""
        return "fake-api-document-embedding-model"

    @property
    def dimension(self) -> int:
        """Return the production embedding dimension."""
        return EMBEDDING_DIMENSION

    @property
    def max_sequence_length(self) -> int:
        """Return a realistic sequence limit."""
        return 8192

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Return deterministic document embeddings."""
        embeddings: list[list[float]] = []

        for index, _ in enumerate(texts):
            vector = [0.0] * EMBEDDING_DIMENSION
            vector[0] = 1.0
            vector[1] = float(index + 1) / 100.0
            embeddings.append(vector)

        return embeddings

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Return one deterministic query embedding."""
        vector = [0.0] * EMBEDDING_DIMENSION
        vector[0] = 1.0
        return vector

    def count_document_tokens(
        self,
        text: str,
    ) -> int:
        """Return a deterministic token count."""
        return len(text.split())

    async def aclose(self) -> None:
        """Release provider resources."""
        return None


def _build_test_app(
    db_session: AsyncSession,
) -> FastAPI:
    """Build an isolated FastAPI app backed by the integration database."""
    app = FastAPI()
    app.include_router(api_router)

    embedding_service = EmbeddingService(FakeEmbeddingProvider())
    session_factory = MagicMock()

    async def override_database_session() -> AsyncIterator[AsyncSession]:
        """Provide the transaction-scoped integration session."""
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
        """Provide deterministic embeddings without external providers."""
        return embedding_service

    async def bypass_rate_limit() -> None:
        """Disable unrelated rate limiting in these tests."""
        return None

    app.dependency_overrides[get_database_session] = override_database_session
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    app.dependency_overrides[get_auth_token_service] = override_auth_token_service
    app.dependency_overrides[get_embedding_service] = override_embedding_service
    app.dependency_overrides[enforce_register_rate_limit] = bypass_rate_limit
    app.dependency_overrides[enforce_login_rate_limit] = bypass_rate_limit
    app.dependency_overrides[enforce_document_upload_rate_limit] = bypass_rate_limit

    return app


async def _build_client(
    db_session: AsyncSession,
) -> AsyncClient:
    """Build an HTTP client connected directly to FastAPI."""
    app = _build_test_app(db_session)

    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    )


async def _register_user(
    client: AsyncClient,
    *,
    email: str,
) -> str:
    """Register one user and return its Bearer token."""
    response = await client.post(
        "/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPassword123!",
        },
    )

    assert response.status_code == 201

    return response.json()["access_token"]


async def _upload_document(
    client: AsyncClient,
    *,
    token: str,
    file_name: str,
    title: str,
    text: str,
):
    """Upload and ingest one RTF document."""
    return await client.post(
        "/v1/documents/upload",
        headers={
            "Authorization": f"Bearer {token}",
        },
        data={
            "title": title,
        },
        files={
            "file": (
                file_name,
                _rtf_content(text),
                "application/rtf",
            ),
        },
    )


@pytest.mark.asyncio
async def test_upload_document_ingests_and_returns_ready_document(
    db_session: AsyncSession,
) -> None:
    """Upload a supported document through the complete HTTP pipeline."""
    async with await _build_client(db_session) as client:
        token = await _register_user(
            client,
            email="api-document-upload@example.com",
        )

        response = await _upload_document(
            client,
            token=token,
            file_name="architecture.rtf",
            title="AI Architecture",
            text=(
                "Retrieval augmented generation combines semantic retrieval "
                "with language model generation. Neighboring chunks provide "
                "additional context for grounded answers."
            ),
        )

    assert response.status_code == 201

    payload = response.json()

    assert isinstance(payload["id"], int)
    assert payload["title"] == "AI Architecture"
    assert payload["file_name"] == "architecture.rtf"
    assert payload["mime_type"] == "application/rtf"
    assert payload["status"] == "ready"


@pytest.mark.asyncio
async def test_upload_retries_failed_document_without_creating_duplicate(
    db_session: AsyncSession,
) -> None:
    """Re-upload failed content and reuse the existing document resource."""
    email = "api-document-retry@example.com"
    content = _rtf_content(
        "A failed ingestion can be retried later using the same uploaded bytes."
    )

    async with await _build_client(db_session) as client:
        token = await _register_user(client, email=email)

        user = await UserRepository(db_session).get_by_email(email)
        assert user is not None

        document_service = DocumentService(db_session)
        failed_document = await document_service.register_uploaded_document(
            user_id=user.id,
            title="Retry through API",
            file_name="retry-api.rtf",
            mime_type="application/rtf",
            content=content,
        )
        await document_service.update_status(
            document_id=failed_document.id,
            user_id=user.id,
            status="failed",
        )

        response = await client.post(
            "/v1/documents/upload",
            headers={"Authorization": f"Bearer {token}"},
            data={"title": "Retry through API"},
            files={
                "file": (
                    "retry-api.rtf",
                    content,
                    "application/rtf",
                )
            },
        )

        listed = await client.get(
            "/v1/documents",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 201
    assert response.json()["id"] == failed_document.id
    assert response.json()["status"] == "ready"
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [failed_document.id]


@pytest.mark.asyncio
async def test_upload_recovers_interrupted_processing_document(
    db_session: AsyncSession,
) -> None:
    """Re-upload legacy or expired processing content and resume with same id."""
    email = "api-document-interrupted@example.com"
    content = _rtf_content(
        "Interrupted ingestion should be recoverable without creating a duplicate."
    )

    async with await _build_client(db_session) as client:
        token = await _register_user(client, email=email)

        user = await UserRepository(db_session).get_by_email(email)
        assert user is not None

        document_service = DocumentService(db_session)
        processing_document = await document_service.register_uploaded_document(
            user_id=user.id,
            title="Interrupted through API",
            file_name="interrupted-api.rtf",
            mime_type="application/rtf",
            content=content,
        )
        await document_service.update_status(
            document_id=processing_document.id,
            user_id=user.id,
            status="processing",
        )

        response = await client.post(
            "/v1/documents/upload",
            headers={"Authorization": f"Bearer {token}"},
            data={"title": "Interrupted through API"},
            files={
                "file": (
                    "interrupted-api.rtf",
                    content,
                    "application/rtf",
                )
            },
        )

        errors = await ErrorEventRepository(db_session).list_by_document_id(
            processing_document.id
        )

    assert response.status_code == 201
    assert response.json()["id"] == processing_document.id
    assert response.json()["status"] == "ready"
    assert len(errors) == 1
    assert errors[0].error_code == "INGESTION_INTERRUPTED"
    assert errors[0].retryable is True


@pytest.mark.asyncio
async def test_upload_rejects_duplicate_with_active_processing_lease(
    db_session: AsyncSession,
) -> None:
    """Keep duplicate upload blocked while a processing heartbeat is valid."""
    email = "api-document-active-processing@example.com"
    content = _rtf_content(
        "An active processing lease must block a concurrent duplicate upload."
    )

    async with await _build_client(db_session) as client:
        token = await _register_user(client, email=email)

        user = await UserRepository(db_session).get_by_email(email)
        assert user is not None

        document_service = DocumentService(db_session)
        document = await document_service.register_uploaded_document(
            user_id=user.id,
            title="Active processing",
            file_name="active-processing.rtf",
            mime_type="application/rtf",
            content=content,
        )

        repository = DocumentRepository(db_session)
        claimed = await repository.claim_pending_for_processing(
            document_id=document.id,
            user_id=user.id,
            processing_token="a" * 32,
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=2),
        )
        assert claimed is not None
        await db_session.commit()

        response = await client.post(
            "/v1/documents/upload",
            headers={"Authorization": f"Bearer {token}"},
            data={"title": "Active processing"},
            files={
                "file": (
                    "active-processing.rtf",
                    content,
                    "application/rtf",
                )
            },
        )

    assert response.status_code == 409
    assert "already being processed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_documents_returns_only_authenticated_users_documents(
    db_session: AsyncSession,
) -> None:
    """List documents exclusively for the authenticated owner."""
    async with await _build_client(db_session) as client:
        first_token = await _register_user(
            client,
            email="api-documents-owner@example.com",
        )
        second_token = await _register_user(
            client,
            email="api-documents-other@example.com",
        )

        first_upload = await _upload_document(
            client,
            token=first_token,
            file_name="first.rtf",
            title="First Owner Document",
            text="First owner's private document content.",
        )
        second_upload = await _upload_document(
            client,
            token=first_token,
            file_name="second.rtf",
            title="Second Owner Document",
            text="Second document belonging to the first owner.",
        )
        other_upload = await _upload_document(
            client,
            token=second_token,
            file_name="other.rtf",
            title="Other User Document",
            text="Content belonging to another user.",
        )

        assert first_upload.status_code == 201
        assert second_upload.status_code == 201
        assert other_upload.status_code == 201

        response = await client.get(
            "/v1/documents",
            headers={
                "Authorization": f"Bearer {first_token}",
            },
        )

    assert response.status_code == 200

    documents = response.json()

    assert len(documents) == 2

    titles = {document["title"] for document in documents}

    assert titles == {
        "First Owner Document",
        "Second Owner Document",
    }

    assert "Other User Document" not in titles


@pytest.mark.asyncio
async def test_get_document_returns_owned_document(
    db_session: AsyncSession,
) -> None:
    """Retrieve one document through its authenticated HTTP endpoint."""
    async with await _build_client(db_session) as client:
        token = await _register_user(
            client,
            email="api-document-get@example.com",
        )

        upload_response = await _upload_document(
            client,
            token=token,
            file_name="get-document.rtf",
            title="Get Document",
            text="Document content used by the get endpoint test.",
        )

        assert upload_response.status_code == 201

        document_id = upload_response.json()["id"]

        response = await client.get(
            f"/v1/documents/{document_id}",
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["id"] == document_id
    assert payload["title"] == "Get Document"
    assert payload["status"] == "ready"


@pytest.mark.asyncio
async def test_get_document_hides_another_users_document(
    db_session: AsyncSession,
) -> None:
    """Return 404 instead of exposing another user's document."""
    async with await _build_client(db_session) as client:
        owner_token = await _register_user(
            client,
            email="api-get-owner@example.com",
        )
        other_token = await _register_user(
            client,
            email="api-get-other@example.com",
        )

        upload_response = await _upload_document(
            client,
            token=owner_token,
            file_name="private.rtf",
            title="Private Document",
            text="Private document that must not cross user boundaries.",
        )

        assert upload_response.status_code == 201

        document_id = upload_response.json()["id"]

        response = await client.get(
            f"/v1/documents/{document_id}",
            headers={
                "Authorization": f"Bearer {other_token}",
            },
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Document not found",
    }


@pytest.mark.asyncio
async def test_delete_document_removes_owned_document(
    db_session: AsyncSession,
) -> None:
    """Delete an owned document through HTTP."""
    async with await _build_client(db_session) as client:
        token = await _register_user(
            client,
            email="api-document-delete@example.com",
        )

        upload_response = await _upload_document(
            client,
            token=token,
            file_name="delete.rtf",
            title="Delete Me",
            text="Document that will be deleted through the API.",
        )

        assert upload_response.status_code == 201

        document_id = upload_response.json()["id"]

        delete_response = await client.delete(
            f"/v1/documents/{document_id}",
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

        assert delete_response.status_code == 204
        assert delete_response.content == b""

        get_response = await client.get(
            f"/v1/documents/{document_id}",
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_cannot_delete_another_users_document(
    db_session: AsyncSession,
) -> None:
    """Prevent document deletion across user boundaries."""
    async with await _build_client(db_session) as client:
        owner_token = await _register_user(
            client,
            email="api-delete-owner@example.com",
        )
        other_token = await _register_user(
            client,
            email="api-delete-other@example.com",
        )

        upload_response = await _upload_document(
            client,
            token=owner_token,
            file_name="protected.rtf",
            title="Protected Document",
            text="Document protected by ownership checks.",
        )

        assert upload_response.status_code == 201

        document_id = upload_response.json()["id"]

        delete_response = await client.delete(
            f"/v1/documents/{document_id}",
            headers={
                "Authorization": f"Bearer {other_token}",
            },
        )

        owner_get_response = await client.get(
            f"/v1/documents/{document_id}",
            headers={
                "Authorization": f"Bearer {owner_token}",
            },
        )

    assert delete_response.status_code == 404
    assert delete_response.json() == {
        "detail": "Document not found",
    }

    assert owner_get_response.status_code == 200


@pytest.mark.asyncio
async def test_bulk_delete_removes_all_requested_documents(
    db_session: AsyncSession,
) -> None:
    """Delete multiple owned documents atomically through HTTP."""
    async with await _build_client(db_session) as client:
        token = await _register_user(
            client,
            email="api-bulk-delete@example.com",
        )

        first_upload = await _upload_document(
            client,
            token=token,
            file_name="bulk-first.rtf",
            title="Bulk First",
            text="First bulk-delete document.",
        )
        second_upload = await _upload_document(
            client,
            token=token,
            file_name="bulk-second.rtf",
            title="Bulk Second",
            text="Second bulk-delete document.",
        )

        assert first_upload.status_code == 201
        assert second_upload.status_code == 201

        first_id = first_upload.json()["id"]
        second_id = second_upload.json()["id"]

        response = await client.post(
            "/v1/documents/bulk-delete",
            headers={
                "Authorization": f"Bearer {token}",
            },
            json={
                "document_ids": [
                    first_id,
                    second_id,
                ],
            },
        )

        assert response.status_code == 204
        assert response.content == b""

        list_response = await client.get(
            "/v1/documents",
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

    assert list_response.status_code == 200
    assert list_response.json() == []


@pytest.mark.asyncio
async def test_bulk_delete_is_all_or_nothing_when_one_document_is_missing(
    db_session: AsyncSession,
) -> None:
    """Keep valid documents when bulk validation finds a missing id."""
    async with await _build_client(db_session) as client:
        token = await _register_user(
            client,
            email="api-bulk-atomic@example.com",
        )

        upload_response = await _upload_document(
            client,
            token=token,
            file_name="bulk-preserved.rtf",
            title="Preserved Document",
            text="This document must survive failed bulk deletion.",
        )

        assert upload_response.status_code == 201

        document_id = upload_response.json()["id"]

        response = await client.post(
            "/v1/documents/bulk-delete",
            headers={
                "Authorization": f"Bearer {token}",
            },
            json={
                "document_ids": [
                    document_id,
                    999_999_999,
                ],
            },
        )

        preserved_response = await client.get(
            f"/v1/documents/{document_id}",
            headers={
                "Authorization": f"Bearer {token}",
            },
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "One or more documents were not found",
    }

    assert preserved_response.status_code == 200
    assert preserved_response.json()["id"] == document_id


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_file_extension(
    db_session: AsyncSession,
) -> None:
    """Reject unsupported document formats before registration."""
    async with await _build_client(db_session) as client:
        token = await _register_user(
            client,
            email="api-document-unsupported@example.com",
        )

        response = await client.post(
            "/v1/documents/upload",
            headers={
                "Authorization": f"Bearer {token}",
            },
            files={
                "file": (
                    "unsupported.txt",
                    b"plain text",
                    "text/plain",
                ),
            },
        )

    assert response.status_code == 415
    assert response.json() == {
        "detail": "Unsupported document extension: .txt",
    }


@pytest.mark.asyncio
async def test_document_routes_require_authentication(
    db_session: AsyncSession,
) -> None:
    """Protect document endpoints from anonymous access."""
    async with await _build_client(db_session) as client:
        response = await client.get(
            "/v1/documents",
        )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials",
    }


@pytest.mark.asyncio
async def test_upload_uses_file_stem_when_title_is_blank(
    db_session: AsyncSession,
) -> None:
    """Use the sanitized file stem when an explicit title is blank."""
    async with await _build_client(db_session) as client:
        token = await _register_user(
            client,
            email="api-document-default-title@example.com",
        )

        response = await client.post(
            "/v1/documents/upload",
            headers={
                "Authorization": f"Bearer {token}",
            },
            data={
                "title": "   ",
            },
            files={
                "file": (
                    "fallback-title.rtf",
                    _rtf_content("Document with a fallback title."),
                    "application/rtf",
                ),
            },
        )

    assert response.status_code == 201
    assert response.json()["title"] == "fallback-title"


@pytest.mark.asyncio
async def test_upload_rejects_empty_document(
    db_session: AsyncSession,
) -> None:
    """Reject an empty supported file before document registration."""
    async with await _build_client(db_session) as client:
        token = await _register_user(
            client,
            email="api-document-empty@example.com",
        )

        response = await client.post(
            "/v1/documents/upload",
            headers={
                "Authorization": f"Bearer {token}",
            },
            files={
                "file": (
                    "empty.rtf",
                    b"",
                    "application/rtf",
                ),
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded document cannot be empty",
    }


@pytest.mark.asyncio
async def test_upload_rejects_document_above_configured_size(
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject oversized uploads before loading the complete file into memory."""
    monkeypatch.setattr(
        routes_module,
        "get_settings",
        lambda: SimpleNamespace(
            document_upload_max_size_bytes=1024 * 1024,
        ),
    )

    async with await _build_client(db_session) as client:
        token = await _register_user(
            client,
            email="api-document-oversized@example.com",
        )

        response = await client.post(
            "/v1/documents/upload",
            headers={
                "Authorization": f"Bearer {token}",
            },
            files={
                "file": (
                    "oversized.rtf",
                    b"x" * (1024 * 1024 + 1),
                    "application/rtf",
                ),
            },
        )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Uploaded document exceeds the maximum allowed size of 1 MB",
    }
