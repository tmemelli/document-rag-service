"""
Tests for document ingestion orchestration and failure recovery.
"""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.services.ingestion_service import IngestionService


def _build_document(
    *,
    document_id: int = 1,
    user_id: int = 42,
    status: str = "pending",
) -> Document:
    """Create an in-memory document for ingestion tests."""
    document = Document(
        user_id=user_id,
        title="Test document",
        source_type="upload",
        file_name="document.pdf",
        mime_type="application/pdf",
        status=status,
    )
    document.id = document_id
    return document


def _replace_method(
    target: object,
    name: str,
    replacement: object,
) -> None:
    """Replace a method with a test double without static assignment conflicts."""
    setattr(target, name, replacement)


def _build_service() -> tuple[
    IngestionService,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    """Create an ingestion service with isolated dependency doubles."""
    session = MagicMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    document_repository = MagicMock()
    document_repository.get_by_id = AsyncMock()
    document_repository.update_status = AsyncMock()

    chunk_repository = MagicMock()
    chunk_repository.create = AsyncMock()

    status_history_repository = MagicMock()
    status_history_repository.create = AsyncMock()
    status_history_repository.get_latest_attempt_number = AsyncMock(return_value=1)

    error_repository = MagicMock()
    error_repository.create = AsyncMock()

    chunking_service = MagicMock()
    chunking_service.split_with_token_limit = MagicMock()

    embedding_service = MagicMock()
    embedding_service.embed_documents = AsyncMock()
    embedding_service.count_document_tokens = MagicMock()
    embedding_service.max_sequence_length = 8192

    text_extraction_service = MagicMock()
    text_extraction_service.extract_text = MagicMock()

    service = IngestionService(
        session=session,
        embedding_service=embedding_service,
        chunking_service=chunking_service,
    )

    service._documents = document_repository
    service._chunks = chunk_repository
    service._status_history = status_history_repository
    service._errors = error_repository
    service._text_extraction = text_extraction_service

    return (
        service,
        session,
        document_repository,
        chunk_repository,
        status_history_repository,
        error_repository,
        chunking_service,
        embedding_service,
    )


@pytest.mark.asyncio
async def test_ingest_file_content_extracts_text_from_uploaded_content() -> None:
    """Extract plain text before entering the text-ingestion pipeline."""
    (
        service,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _build_service()

    text_extraction = cast(
        MagicMock,
        service._text_extraction,
    )
    text_extraction.extract_text.return_value = "Extracted text"

    ingest_text = AsyncMock()
    _replace_method(
        service,
        "ingest_text",
        ingest_text,
    )

    await service.ingest_file_content(
        document_id=10,
        user_id=42,
        content=b"binary file content",
        file_name="document.pdf",
    )

    text_extraction.extract_text.assert_called_once_with(
        content=b"binary file content",
        file_name="document.pdf",
    )

    ingest_text.assert_awaited_once_with(
        document_id=10,
        user_id=42,
        text="Extracted text",
    )


@pytest.mark.asyncio
async def test_ingest_file_content_records_extraction_failure() -> None:
    """Record a structured ingestion failure when text extraction fails."""
    (
        service,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _build_service()

    extraction_error = ValueError("invalid PDF")

    text_extraction = cast(
        MagicMock,
        service._text_extraction,
    )
    text_extraction.extract_text.side_effect = extraction_error

    record_failure = AsyncMock()
    _replace_method(
        service,
        "_record_failure",
        record_failure,
    )

    with pytest.raises(
        ValueError,
        match="invalid PDF",
    ) as exc_info:
        await service.ingest_file_content(
            document_id=10,
            user_id=42,
            content=b"invalid content",
            file_name="document.pdf",
        )

    assert exc_info.value is extraction_error

    record_failure.assert_awaited_once_with(
        document_id=10,
        user_id=42,
        operation="extract_text",
        error_code="TEXT_EXTRACTION_FAILED",
        message="Failed to extract text from document.",
        exception=extraction_error,
    )


@pytest.mark.asyncio
async def test_ingest_file_content_does_not_continue_after_extraction_failure() -> None:
    """Stop before chunking when file extraction cannot be completed."""
    (
        service,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _build_service()

    text_extraction = cast(
        MagicMock,
        service._text_extraction,
    )
    text_extraction.extract_text.side_effect = ValueError("extraction failed")

    record_failure = AsyncMock()
    _replace_method(
        service,
        "_record_failure",
        record_failure,
    )

    ingest_text = AsyncMock()
    _replace_method(
        service,
        "ingest_text",
        ingest_text,
    )

    with pytest.raises(ValueError):
        await service.ingest_file_content(
            document_id=1,
            user_id=42,
            content=b"content",
            file_name="document.pdf",
        )

    ingest_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_text_uses_embedding_tokenizer_and_sequence_limit() -> None:
    """Chunk text using the active embedding provider's token contract."""
    (
        service,
        _,
        _,
        _,
        _,
        _,
        chunking,
        embedding,
    ) = _build_service()

    chunking.split_with_token_limit.return_value = [
        ("chunk one", 2),
    ]

    persist_chunks = AsyncMock()
    _replace_method(
        service,
        "persist_chunks",
        persist_chunks,
    )

    await service.ingest_text(
        document_id=1,
        user_id=42,
        text="document text",
    )

    chunking.split_with_token_limit.assert_called_once_with(
        "document text",
        count_tokens=embedding.count_document_tokens,
        max_tokens=8192,
    )


@pytest.mark.asyncio
async def test_ingest_text_persists_generated_chunks() -> None:
    """Forward token-safe chunks to the persistence stage."""
    (
        service,
        _,
        _,
        _,
        _,
        _,
        chunking,
        _,
    ) = _build_service()

    chunks = [
        ("chunk one", 2),
        ("chunk two", 3),
    ]

    chunking.split_with_token_limit.return_value = chunks

    persist_chunks = AsyncMock()
    _replace_method(
        service,
        "persist_chunks",
        persist_chunks,
    )

    await service.ingest_text(
        document_id=7,
        user_id=42,
        text="document text",
    )

    persist_chunks.assert_awaited_once_with(
        document_id=7,
        user_id=42,
        chunks=chunks,
    )


@pytest.mark.asyncio
async def test_ingest_text_rejects_empty_chunk_result() -> None:
    """Reject ingestion when no valid chunk can be produced."""
    (
        service,
        _,
        _,
        _,
        _,
        _,
        chunking,
        _,
    ) = _build_service()

    chunking.split_with_token_limit.return_value = []

    record_failure = AsyncMock()
    _replace_method(
        service,
        "_record_failure",
        record_failure,
    )

    with pytest.raises(
        ValueError,
        match="Document text cannot be empty",
    ) as exc_info:
        await service.ingest_text(
            document_id=1,
            user_id=42,
            text="",
        )

    record_failure.assert_awaited_once_with(
        document_id=1,
        user_id=42,
        operation="chunk_document",
        error_code="CHUNKING_FAILED",
        message="Failed to split document into valid chunks.",
        exception=exc_info.value,
    )


@pytest.mark.asyncio
async def test_ingest_text_records_chunking_failure() -> None:
    """Record chunking and tokenization errors before propagating them."""
    (
        service,
        _,
        _,
        _,
        _,
        _,
        chunking,
        _,
    ) = _build_service()

    chunking_error = RuntimeError("tokenizer failed")

    chunking.split_with_token_limit.side_effect = chunking_error

    record_failure = AsyncMock()
    _replace_method(
        service,
        "_record_failure",
        record_failure,
    )

    with pytest.raises(
        RuntimeError,
        match="tokenizer failed",
    ) as exc_info:
        await service.ingest_text(
            document_id=1,
            user_id=42,
            text="document text",
        )

    assert exc_info.value is chunking_error

    record_failure.assert_awaited_once_with(
        document_id=1,
        user_id=42,
        operation="chunk_document",
        error_code="CHUNKING_FAILED",
        message="Failed to split document into valid chunks.",
        exception=chunking_error,
    )


@pytest.mark.asyncio
async def test_persist_chunks_rejects_missing_document() -> None:
    """Reject persistence when the target document does not exist."""
    (
        service,
        session,
        documents,
        chunks,
        history,
        _,
        _,
        embedding,
    ) = _build_service()

    documents.get_by_id.return_value = None

    with pytest.raises(
        ValueError,
        match="Document 99 not found",
    ):
        await service.persist_chunks(
            document_id=99,
            user_id=42,
            chunks=[
                ("chunk", 1),
            ],
        )

    documents.update_status.assert_not_awaited()
    history.create.assert_not_awaited()
    embedding.embed_documents.assert_not_awaited()
    chunks.create.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_chunks_transitions_document_to_processing() -> None:
    """Persist the durable processing transition before embedding work begins."""
    (
        service,
        _,
        documents,
        _,
        history,
        _,
        _,
        embedding,
    ) = _build_service()

    document = _build_document(
        document_id=1,
        status="pending",
    )

    documents.get_by_id.return_value = document
    embedding.embed_documents.return_value = [
        [0.1, 0.2],
    ]

    await service.persist_chunks(
        document_id=1,
        user_id=42,
        chunks=[
            ("chunk", 1),
        ],
    )

    first_status_call = documents.update_status.await_args_list[0]

    assert first_status_call.kwargs == {
        "document": document,
        "status": "processing",
    }

    history.create.assert_any_await(
        document_id=1,
        from_status="pending",
        to_status="processing",
        component="ingestion_service",
        operation="persist_chunks",
        attempt_number=1,
    )


@pytest.mark.asyncio
async def test_persist_chunks_commits_processing_state_before_embedding() -> None:
    """Commit the processing lifecycle state before external embedding work."""
    (
        service,
        session,
        documents,
        _,
        _,
        _,
        _,
        embedding,
    ) = _build_service()

    document = _build_document(
        status="pending",
    )

    documents.get_by_id.return_value = document
    embedding.embed_documents.return_value = [
        [0.1],
    ]

    events: list[str] = []

    async def commit() -> None:
        events.append("commit")

    async def embed_documents(
        texts: list[str],
    ) -> list[list[float]]:
        events.append("embed")
        return [[0.1]]

    session.commit.side_effect = commit
    embedding.embed_documents.side_effect = embed_documents

    await service.persist_chunks(
        document_id=1,
        user_id=42,
        chunks=[
            ("chunk", 1),
        ],
    )

    assert events[0:2] == [
        "commit",
        "embed",
    ]


@pytest.mark.asyncio
async def test_persist_chunks_embeds_chunk_contents_in_order() -> None:
    """Generate embeddings from chunk contents while preserving chunk order."""
    (
        service,
        _,
        documents,
        _,
        _,
        _,
        _,
        embedding,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document()

    embedding.embed_documents.return_value = [
        [0.1],
        [0.2],
        [0.3],
    ]

    await service.persist_chunks(
        document_id=1,
        user_id=42,
        chunks=[
            ("first chunk", 2),
            ("second chunk", 3),
            ("third chunk", 4),
        ],
    )

    embedding.embed_documents.assert_awaited_once_with(
        [
            "first chunk",
            "second chunk",
            "third chunk",
        ]
    )


@pytest.mark.asyncio
async def test_persist_chunks_creates_chunk_records_with_indexes() -> None:
    """Persist ordered chunk records with token counts and embeddings."""
    (
        service,
        _,
        documents,
        chunk_repository,
        _,
        _,
        _,
        embedding,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document()

    embedding.embed_documents.return_value = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]

    await service.persist_chunks(
        document_id=1,
        user_id=42,
        chunks=[
            ("first chunk", 5),
            ("second chunk", 7),
        ],
    )

    assert chunk_repository.create.await_count == 2

    chunk_repository.create.assert_any_await(
        document_id=1,
        chunk_index=0,
        content="first chunk",
        embedding=[0.1, 0.2],
        token_count=5,
    )

    chunk_repository.create.assert_any_await(
        document_id=1,
        chunk_index=1,
        content="second chunk",
        embedding=[0.3, 0.4],
        token_count=7,
    )


@pytest.mark.asyncio
async def test_persist_chunks_transitions_document_to_ready() -> None:
    """Move the document to ready after all chunks are persisted."""
    (
        service,
        _,
        documents,
        _,
        history,
        _,
        _,
        embedding,
    ) = _build_service()

    document = _build_document(
        status="pending",
    )

    documents.get_by_id.return_value = document
    embedding.embed_documents.return_value = [
        [0.1],
    ]

    await service.persist_chunks(
        document_id=1,
        user_id=42,
        chunks=[
            ("chunk", 1),
        ],
    )

    assert documents.update_status.await_count == 2

    second_status_call = documents.update_status.await_args_list[1]

    assert second_status_call.kwargs == {
        "document": document,
        "status": "ready",
    }

    history.create.assert_any_await(
        document_id=1,
        from_status="processing",
        to_status="ready",
        component="ingestion_service",
        operation="persist_chunks",
        attempt_number=1,
    )


@pytest.mark.asyncio
async def test_persist_chunks_commits_processing_and_ready_transactions() -> None:
    """Commit once for processing and once for the completed ready state."""
    (
        service,
        session,
        documents,
        _,
        _,
        _,
        _,
        embedding,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document()

    embedding.embed_documents.return_value = [
        [0.1],
    ]

    await service.persist_chunks(
        document_id=1,
        user_id=42,
        chunks=[
            ("chunk", 1),
        ],
    )

    assert session.commit.await_count == 2


@pytest.mark.asyncio
async def test_persist_chunks_records_embedding_failure() -> None:
    """Record ingestion failure when embedding generation fails."""
    (
        service,
        _,
        documents,
        _,
        _,
        _,
        _,
        embedding,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document()

    embedding_error = RuntimeError("embedding provider unavailable")

    embedding.embed_documents.side_effect = embedding_error

    record_failure = AsyncMock()
    _replace_method(
        service,
        "_record_failure",
        record_failure,
    )

    with pytest.raises(
        RuntimeError,
        match="embedding provider unavailable",
    ) as exc_info:
        await service.persist_chunks(
            document_id=1,
            user_id=42,
            chunks=[
                ("chunk", 1),
            ],
        )

    assert exc_info.value is embedding_error

    record_failure.assert_awaited_once_with(
        document_id=1,
        user_id=42,
        operation="persist_chunks",
        error_code="CHUNK_PERSISTENCE_FAILED",
        message="Failed to persist document chunks.",
        exception=embedding_error,
    )


@pytest.mark.asyncio
async def test_persist_chunks_records_chunk_repository_failure() -> None:
    """Record ingestion failure when chunk persistence fails."""
    (
        service,
        _,
        documents,
        chunks,
        _,
        _,
        _,
        embedding,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document()

    embedding.embed_documents.return_value = [
        [0.1],
    ]

    persistence_error = RuntimeError("chunk persistence failed")

    chunks.create.side_effect = persistence_error

    record_failure = AsyncMock()
    _replace_method(
        service,
        "_record_failure",
        record_failure,
    )

    with pytest.raises(
        RuntimeError,
        match="chunk persistence failed",
    ):
        await service.persist_chunks(
            document_id=1,
            user_id=42,
            chunks=[
                ("chunk", 1),
            ],
        )

    record_failure.assert_awaited_once_with(
        document_id=1,
        user_id=42,
        operation="persist_chunks",
        error_code="CHUNK_PERSISTENCE_FAILED",
        message="Failed to persist document chunks.",
        exception=persistence_error,
    )


@pytest.mark.asyncio
async def test_persist_chunks_records_embedding_count_mismatch() -> None:
    """Treat mismatched chunk and embedding counts as persistence failure."""
    (
        service,
        _,
        documents,
        _,
        _,
        _,
        _,
        embedding,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document()

    embedding.embed_documents.return_value = [
        [0.1],
    ]

    record_failure = AsyncMock()
    _replace_method(
        service,
        "_record_failure",
        record_failure,
    )

    with pytest.raises(ValueError):
        await service.persist_chunks(
            document_id=1,
            user_id=42,
            chunks=[
                ("first", 1),
                ("second", 1),
            ],
        )

    record_failure.assert_awaited_once()

    assert record_failure.await_args is not None

    failure_exception = record_failure.await_args.kwargs["exception"]

    assert isinstance(
        failure_exception,
        ValueError,
    )


@pytest.mark.asyncio
async def test_record_failure_rolls_back_active_transaction_first() -> None:
    """Rollback partial ingestion work before reloading durable document state."""
    (
        service,
        session,
        documents,
        _,
        _,
        _,
        _,
        _,
    ) = _build_service()

    document = _build_document(
        status="processing",
    )

    events: list[str] = []

    async def rollback() -> None:
        events.append("rollback")

    async def get_by_id(
        *,
        document_id: int,
        user_id: int,
    ) -> Document:
        events.append("reload")
        return document

    session.rollback.side_effect = rollback
    documents.get_by_id.side_effect = get_by_id

    await service._record_failure(
        document_id=1,
        user_id=42,
        operation="persist_chunks",
        error_code="TEST_ERROR",
        message="Failure message",
        exception=RuntimeError("boom"),
    )

    assert events[0:2] == [
        "rollback",
        "reload",
    ]


@pytest.mark.asyncio
async def test_record_failure_returns_when_document_no_longer_exists() -> None:
    """Stop after rollback when the failed document cannot be reloaded."""
    (
        service,
        session,
        documents,
        _,
        history,
        errors,
        _,
        _,
    ) = _build_service()

    documents.get_by_id.return_value = None

    await service._record_failure(
        document_id=1,
        user_id=42,
        operation="persist_chunks",
        error_code="TEST_ERROR",
        message="Failure message",
        exception=RuntimeError("boom"),
    )

    session.rollback.assert_awaited_once_with()

    documents.update_status.assert_not_awaited()
    history.create.assert_not_awaited()
    errors.create.assert_not_awaited()
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_failure_transitions_document_to_failed() -> None:
    """Transition the durable document state to failed after rollback."""
    (
        service,
        _,
        documents,
        _,
        history,
        _,
        _,
        _,
    ) = _build_service()

    document = _build_document(
        status="processing",
    )

    documents.get_by_id.return_value = document

    await service._record_failure(
        document_id=1,
        user_id=42,
        operation="persist_chunks",
        error_code="TEST_ERROR",
        message="Failure message",
        exception=RuntimeError("boom"),
    )

    documents.update_status.assert_awaited_once_with(
        document=document,
        status="failed",
    )

    history.create.assert_awaited_once_with(
        document_id=1,
        from_status="processing",
        to_status="failed",
        component="ingestion_service",
        operation="persist_chunks",
        attempt_number=1,
    )


@pytest.mark.asyncio
async def test_record_failure_persists_structured_error_event() -> None:
    """Persist safe structured observability data for the ingestion failure."""
    (
        service,
        _,
        documents,
        _,
        _,
        errors,
        _,
        _,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document(
        status="processing",
    )

    exception = ValueError("sensitive internal detail")

    await service._record_failure(
        document_id=1,
        user_id=42,
        operation="chunk_document",
        error_code="CHUNKING_FAILED",
        message="Failed to split document into valid chunks.",
        exception=exception,
    )

    errors.create.assert_awaited_once_with(
        document_id=1,
        component="ingestion_service",
        operation="chunk_document",
        error_type="ValueError",
        error_code="CHUNKING_FAILED",
        message=(
            "Failed to split document into valid chunks. "
            "ValueError: sensitive internal detail"
        ),
        details={
            "summary": "Failed to split document into valid chunks.",
            "exception_type": "ValueError",
            "exception_message": "sensitive internal detail",
        },
        retryable=False,
        attempt_number=1,
    )


@pytest.mark.asyncio
async def test_record_failure_preserves_retry_attempt_number() -> None:
    """Associate retry failures with the active ingestion attempt."""
    (
        service,
        _,
        documents,
        _,
        history,
        errors,
        _,
        _,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document(status="processing")
    history.get_latest_attempt_number.return_value = 2

    await service._record_failure(
        document_id=1,
        user_id=42,
        operation="persist_chunks",
        error_code="CHUNK_PERSISTENCE_FAILED",
        message="Failed to persist document chunks.",
        exception=RuntimeError("temporary failure"),
    )

    history.create.assert_awaited_once_with(
        document_id=1,
        from_status="processing",
        to_status="failed",
        component="ingestion_service",
        operation="persist_chunks",
        attempt_number=2,
    )
    errors.create.assert_awaited_once_with(
        document_id=1,
        component="ingestion_service",
        operation="persist_chunks",
        error_type="RuntimeError",
        error_code="CHUNK_PERSISTENCE_FAILED",
        message=(
            "Failed to persist document chunks. "
            "RuntimeError: temporary failure"
        ),
        details={
            "summary": "Failed to persist document chunks.",
            "exception_type": "RuntimeError",
            "exception_message": "temporary failure",
        },
        retryable=False,
        attempt_number=2,
    )


@pytest.mark.asyncio
async def test_record_failure_redacts_sensitive_exception_diagnostics() -> None:
    """Persist technical diagnostics without retaining common credential forms."""
    (
        service,
        _,
        documents,
        _,
        _,
        errors,
        _,
        _,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document(
        status="processing",
    )

    exception = RuntimeError("database password=super-secret")

    await service._record_failure(
        document_id=1,
        user_id=42,
        operation="persist_chunks",
        error_code="CHUNK_PERSISTENCE_FAILED",
        message="Failed to persist document chunks.",
        exception=exception,
    )

    assert errors.create.await_args is not None

    stored_message = errors.create.await_args.kwargs["message"]
    details = errors.create.await_args.kwargs["details"]

    assert stored_message == (
        "Failed to persist document chunks. "
        "RuntimeError: database password=[REDACTED]"
    )
    assert details == {
        "summary": "Failed to persist document chunks.",
        "exception_type": "RuntimeError",
        "exception_message": "database password=[REDACTED]",
    }
    assert "super-secret" not in stored_message
    assert "super-secret" not in str(details)


@pytest.mark.asyncio
async def test_record_failure_extracts_cloudflare_provider_diagnostics() -> None:
    """Persist Cloudflare HTTP and provider codes without requiring log lookup."""
    (
        service,
        _,
        documents,
        _,
        _,
        errors,
        _,
        _,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document(
        status="processing",
    )

    exception_message = (
        "Cloudflare embedding request failed: status=400 "
        'body={"errors":[{"message":"AiError: AiError: input too big '
        '(df31816f-4106-46f6-b7d4-c347c6bdf91d)","code":3030}],'
        '"success":false,"result":{},"messages":[]}'
    )
    exception = RuntimeError(exception_message)

    await service._record_failure(
        document_id=1,
        user_id=42,
        operation="persist_chunks",
        error_code="CHUNK_PERSISTENCE_FAILED",
        message="Failed to persist document chunks.",
        exception=exception,
    )

    assert errors.create.await_args is not None

    stored_message = errors.create.await_args.kwargs["message"]
    details = errors.create.await_args.kwargs["details"]

    assert stored_message == (
        "Failed to persist document chunks. "
        f"RuntimeError: {exception_message}"
    )
    assert details == {
        "summary": "Failed to persist document chunks.",
        "exception_type": "RuntimeError",
        "exception_message": exception_message,
        "provider": "cloudflare",
        "http_status": 400,
        "provider_code": 3030,
        "provider_message": (
            "AiError: AiError: input too big "
            "(df31816f-4106-46f6-b7d4-c347c6bdf91d)"
        ),
    }


@pytest.mark.asyncio
async def test_record_failure_commits_failure_state_and_error_event() -> None:
    """Commit the failed lifecycle transition and observability record together."""
    (
        service,
        session,
        documents,
        _,
        _,
        _,
        _,
        _,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document(
        status="processing",
    )

    await service._record_failure(
        document_id=1,
        user_id=42,
        operation="persist_chunks",
        error_code="TEST_ERROR",
        message="Failure message",
        exception=RuntimeError("boom"),
    )

    session.commit.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_record_failure_preserves_reloaded_document_status_as_from_status() -> (
    None
):
    """Base failure history on the durable status reloaded after rollback."""
    (
        service,
        _,
        documents,
        _,
        history,
        _,
        _,
        _,
    ) = _build_service()

    documents.get_by_id.return_value = _build_document(
        status="pending",
    )

    await service._record_failure(
        document_id=1,
        user_id=42,
        operation="extract_text",
        error_code="TEXT_EXTRACTION_FAILED",
        message="Failed to extract text from document.",
        exception=ValueError("bad file"),
    )

    history.create.assert_awaited_once_with(
        document_id=1,
        from_status="pending",
        to_status="failed",
        component="ingestion_service",
        operation="extract_text",
        attempt_number=1,
    )
