"""
Document ingestion orchestration, lifecycle tracking, and chunk persistence.
"""

import asyncio
import json
import re
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.db.session import AsyncSessionFactory
from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.document_status_history_repository import (
    DocumentStatusHistoryRepository,
)
from document_rag_service.repositories.error_event_repository import (
    ErrorEventRepository,
)
from document_rag_service.services.chunking_service import (
    ChunkingService,
)
from document_rag_service.services.embedding_service import (
    EmbeddingService,
)
from document_rag_service.services.text_extraction_service import (
    TextExtractionService,
)

_MAX_DIAGNOSTIC_MESSAGE_LENGTH = 4_000

_BEARER_TOKEN_PATTERN = re.compile(
    r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]+"
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"""(?ix)
    (
        \b(?:
            password|passwd|pwd|secret|api[_-]?key|api[_-]?token|
            access[_-]?token|refresh[_-]?token
        )\b
        ["']?\s*[:=]\s*
    )
    (?:
        "(?:\\.|[^"])*"
        |
        '(?:\\.|[^'])*'
        |
        [^,\s;}\]&]+
    )
    """
)
_URL_CREDENTIAL_PATTERN = re.compile(
    r"(?i)([a-z][a-z0-9+.-]*://[^:/\s]+:)[^@\s]+(@)"
)
_CLOUDFLARE_STATUS_PATTERN = re.compile(r"\bstatus=(\d{3})\b")


class ProcessingLeaseLostError(RuntimeError):
    """Signal that an ingestion attempt no longer owns its processing lease."""


class IngestionService:
    """
    Orchestrate document extraction, chunking, embedding, and persistence.

    The service coordinates the ingestion pipeline from raw uploaded content to
    persisted vectorized chunks. It also owns document processing transitions
    and records structured failure information when extraction, chunking,
    embedding, or persistence cannot be completed.

    Database transaction boundaries are intentionally placed around durable
    lifecycle transitions so partial chunk persistence is rolled back while
    failure state and observability records remain committed.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedding_service: EmbeddingService,
        chunking_service: ChunkingService,
        *,
        session_factory: AsyncSessionFactory | None = None,
        processing_lease_seconds: int = 120,
        processing_heartbeat_seconds: int = 30,
    ) -> None:
        """
        Initialize the document ingestion service.

        Args:
            session: Database session used for transactional persistence.
            embedding_service: Service used for token counting and embedding
                generation.
            chunking_service: Service used to split extracted text into bounded
                chunks.
            session_factory: Optional application session factory used by the
                processing-lease heartbeat. Production dependency composition
                supplies this factory; isolated tests may omit it.
            processing_lease_seconds: Duration of each processing lease.
            processing_heartbeat_seconds: Interval used to renew active leases.

        Raises:
            ValueError: If lease timing configuration is invalid.
        """

        if processing_lease_seconds <= 0:
            raise ValueError("Processing lease duration must be greater than zero")

        if processing_heartbeat_seconds <= 0:
            raise ValueError("Processing heartbeat interval must be greater than zero")

        if processing_heartbeat_seconds >= processing_lease_seconds:
            raise ValueError(
                "Processing heartbeat interval must be smaller than lease duration"
            )

        self._session = session
        self._documents = DocumentRepository(session)
        self._chunks = DocumentChunkRepository(session)
        self._status_history = DocumentStatusHistoryRepository(session)
        self._errors = ErrorEventRepository(session)
        self._chunking = chunking_service
        self._text_extraction = TextExtractionService()
        self._embedding = embedding_service
        self._session_factory = session_factory
        self._processing_lease_seconds = processing_lease_seconds
        self._processing_heartbeat_seconds = processing_heartbeat_seconds

    async def ingest_file_content(
        self,
        *,
        document_id: int,
        user_id: int,
        content: bytes,
        file_name: str,
    ) -> None:
        """
        Extract text from an uploaded file and execute the ingestion pipeline.

        File extraction is treated as part of the document lifecycle. If the
        uploaded content cannot be parsed, the document is transitioned to
        ``failed`` and a structured error event is persisted before the
        original exception is propagated.

        Args:
            document_id: Identifier of the document being ingested.
            user_id: Identifier of the user who owns the document.
            content: Raw uploaded file bytes.
            file_name: Original file name used to determine the extraction
                format.

        Raises:
            Exception: Propagates extraction or downstream ingestion failures
                after the corresponding failure state has been recorded.
        """

        try:
            text = self._text_extraction.extract_text(
                content=content,
                file_name=file_name,
            )
        except Exception as exc:
            await self._record_failure(
                document_id=document_id,
                user_id=user_id,
                operation="extract_text",
                error_code="TEXT_EXTRACTION_FAILED",
                message="Failed to extract text from document.",
                exception=exc,
            )
            raise

        await self.ingest_text(
            document_id=document_id,
            user_id=user_id,
            text=text,
        )

    async def ingest_text(
        self,
        *,
        document_id: int,
        user_id: int,
        text: str,
    ) -> None:
        """
        Split extracted text into token-safe chunks and persist them.

        Chunking uses the tokenizer associated with the active embedding
        provider so every produced chunk remains within the model's maximum
        sequence length. Failures during chunking or token counting are
        persisted as document failures before being propagated.

        Args:
            document_id: Identifier of the document being ingested.
            user_id: Identifier of the user who owns the document.
            text: Extracted plain-text document content.

        Raises:
            ValueError: If no valid chunks can be produced from the document.
            Exception: Propagates chunking, tokenization, embedding, or
                persistence failures after their failure state is recorded.
        """

        try:
            chunks_with_token_count = self._chunking.split_with_token_limit(
                text,
                count_tokens=self._embedding.count_document_tokens,
                max_tokens=self._embedding.max_sequence_length,
            )

            if not chunks_with_token_count:
                raise ValueError("Document text cannot be empty")

        except Exception as exc:
            await self._record_failure(
                document_id=document_id,
                user_id=user_id,
                operation="chunk_document",
                error_code="CHUNKING_FAILED",
                message="Failed to split document into valid chunks.",
                exception=exc,
            )
            raise

        await self.persist_chunks(
            document_id=document_id,
            user_id=user_id,
            chunks=chunks_with_token_count,
        )

    async def persist_chunks(
        self,
        *,
        document_id: int,
        user_id: int,
        chunks: list[tuple[str, int]],
    ) -> None:
        """
        Generate embeddings and persist document chunks transactionally.

        A pending document is atomically claimed with a unique processing token
        and a renewable lease. The lease heartbeat proves that the ingestion is
        still alive while long-running remote embedding work is in progress.

        Completion and failure transitions are fenced by the processing token.
        If a newer retry recovers an expired lease, the older worker can no
        longer commit chunks or overwrite the newer lifecycle state.

        Args:
            document_id: Identifier of the document receiving the chunks.
            user_id: Identifier of the user who owns the document.
            chunks: Ordered ``(content, token_count)`` pairs to embed and
                persist.

        Raises:
            ValueError: If the target document does not exist or is no longer
                pending.
            ProcessingLeaseLostError: If a newer attempt recovers the lease
                before this worker commits.
            Exception: Propagates embedding or persistence failures after the
                database transaction is rolled back and failure state recorded.
        """

        existing_document = await self._documents.get_by_id(
            document_id=document_id,
            user_id=user_id,
        )

        if existing_document is None:
            raise ValueError(f"Document {document_id} not found")

        attempt_number = (
            await self._status_history.get_latest_attempt_number(document_id)
            or 1
        )
        processing_token = uuid4().hex

        document = await self._documents.claim_pending_for_processing(
            document_id=document_id,
            user_id=user_id,
            processing_token=processing_token,
            lease_expires_at=self._next_processing_lease_expiration(),
        )

        if document is None:
            raise ValueError(
                f"Document {document_id} is not available for processing"
            )

        await self._status_history.create(
            document_id=document.id,
            from_status="pending",
            to_status="processing",
            component="ingestion_service",
            operation="persist_chunks",
            attempt_number=attempt_number,
            details={
                "processing_token": processing_token,
            },
        )

        await self._session.commit()

        heartbeat_task = self._start_processing_heartbeat(
            document_id=document_id,
            user_id=user_id,
            processing_token=processing_token,
        )

        try:
            contents = [content for content, _ in chunks]

            embeddings = await self._embedding.embed_documents(contents)

            for chunk_index, ((content, token_count), embedding) in enumerate(
                zip(
                    chunks,
                    embeddings,
                    strict=True,
                )
            ):
                await self._chunks.create(
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=content,
                    embedding=embedding,
                    token_count=token_count,
                )

            completed_document = await self._documents.complete_processing(
                document_id=document_id,
                user_id=user_id,
                processing_token=processing_token,
            )

            if completed_document is None:
                raise ProcessingLeaseLostError(
                    f"Processing lease lost for document {document_id}"
                )

            await self._status_history.create(
                document_id=document.id,
                from_status="processing",
                to_status="ready",
                component="ingestion_service",
                operation="persist_chunks",
                attempt_number=attempt_number,
            )

            await self._session.commit()

        except asyncio.CancelledError as exc:
            await self._record_failure(
                document_id=document_id,
                user_id=user_id,
                operation="persist_chunks",
                error_code="INGESTION_INTERRUPTED",
                message="Document ingestion was interrupted before completion.",
                exception=exc,
                processing_token=processing_token,
                retryable=True,
            )
            raise

        except ProcessingLeaseLostError:
            await self._session.rollback()
            raise

        except Exception as exc:
            await self._record_failure(
                document_id=document_id,
                user_id=user_id,
                operation="persist_chunks",
                error_code="CHUNK_PERSISTENCE_FAILED",
                message="Failed to persist document chunks.",
                exception=exc,
                processing_token=processing_token,
            )
            raise

        finally:
            await self._stop_processing_heartbeat(heartbeat_task)

    def _next_processing_lease_expiration(self) -> datetime:
        """Return the UTC expiration timestamp for a newly renewed lease."""

        return datetime.now(UTC) + timedelta(
            seconds=self._processing_lease_seconds
        )

    def _start_processing_heartbeat(
        self,
        *,
        document_id: int,
        user_id: int,
        processing_token: str,
    ) -> asyncio.Task[None] | None:
        """
        Start a background lease heartbeat when a session factory is available.

        Unit and integration service tests that construct the service directly
        may omit a session factory. Production dependency composition supplies
        one so long-running ingestions continuously renew their lease.
        """

        if self._session_factory is None:
            return None

        return asyncio.create_task(
            self._run_processing_heartbeat(
                document_id=document_id,
                user_id=user_id,
                processing_token=processing_token,
            )
        )

    async def _run_processing_heartbeat(
        self,
        *,
        document_id: int,
        user_id: int,
        processing_token: str,
    ) -> None:
        """
        Renew an active processing lease until ownership changes or task stops.

        Heartbeat failures intentionally stop renewal instead of crashing the
        foreground ingestion. If the lease later expires and another request
        recovers it, the processing token prevents this worker from committing
        stale results.
        """

        if self._session_factory is None:
            return

        while True:
            await asyncio.sleep(self._processing_heartbeat_seconds)

            try:
                async with self._session_factory() as heartbeat_session:
                    repository = DocumentRepository(heartbeat_session)

                    renewed = await repository.renew_processing_lease(
                        document_id=document_id,
                        user_id=user_id,
                        processing_token=processing_token,
                        lease_expires_at=(
                            self._next_processing_lease_expiration()
                        ),
                    )

                    if not renewed:
                        await heartbeat_session.rollback()
                        return

                    await heartbeat_session.commit()

            except Exception:
                return

    @staticmethod
    async def _stop_processing_heartbeat(
        heartbeat_task: asyncio.Task[None] | None,
    ) -> None:
        """Cancel and await the optional processing heartbeat task."""

        if heartbeat_task is None:
            return

        heartbeat_task.cancel()

        with suppress(asyncio.CancelledError):
            await heartbeat_task

    async def _record_failure(
        self,
        *,
        document_id: int,
        user_id: int,
        operation: str,
        error_code: str,
        message: str,
        exception: BaseException,
        processing_token: str | None = None,
        retryable: bool = False,
    ) -> None:
        """
        Roll back active work and persist a document ingestion failure.

        The document is reloaded after rollback so the failure transition is
        based on the last durable database state rather than on potentially
        stale in-memory ORM state.

        Args:
            document_id: Identifier of the document whose ingestion failed.
            user_id: Identifier of the document owner.
            operation: Ingestion operation in which the failure occurred.
            error_code: Structured application error classification.
            message: Safe operation-level summary of the failure.
            exception: Original exception used to record the concrete error
                type and sanitized technical diagnostics.
            processing_token: Optional lease token used to fence a processing
                failure transition.
            retryable: Whether the failed operation may be safely retried.
        """

        await self._session.rollback()

        if processing_token is not None:
            document = await self._documents.fail_processing_if_owned(
                document_id=document_id,
                user_id=user_id,
                processing_token=processing_token,
            )
            from_status = "processing"
        else:
            document = await self._documents.get_by_id(
                document_id=document_id,
                user_id=user_id,
            )

            if document is not None:
                from_status = document.status

                await self._documents.update_status(
                    document=document,
                    status="failed",
                )
            else:
                from_status = None

        if document is None:
            return

        attempt_number = (
            await self._status_history.get_latest_attempt_number(document.id)
            or 1
        )

        await self._status_history.create(
            document_id=document.id,
            from_status=from_status,
            to_status="failed",
            component="ingestion_service",
            operation=operation,
            attempt_number=attempt_number,
        )

        diagnostic_details = self._build_error_details(
            summary=message,
            exception=exception,
        )
        diagnostic_message = (
            f"{message} "
            f"{diagnostic_details['exception_type']}: "
            f"{diagnostic_details['exception_message']}"
        )

        await self._errors.create(
            document_id=document.id,
            component="ingestion_service",
            operation=operation,
            error_type=type(exception).__name__,
            error_code=error_code,
            message=diagnostic_message,
            details=diagnostic_details,
            retryable=retryable,
            attempt_number=attempt_number,
        )

        await self._session.commit()

    @classmethod
    def _build_error_details(
        cls,
        *,
        summary: str,
        exception: BaseException,
    ) -> dict[str, Any]:
        """
        Build safe structured diagnostics for a persisted error event.

        The original exception message is retained only after redaction and
        length bounding. Known Cloudflare embedding errors are additionally
        decomposed into provider-specific fields so operational failures can be
        inspected directly from ``error_events`` without searching application
        logs.

        Args:
            summary: Safe operation-level description of the failure.
            exception: Exception raised by the failed operation.

        Returns:
            JSON-serializable diagnostic metadata safe for database persistence.
        """

        raw_exception_message = str(exception)
        exception_message = cls._sanitize_diagnostic_text(raw_exception_message)

        if not exception_message:
            exception_message = "Task cancellation requested."

        details: dict[str, Any] = {
            "summary": summary,
            "exception_type": type(exception).__name__,
            "exception_message": exception_message,
        }

        details.update(
            cls._extract_cloudflare_error_details(raw_exception_message)
        )

        return details

    @classmethod
    def _sanitize_diagnostic_text(
        cls,
        value: str,
    ) -> str:
        """
        Redact common credential forms and bound diagnostic message length.

        Args:
            value: Raw diagnostic text produced by an exception or provider.

        Returns:
            Redacted diagnostic text limited to a safe persistence size.
        """

        sanitized = value.strip()

        sanitized = _BEARER_TOKEN_PATTERN.sub(
            r"\1[REDACTED]",
            sanitized,
        )
        sanitized = _SECRET_ASSIGNMENT_PATTERN.sub(
            r"\1[REDACTED]",
            sanitized,
        )
        sanitized = _URL_CREDENTIAL_PATTERN.sub(
            r"\1[REDACTED]\2",
            sanitized,
        )

        if len(sanitized) > _MAX_DIAGNOSTIC_MESSAGE_LENGTH:
            return (
                sanitized[: _MAX_DIAGNOSTIC_MESSAGE_LENGTH - 3]
                + "..."
            )

        return sanitized

    @classmethod
    def _extract_cloudflare_error_details(
        cls,
        raw_exception_message: str,
    ) -> dict[str, Any]:
        """
        Extract structured metadata from Cloudflare embedding error messages.

        Cloudflare HTTP failures currently reach the ingestion layer as runtime
        errors containing ``status=<code> body=<json>``. Only selected provider
        fields are retained; the complete raw response is not copied into
        structured metadata.

        Args:
            raw_exception_message: Unmodified exception message.

        Returns:
            Structured Cloudflare metadata when recognized, otherwise an empty
            dictionary.
        """

        if "Cloudflare embedding" not in raw_exception_message:
            return {}

        details: dict[str, Any] = {
            "provider": "cloudflare",
        }

        status_match = _CLOUDFLARE_STATUS_PATTERN.search(
            raw_exception_message
        )

        if status_match is not None:
            details["http_status"] = int(status_match.group(1))

        body_marker = " body="

        if body_marker not in raw_exception_message:
            return details

        raw_body = raw_exception_message.split(body_marker, maxsplit=1)[1]

        try:
            payload = json.loads(raw_body)
        except (json.JSONDecodeError, TypeError):
            return details

        if not isinstance(payload, dict):
            return details

        errors = payload.get("errors")

        if not isinstance(errors, list) or not errors:
            return details

        first_error = errors[0]

        if not isinstance(first_error, dict):
            return details

        provider_code = first_error.get("code")

        if isinstance(provider_code, int | str):
            details["provider_code"] = provider_code

        provider_message = first_error.get("message")

        if isinstance(provider_message, str):
            details["provider_message"] = cls._sanitize_diagnostic_text(
                provider_message
            )

        return details
