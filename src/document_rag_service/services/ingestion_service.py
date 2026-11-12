"""
Document ingestion orchestration, lifecycle tracking, and chunk persistence.
"""

from sqlalchemy.ext.asyncio import AsyncSession

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
    ) -> None:
        """
        Initialize the document ingestion service.

        Args:
            session: Database session used for transactional persistence.
            embedding_service: Service used for token counting and embedding
                generation.
            chunking_service: Service used to split extracted text into bounded
                chunks.
        """

        self._session = session
        self._documents = DocumentRepository(session)
        self._chunks = DocumentChunkRepository(session)
        self._status_history = DocumentStatusHistoryRepository(session)
        self._errors = ErrorEventRepository(session)
        self._chunking = chunking_service
        self._text_extraction = TextExtractionService()
        self._embedding = embedding_service

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

        The document is first transitioned to ``processing`` and that lifecycle
        change is committed before external embedding work begins. Generated
        embeddings, chunk records, and the final ``ready`` transition are then
        persisted as one transaction.

        If embedding generation or database persistence fails, the active
        transaction is rolled back so partially written chunks are not retained.
        The document is subsequently transitioned to ``failed`` and the failure
        is recorded for observability.

        Args:
            document_id: Identifier of the document receiving the chunks.
            user_id: Identifier of the user who owns the document.
            chunks: Ordered ``(content, token_count)`` pairs to embed and
                persist.

        Raises:
            ValueError: If the target document does not exist.
            Exception: Propagates embedding or persistence failures after the
                database transaction is rolled back and failure state recorded.
        """

        document = await self._documents.get_by_id(
            document_id=document_id,
            user_id=user_id,
        )

        if document is None:
            raise ValueError(f"Document {document_id} not found")

        from_status = document.status

        await self._documents.update_status(
            document=document,
            status="processing",
        )

        await self._status_history.create(
            document_id=document.id,
            from_status=from_status,
            to_status="processing",
            component="ingestion_service",
            operation="persist_chunks",
        )

        await self._session.commit()

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

            await self._documents.update_status(
                document=document,
                status="ready",
            )

            await self._status_history.create(
                document_id=document.id,
                from_status="processing",
                to_status="ready",
                component="ingestion_service",
                operation="persist_chunks",
            )

            await self._session.commit()

        except Exception as exc:
            await self._record_failure(
                document_id=document_id,
                user_id=user_id,
                operation="persist_chunks",
                error_code="CHUNK_PERSISTENCE_FAILED",
                message="Failed to persist document chunks.",
                exception=exc,
            )
            raise

    async def _record_failure(
        self,
        *,
        document_id: int,
        user_id: int,
        operation: str,
        error_code: str,
        message: str,
        exception: Exception,
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
            message: Safe diagnostic message persisted with the error event.
            exception: Original exception used to record the concrete error
                type.
        """

        await self._session.rollback()

        document = await self._documents.get_by_id(
            document_id=document_id,
            user_id=user_id,
        )

        if document is None:
            return

        from_status = document.status

        await self._documents.update_status(
            document=document,
            status="failed",
        )

        await self._status_history.create(
            document_id=document.id,
            from_status=from_status,
            to_status="failed",
            component="ingestion_service",
            operation=operation,
        )

        await self._errors.create(
            document_id=document.id,
            component="ingestion_service",
            operation=operation,
            error_type=type(exception).__name__,
            error_code=error_code,
            message=message,
            retryable=False,
        )

        await self._session.commit()
