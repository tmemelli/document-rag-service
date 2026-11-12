"""
Manually ingest one document through the complete ingestion pipeline.
"""

import argparse
import asyncio
from pathlib import Path

from document_rag_service.core.embedding_provider_factory import (
    create_embedding_provider,
)
from document_rag_service.core.settings import get_settings
from document_rag_service.db.session import (
    create_database_engine,
    create_session_factory,
)
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
from document_rag_service.repositories.user_repository import UserRepository
from document_rag_service.services.chunking_service import ChunkingService
from document_rag_service.services.document_service import DocumentService
from document_rag_service.services.embedding_service import EmbeddingService
from document_rag_service.services.ingestion_service import IngestionService

DEBUG_EMAIL = "tmemelli@gmail.com"

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".rtf": "application/rtf",
    ".fodt": "application/vnd.oasis.opendocument.text-flat-xml",
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for manual document ingestion.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Ingest exactly one document into the RAG service."
    )
    parser.add_argument("file_path", type=Path)
    parser.add_argument(
        "--title",
        help="Document title. Defaults to the file name without extension.",
    )
    return parser.parse_args()


def resolve_mime_type(file_path: Path) -> str:
    """
    Resolve the supported MIME type for a document path.

    Args:
        file_path: Uploaded document path.

    Returns:
        Supported MIME type.

    Raises:
        ValueError: If the file extension is unsupported.
    """

    extension = file_path.suffix.lower()

    try:
        return MIME_TYPES[extension]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported document extension: {extension or '<missing>'}"
        ) from exc


async def main(
    *,
    file_path: Path,
    title: str | None,
) -> None:
    """
    Execute manual ingestion and print persisted pipeline state.

    Args:
        file_path: File to ingest.
        title: Optional document title override.
    """

    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    content = file_path.read_bytes()

    if not content:
        raise ValueError("Document file cannot be empty")

    mime_type = resolve_mime_type(file_path)
    document_title = title or file_path.stem

    settings = get_settings()
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)
    embedding_provider = create_embedding_provider(settings)
    embedding_service = EmbeddingService(embedding_provider)
    chunking_service = ChunkingService(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    try:
        async with session_factory() as session:
            user_repository = UserRepository(session)

            user = await user_repository.get_by_email(DEBUG_EMAIL)

            if user is None:
                raise RuntimeError(f"Debug user not found: {DEBUG_EMAIL}")

            document_service = DocumentService(session)
            ingestion_service = IngestionService(
                session,
                embedding_service,
                chunking_service,
            )

            document_repository = DocumentRepository(session)
            chunk_repository = DocumentChunkRepository(session)
            history_repository = DocumentStatusHistoryRepository(session)
            error_repository = ErrorEventRepository(session)

            document = await document_service.register_uploaded_document(
                user_id=user.id,
                title=document_title,
                file_name=file_path.name,
                mime_type=mime_type,
                content=content,
            )

            ingestion_error: Exception | None = None

            try:
                await ingestion_service.ingest_file_content(
                    document_id=document.id,
                    user_id=user.id,
                    content=content,
                    file_name=file_path.name,
                )
            except Exception as exc:
                ingestion_error = exc

            refreshed_document = await document_repository.get_by_id(
                document_id=document.id,
                user_id=user.id,
            )

            if refreshed_document is None:
                raise RuntimeError(f"Document {document.id} not found after ingestion")

            chunks = await chunk_repository.list_by_document_id(
                refreshed_document.id,
            )
            history = await history_repository.list_by_document_id(
                refreshed_document.id,
            )
            errors = await error_repository.list_by_document_id(
                refreshed_document.id,
            )

            print("\nUSER")
            print("id=", user.id)
            print("email=", user.email)

            print("\nDOCUMENT")
            print("id=", refreshed_document.id)
            print("title=", refreshed_document.title)
            print("file_name=", refreshed_document.file_name)
            print("status=", refreshed_document.status)
            print("checksum=", refreshed_document.checksum)

            print("\nCHUNKS")
            print("total=", len(chunks))

            if chunks:
                first = chunks[0]
                last = chunks[-1]

                print(
                    "first:",
                    "id=",
                    first.id,
                    "chunk_index=",
                    first.chunk_index,
                    "characters=",
                    len(first.content),
                    "tokens=",
                    first.token_count,
                    "embedding_dimensions=",
                    len(first.embedding) if first.embedding is not None else None,
                )

                print(
                    "last:",
                    "id=",
                    last.id,
                    "chunk_index=",
                    last.chunk_index,
                    "characters=",
                    len(last.content),
                    "tokens=",
                    last.token_count,
                    "embedding_dimensions=",
                    len(last.embedding) if last.embedding is not None else None,
                )

            print("\nSTATUS HISTORY")
            for event in history:
                print(event.from_status, "->", event.to_status)

            print("\nERRORS")
            for error in errors:
                print(
                    error.error_type,
                    error.error_code,
                    error.message,
                )

            if ingestion_error is not None:
                raise ingestion_error

    finally:
        await embedding_service.aclose()
        await engine.dispose()


if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        main(
            file_path=args.file_path,
            title=args.title,
        )
    )
