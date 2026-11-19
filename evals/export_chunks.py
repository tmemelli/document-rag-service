"""
Export selected document chunks to a JSON holdout dataset.
"""

import asyncio
import json
from pathlib import Path

from document_rag_service.core.settings import get_settings
from document_rag_service.db.session import (
    create_database_engine,
    create_session_factory,
)
from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)

EVALS_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = EVALS_DIR / "chunks_for_evaluation.json"

DOCUMENT_IDS = tuple(range(1, 5))


async def main() -> None:
    """
    Export configured document chunks into the frozen holdout dataset.

    Raises:
        RuntimeError: If one of the configured documents has no chunks.
    """

    settings = get_settings()
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    try:
        exported_chunks: list[dict[str, int | str]] = []

        async with session_factory() as session:
            repository = DocumentChunkRepository(session)

            for document_id in DOCUMENT_IDS:
                chunks = await repository.list_by_document_id(document_id)

                if not chunks:
                    raise RuntimeError(
                        f"Document {document_id} has no chunks to export"
                    )

                for chunk in chunks:
                    exported_chunks.append(
                        {
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.chunk_index,
                            "content": chunk.content,
                        }
                    )

        exported_chunks.sort(
            key=lambda item: (
                int(item["document_id"]),
                int(item["chunk_index"]),
            )
        )

        OUTPUT_PATH.write_text(
            json.dumps(
                exported_chunks,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("EXPORT COMPLETE")
        print("documents=", DOCUMENT_IDS)
        print("chunks=", len(exported_chunks))
        print("output=", OUTPUT_PATH)

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
