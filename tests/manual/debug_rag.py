"""
Manually execute the complete RAG pipeline and print observability details.
"""

import argparse
import asyncio

import httpx

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
from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.repositories.user_repository import UserRepository
from document_rag_service.services.context_builder_service import (
    ContextBuilderService,
)
from document_rag_service.services.context_expansion_service import (
    ContextExpansionService,
)
from document_rag_service.services.embedding_service import EmbeddingService
from document_rag_service.services.generation_service import GenerationService
from document_rag_service.services.query_transformation_service import (
    QueryTransformationService,
)
from document_rag_service.services.rag_service import RagService
from document_rag_service.services.retrieval_service import RetrievalService

DEBUG_EMAIL = "tmemelli@gmail.com"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the end-to-end RAG debugger.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Run a complete end-to-end RAG query.")
    parser.add_argument(
        "--documents",
        type=int,
        nargs="+",
        required=True,
        dest="document_ids",
        help="One or more document IDs to search.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question to answer using the selected documents.",
    )
    parser.add_argument(
        "--language",
        default="pt-BR",
        help="Language used in the final answer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of chunks to retrieve.",
    )
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=1,
        help=(
            "Number of adjacent chunks to include on each side of every "
            "retrieved chunk."
        ),
    )
    parser.add_argument(
        "--transform-query",
        action="store_true",
        help=(
            "Transform the original query into focused retrieval queries "
            "and fuse results with RRF."
        ),
    )
    return parser.parse_args()


async def main(
    *,
    document_ids: list[int],
    query: str,
    language: str,
    limit: int,
    neighbor_window: int,
    transform_query: bool,
) -> None:
    """
    Execute the complete RAG pipeline for one manual query.

    Args:
        document_ids: Documents eligible for retrieval.
        query: Original user question.
        language: Requested answer language.
        limit: Number of final retrieval anchors.
        neighbor_window: Number of neighboring chunks added per anchor.
        transform_query: Whether query transformation should be enabled.
    """

    settings = get_settings()
    llm_http_client = httpx.AsyncClient(
        base_url=(settings.llm_inference_base_url.rstrip("/") + "/"),
        timeout=settings.llm_inference_timeout_seconds,
    )
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)
    embedding_provider = create_embedding_provider(settings)
    embedding_service = EmbeddingService(embedding_provider)

    try:
        llm_provider = LLMInferenceProvider(
            client=llm_http_client,
            provider=settings.llm_inference_provider,
            model=settings.llm_inference_model,
            temperature=settings.llm_inference_temperature,
            max_tokens=settings.llm_inference_max_tokens,
        )

        generation_service = GenerationService(
            llm_provider,
            max_prompt_characters=(settings.llm_inference_max_prompt_characters),
        )

        query_transformation_service = None

        if transform_query:
            transformation_provider = LLMInferenceProvider(
                client=llm_http_client,
                provider=settings.llm_inference_provider,
                model=settings.llm_inference_model,
                temperature=settings.query_transformation_temperature,
                max_tokens=settings.query_transformation_max_tokens,
            )

            query_transformation_service = QueryTransformationService(
                transformation_provider,
                max_generated_queries=(
                    settings.query_transformation_max_generated_queries
                ),
                max_attempts=settings.query_transformation_max_attempts,
            )

        async with session_factory() as session:
            user_repository = UserRepository(session)

            user = await user_repository.get_by_email(DEBUG_EMAIL)

            if user is None:
                raise RuntimeError(f"Debug user not found: {DEBUG_EMAIL}")

            document_repository = DocumentRepository(session)
            chunk_repository = DocumentChunkRepository(session)

            retrieval_service = RetrievalService(
                chunk_repository,
                document_repository,
                embedding_service,
            )

            context_expansion_service = ContextExpansionService(chunk_repository)
            context_builder = ContextBuilderService()

            rag_service = RagService(
                retrieval_service,
                context_expansion_service,
                context_builder,
                generation_service,
                query_transformation_service=(query_transformation_service),
            )

            result = await rag_service.answer(
                user_id=user.id,
                document_ids=document_ids,
                query=query,
                language=language,
                limit=limit,
                neighbor_window=neighbor_window,
                transform_query=transform_query,
            )

            print("\nQUERY")
            print(query)

            print("\nRETRIEVAL")
            print("strategy=", result.retrieval_strategy)
            print("retrieval_queries:")
            for position, retrieval_query in enumerate(
                result.retrieval_queries,
                start=1,
            ):
                print(f"  {position}. {retrieval_query}")

            print("\nRETRIEVAL HITS")

            for position, hit in enumerate(
                result.retrieval_hits,
                start=1,
            ):
                print(
                    f"{position}. "
                    f"document_id={hit.document_id} "
                    f"title={hit.document_title!r} "
                    f"chunk_index={hit.chunk_index} "
                    f"score={hit.score:.6f} "
                    f"best_similarity={hit.best_similarity:.6f} "
                    f"matched_query_indexes={hit.matched_query_indexes}"
                )

            print("\nCONTEXT CHUNKS")

            for position, chunk in enumerate(
                result.context_chunks,
                start=1,
            ):
                print(
                    f"{position}. "
                    f"document_id={chunk.document_id} "
                    f"title={chunk.document_title!r} "
                    f"chunk_index={chunk.chunk_index} "
                    f"role={chunk.role}"
                )

            print("\nCONTEXT BUDGET")
            print("included_chunks=", result.context_chunks_count)
            print("context_characters=", result.context_characters)
            print(
                "max_context_characters=",
                result.max_context_characters,
            )

            print("\nANSWER")
            print(result.answer)

            print("\nSOURCES")
            for source in result.sources:
                print(f"- {source}")

    finally:
        await llm_http_client.aclose()
        await embedding_service.aclose()
        await engine.dispose()


if __name__ == "__main__":
    args = parse_args()

    asyncio.run(
        main(
            document_ids=args.document_ids,
            query=args.query,
            language=args.language,
            limit=args.limit,
            neighbor_window=args.neighbor_window,
            transform_query=args.transform_query,
        )
    )
