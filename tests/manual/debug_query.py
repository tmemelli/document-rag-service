"""
Manually inspect retrieval, query transformation, and context expansion.
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
from document_rag_service.services.query_transformation_service import (
    QueryTransformationService,
)
from document_rag_service.services.retrieval_service import RetrievalService

DEBUG_EMAIL = "tmemelli@gmail.com"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the manual retrieval debugger.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run a RAG retrieval query against one or more selected documents."
        )
    )
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
        help="Question to retrieve context for.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of final retrieval anchors.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=10,
        help="Candidates retrieved per retrieval query before rank fusion.",
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
        help=("Generate focused retrieval queries and fuse their rankings with RRF."),
    )
    return parser.parse_args()


async def main(
    *,
    document_ids: list[int],
    query: str,
    limit: int,
    candidate_limit: int,
    neighbor_window: int,
    transform_query: bool,
) -> None:
    """
    Execute retrieval debugging for one query.

    Args:
        document_ids: Documents eligible for retrieval.
        query: Original user query.
        limit: Number of final retrieval anchors.
        candidate_limit: Candidate count per retrieval query.
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
        retrieval_queries = [query]
        used_fallback = False
        attempt_count = 0

        if transform_query:
            transformation_provider = LLMInferenceProvider(
                client=llm_http_client,
                provider=settings.llm_inference_provider,
                model=settings.llm_inference_model,
                temperature=settings.query_transformation_temperature,
                max_tokens=settings.query_transformation_max_tokens,
            )

            query_transformer = QueryTransformationService(
                transformation_provider,
                max_generated_queries=(
                    settings.query_transformation_max_generated_queries
                ),
                max_attempts=settings.query_transformation_max_attempts,
            )

            transformation_result = await query_transformer.transform(query)
            retrieval_queries = transformation_result.retrieval_queries
            used_fallback = transformation_result.used_fallback
            attempt_count = transformation_result.attempt_count

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

            retrieval_result = await retrieval_service.retrieve(
                user_id=user.id,
                document_ids=document_ids,
                queries=retrieval_queries,
                limit=limit,
                candidate_limit=max(candidate_limit, limit),
            )

            retrieved_chunks = [hit.chunk for hit in retrieval_result.hits]

            context_chunks = await context_expansion_service.expand(
                user_id=user.id,
                retrieved_chunks=retrieved_chunks,
                neighbor_window=neighbor_window,
            )

            context = context_builder.build(context_chunks)

            print("\nUSER")
            print("id=", user.id)
            print("email=", user.email)

            print("\nSELECTED DOCUMENTS")
            print(document_ids)

            print("\nQUERY TRANSFORMATION")
            print("enabled=", transform_query)
            print("fallback=", used_fallback)
            print("original_query:", query)
            print("attempt_count=", attempt_count)
            print("retrieval_queries:")
            for position, retrieval_query in enumerate(
                retrieval_result.retrieval_queries,
                start=1,
            ):
                print(f"  {position}. {retrieval_query}")

            print("\nRETRIEVAL")
            print("strategy=", retrieval_result.strategy)
            print(
                "candidate_limit_per_query=",
                max(candidate_limit, limit),
            )

            for position, hit in enumerate(
                retrieval_result.hits,
                start=1,
            ):
                print(
                    position,
                    "document_id=",
                    hit.chunk.document_id,
                    "chunk_index=",
                    hit.chunk.chunk_index,
                    "score=",
                    f"{hit.score:.6f}",
                    "best_similarity=",
                    f"{hit.best_similarity:.6f}",
                    "matched_queries=",
                    list(hit.matched_query_indexes),
                )

            print("\nCONTEXT EXPANSION")
            print(
                f"top_k={len(retrieval_result.hits)}",
                f"neighbor_window={neighbor_window}",
                f"context_chunks={len(context_chunks)}",
            )

            anchor_locations = {
                (hit.chunk.document_id, hit.chunk.chunk_index)
                for hit in retrieval_result.hits
            }

            for position, chunk in enumerate(
                context_chunks,
                start=1,
            ):
                role = (
                    "anchor"
                    if (
                        chunk.document_id,
                        chunk.chunk_index,
                    )
                    in anchor_locations
                    else "neighbor"
                )
                print(
                    position,
                    "document_id=",
                    chunk.document_id,
                    "chunk_index=",
                    chunk.chunk_index,
                    "role=",
                    role,
                )

            print("\nCONTEXT")
            print(context)

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
            limit=args.limit,
            candidate_limit=args.candidate_limit,
            neighbor_window=args.neighbor_window,
            transform_query=args.transform_query,
        )
    )
