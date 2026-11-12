"""
Evaluate dense retrieval quality against the labeled retrieval dataset.
"""

import asyncio
import json
from dataclasses import dataclass
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
from document_rag_service.repositories.user_repository import (
    UserRepository,
)
from document_rag_service.services.embedding_service import EmbeddingService
from document_rag_service.services.retrieval_service import (
    RetrievalResult,
    RetrievalService,
)

EVALS_DIR = Path(__file__).resolve().parent
RETRIEVAL_CASES_PATH = EVALS_DIR / "retrieval_cases.json"

EVAL_USER_EMAIL = "tmemelli@gmail.com"
EVAL_LIMIT = 10_000


@dataclass(frozen=True)
class RelevantChunk:
    """
    Identify one chunk accepted as relevant for an evaluation case.
    """

    document_id: int
    chunk_index: int


@dataclass(frozen=True)
class RetrievalCase:
    """
    Represent one labeled retrieval evaluation case.
    """

    id: str
    query: str
    document_ids: list[int]
    relevant_chunks: list[RelevantChunk]


def find_relevant_rank(
    *,
    case: RetrievalCase,
    retrieval_result: RetrievalResult,
) -> int | None:
    """
    Return the first rank containing a labeled relevant chunk.

    Args:
        case: Evaluation case containing accepted relevant chunks.
        retrieval_result: Ranked retrieval result to inspect.

    Returns:
        One-based rank of the first relevant chunk, or None when not found.
    """

    relevant = {
        (chunk.document_id, chunk.chunk_index) for chunk in case.relevant_chunks
    }

    for position, hit in enumerate(
        retrieval_result.hits,
        start=1,
    ):
        identity = (
            hit.chunk.document_id,
            hit.chunk.chunk_index,
        )

        if identity in relevant:
            return position

    return None


def calculate_metrics(
    ranks: list[int | None],
) -> dict[str, float]:
    """
    Calculate Hit@K and mean reciprocal rank metrics.

    Args:
        ranks: Relevant-chunk ranks for all evaluation cases.

    Returns:
        Mapping containing Hit@K and MRR metrics.

    Raises:
        ValueError: If no retrieval ranks are provided.
    """

    if not ranks:
        raise ValueError("Retrieval ranks cannot be empty")

    total = len(ranks)

    metrics: dict[str, float] = {}

    for k in (1, 3, 5, 10, 20):
        hits = sum(1 for rank in ranks if rank is not None and rank <= k)

        metrics[f"hit@{k}"] = hits / total

    reciprocal_ranks = [0.0 if rank is None else 1.0 / rank for rank in ranks]

    metrics["mrr"] = sum(reciprocal_ranks) / total

    return metrics


def load_cases(path: Path) -> list[RetrievalCase]:
    """
    Load and validate retrieval evaluation cases from JSON.

    Args:
        path: JSON file containing retrieval cases.

    Returns:
        Validated retrieval cases.

    Raises:
        ValueError: If the dataset contains an invalid or empty case.
    """

    raw_cases = json.loads(path.read_text(encoding="utf-8"))

    cases: list[RetrievalCase] = []

    for raw_case in raw_cases:
        case_id = raw_case["id"]
        query = raw_case["query"]
        document_ids = raw_case["document_ids"]
        raw_relevant_chunks = raw_case["relevant_chunks"]

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Retrieval case id cannot be empty")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Retrieval case '{case_id}' has an empty query")

        if not document_ids:
            raise ValueError(f"Retrieval case '{case_id}' has no document ids")

        if not raw_relevant_chunks:
            raise ValueError(f"Retrieval case '{case_id}' has no relevant chunks")

        relevant_chunks = [
            RelevantChunk(
                document_id=item["document_id"],
                chunk_index=item["chunk_index"],
            )
            for item in raw_relevant_chunks
        ]

        cases.append(
            RetrievalCase(
                id=case_id,
                query=query,
                document_ids=document_ids,
                relevant_chunks=relevant_chunks,
            )
        )

    if not cases:
        raise ValueError("Retrieval evaluation dataset is empty")

    return cases


async def main() -> None:
    """
    Execute dense retrieval evaluation against the real application pipeline.
    """

    settings = get_settings()
    engine = create_database_engine(settings)
    session_factory = create_session_factory(engine)

    embedding_provider = create_embedding_provider(settings)
    embedding_service = EmbeddingService(embedding_provider)

    try:
        cases = load_cases(RETRIEVAL_CASES_PATH)

        async with session_factory() as session:
            user_repository = UserRepository(session)

            user = await user_repository.get_by_email(EVAL_USER_EMAIL)

            if user is None:
                raise RuntimeError(f"Evaluation user not found: {EVAL_USER_EMAIL}")

            chunk_repository = DocumentChunkRepository(session)
            document_repository = DocumentRepository(session)

            retrieval_service = RetrievalService(
                chunk_repository,
                document_repository,
                embedding_service,
            )

            ranks: list[int | None] = []

            print("\nDENSE RETRIEVAL BASELINE")
            print("provider=", settings.embedding_provider)
            print("model=", embedding_service.model_name)
            print("dimension=", embedding_service.dimension)
            print("cases=", len(cases))

            for case in cases:
                retrieval_result = await retrieval_service.retrieve(
                    user_id=user.id,
                    document_ids=case.document_ids,
                    queries=[case.query],
                    limit=EVAL_LIMIT,
                )

                rank = find_relevant_rank(
                    case=case,
                    retrieval_result=retrieval_result,
                )

                ranks.append(rank)

                print(
                    case.id,
                    "rank=",
                    rank if rank is not None else "NOT_FOUND",
                )

            metrics = calculate_metrics(ranks)

            print("\nMETRICS")

            for name, value in metrics.items():
                print(
                    name,
                    "=",
                    f"{value:.4f}",
                )

    finally:
        await embedding_service.aclose()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
