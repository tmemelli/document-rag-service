"""
Benchmark local embedding configurations against frozen retrieval datasets.
"""

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EVALS_DIR = Path(__file__).resolve().parent

CHUNK_DATASET_PATH = EVALS_DIR / "chunks_for_evaluation.json"

RETRIEVAL_CASES_PATH = EVALS_DIR / "retrieval_cases.json"


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Describe one embedding configuration under evaluation."""

    name: str
    model_name: str
    dimension: int
    query_prompt_name: str | None = None


@dataclass(frozen=True)
class BenchmarkChunk:
    """Represent one frozen document chunk used by the benchmark."""

    document_id: int
    chunk_index: int
    content: str


@dataclass(frozen=True)
class RelevantChunk:
    """Identify one chunk labeled as relevant for a retrieval case."""

    document_id: int
    chunk_index: int


@dataclass(frozen=True)
class RetrievalCase:
    """Represent one labeled retrieval benchmark case."""

    id: str
    query: str
    document_ids: list[int]
    relevant_chunks: list[RelevantChunk]


MODEL_CONFIGS = (
    EmbeddingModelConfig(
        name="qwen3-embedding-0.6b-384",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        dimension=384,
        query_prompt_name="query",
    ),
    EmbeddingModelConfig(
        name="qwen3-embedding-0.6b-512",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        dimension=512,
        query_prompt_name="query",
    ),
)


def load_chunks(path: Path) -> list[BenchmarkChunk]:
    """
    Load frozen document chunks used by the benchmark.

    Args:
        paths: JSON datasets containing persisted chunk snapshots.

    Returns:
        Combined benchmark chunks.

    Raises:
        ValueError: If no chunks are available or duplicate chunk identities exist.
    """

    raw_chunks = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[BenchmarkChunk] = []

    for item in raw_chunks:
        content = item["content"]

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"Chunk {item['document_id']}:{item['chunk_index']} "
                "has empty content"
            )

        chunks.append(
            BenchmarkChunk(
                document_id=item["document_id"],
                chunk_index=item["chunk_index"],
                content=content,
            )
        )

    if not chunks:
        raise ValueError("Embedding benchmark chunk dataset is empty")

    identities = [(chunk.document_id, chunk.chunk_index) for chunk in chunks]

    if len(identities) != len(set(identities)):
        raise ValueError("Embedding benchmark contains duplicate chunks")

    return chunks


def load_cases(path: Path) -> list[RetrievalCase]:
    """
    Load and validate labeled retrieval benchmark cases.

    Args:
        path: JSON file containing retrieval cases.

    Returns:
        Validated retrieval cases.

    Raises:
        ValueError: If the dataset contains invalid or empty cases.
    """

    raw_cases = json.loads(path.read_text(encoding="utf-8"))

    cases: list[RetrievalCase] = []

    for item in raw_cases:
        case_id = item["id"]
        query = item["query"]
        document_ids = item["document_ids"]
        raw_relevant_chunks = item["relevant_chunks"]

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
                document_id=chunk["document_id"],
                chunk_index=chunk["chunk_index"],
            )
            for chunk in raw_relevant_chunks
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
        raise ValueError("Retrieval benchmark dataset is empty")

    return cases


def validate_dataset(
    *,
    chunks: list[BenchmarkChunk],
    cases: list[RetrievalCase],
) -> None:
    """
    Ensure every labeled relevant chunk exists in the frozen corpus.

    Args:
        chunks: Frozen benchmark chunks.
        cases: Labeled retrieval cases.

    Raises:
        ValueError: If a relevant chunk is absent from the corpus.
    """

    available = {(chunk.document_id, chunk.chunk_index) for chunk in chunks}

    for case in cases:
        for relevant_chunk in case.relevant_chunks:
            identity = (
                relevant_chunk.document_id,
                relevant_chunk.chunk_index,
            )

            if identity not in available:
                raise ValueError(
                    f"Retrieval case '{case.id}' references missing chunk "
                    f"{identity[0]}:{identity[1]}"
                )


def load_model(config: EmbeddingModelConfig) -> SentenceTransformer:
    """
    Load one sentence-transformer configuration.

    Args:
        config: Embedding configuration to load.

    Returns:
        Loaded sentence-transformer model.
    """

    return SentenceTransformer(
        config.model_name,
        truncate_dim=config.dimension,
    )


def encode_documents(
    *,
    model: SentenceTransformer,
    chunks: list[BenchmarkChunk],
) -> np.ndarray:
    """
    Encode frozen document chunks into normalized vectors.

    Args:
        model: Loaded embedding model.
        chunks: Chunks to encode.

    Returns:
        Matrix of normalized document embeddings.
    """

    return model.encode(
        [chunk.content for chunk in chunks],
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )


def encode_query(
    *,
    model: SentenceTransformer,
    config: EmbeddingModelConfig,
    query: str,
) -> np.ndarray:
    """
    Encode one retrieval query into a normalized vector.

    Args:
        model: Loaded embedding model.
        config: Embedding configuration.
        query: Query text.

    Returns:
        Normalized query embedding.
    """

    if config.query_prompt_name is not None:
        return model.encode(
            [query],
            prompt_name=config.query_prompt_name,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

    return model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


def evaluate_case(
    *,
    case: RetrievalCase,
    model: SentenceTransformer,
    config: EmbeddingModelConfig,
    chunks: list[BenchmarkChunk],
    document_embeddings: np.ndarray,
) -> int | None:
    """
    Return the first relevant rank for one benchmark case.

    Args:
        case: Retrieval case to evaluate.
        model: Loaded embedding model.
        config: Embedding configuration.
        chunks: Frozen benchmark corpus.
        document_embeddings: Precomputed document vectors.

    Returns:
        One-based rank of the first relevant chunk, or None when not found.
    """

    query_embedding = encode_query(
        model=model,
        config=config,
        query=case.query,
    )

    candidate_indexes = [
        index
        for index, chunk in enumerate(chunks)
        if chunk.document_id in case.document_ids
    ]

    if not candidate_indexes:
        raise ValueError(f"Retrieval case '{case.id}' has no candidate chunks")

    candidate_embeddings = document_embeddings[candidate_indexes]

    similarities = candidate_embeddings @ query_embedding

    local_ranking = np.argsort(similarities)[::-1]

    relevant = {
        (chunk.document_id, chunk.chunk_index) for chunk in case.relevant_chunks
    }

    for position, local_index in enumerate(
        local_ranking,
        start=1,
    ):
        chunk = chunks[candidate_indexes[local_index]]

        identity = (
            chunk.document_id,
            chunk.chunk_index,
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
        ranks: Relevant-chunk ranks for all benchmark cases.

    Returns:
        Mapping containing Hit@K and MRR metrics.

    Raises:
        ValueError: If no benchmark ranks are provided.
    """

    if not ranks:
        raise ValueError("Embedding benchmark ranks cannot be empty")

    total = len(ranks)

    metrics: dict[str, float] = {}

    for k in (1, 3, 5, 10, 20):
        hits = sum(1 for rank in ranks if rank is not None and rank <= k)

        metrics[f"hit@{k}"] = hits / total

    reciprocal_ranks = [0.0 if rank is None else 1.0 / rank for rank in ranks]

    metrics["mrr"] = sum(reciprocal_ranks) / total

    return metrics


def release_model(
    model: SentenceTransformer,
    document_embeddings: np.ndarray,
) -> None:
    """
    Release model resources between benchmark configurations.

    Args:
        model: Loaded sentence-transformer model.
        document_embeddings: Embedding matrix associated with the model.
    """

    del model
    del document_embeddings

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    """Run the frozen embedding retrieval benchmark."""

    cases = load_cases(RETRIEVAL_CASES_PATH)
    chunks = load_chunks(CHUNK_DATASET_PATH)

    validate_dataset(
        chunks=chunks,
        cases=cases,
    )

    document_ids = sorted(
        {document_id for case in cases for document_id in case.document_ids}
    )

    print("\nEMBEDDING RETRIEVAL BENCHMARK")
    print("cases=", len(cases))
    print("documents=", document_ids)
    print("chunks=", len(chunks))

    for config in MODEL_CONFIGS:
        print(f"\n{'=' * 70}")
        print("MODEL:", config.name)
        print("=" * 70)

        model = load_model(config)

        document_embeddings = encode_documents(
            model=model,
            chunks=chunks,
        )

        ranks: list[int | None] = []

        for case in cases:
            rank = evaluate_case(
                case=case,
                model=model,
                config=config,
                chunks=chunks,
                document_embeddings=document_embeddings,
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
            print(name, "=", f"{value:.4f}")

        release_model(
            model,
            document_embeddings,
        )


if __name__ == "__main__":
    main()
