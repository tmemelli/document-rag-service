"""Semantic retrieval and multi-query rank fusion for document chunks."""

from dataclasses import dataclass

from document_rag_service.models.document import Document
from document_rag_service.models.document_chunk import DocumentChunk
from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)
from document_rag_service.services.embedding_service import EmbeddingService


@dataclass(frozen=True)
class RetrievalHit:
    """
    Describe one chunk selected by the retrieval pipeline.

    Attributes:
        chunk: Retrieved document chunk.
        score: Final ranking score. For single-query retrieval this is the
            dense similarity; for multi-query retrieval this is the fused RRF
            score.
        best_similarity: Highest dense similarity observed for the chunk across
            all retrieval queries.
        matched_query_indexes: One-based indexes of retrieval queries in which
            the chunk appeared.
    """

    chunk: DocumentChunk
    score: float
    best_similarity: float
    matched_query_indexes: tuple[int, ...]


@dataclass(frozen=True)
class RetrievalResult:
    """
    Represent the complete result of a retrieval operation.

    Attributes:
        hits: Final ranked retrieval hits.
        documents: Validated documents included in the retrieval scope.
        retrieval_queries: Normalized queries executed against the vector
            index.
        strategy: Retrieval strategy used to rank the final hits.
    """

    hits: list[RetrievalHit]
    documents: list[Document]
    retrieval_queries: list[str]
    strategy: str


@dataclass
class _FusedCandidate:
    """
    Accumulate ranking information for one multi-query retrieval candidate.

    Attributes:
        chunk: Document chunk represented by the candidate.
        rrf_score: Accumulated Reciprocal Rank Fusion score.
        best_similarity: Highest dense similarity observed across retrieval
            queries.
        matched_query_indexes: One-based indexes of queries that retrieved the
            chunk.
    """

    chunk: DocumentChunk
    rrf_score: float
    best_similarity: float
    matched_query_indexes: list[int]


class RetrievalService:
    """
    Retrieve relevant document chunks using dense search and optional RRF fusion.

    The service validates the user-scoped document selection, generates query
    embeddings, and delegates vector similarity search to the chunk repository.

    A single retrieval query produces a standard dense ranking. When multiple
    queries are supplied, candidate rankings are combined with Reciprocal Rank
    Fusion while retaining each chunk's strongest dense similarity and the
    queries in which it appeared.
    """

    def __init__(
        self,
        chunk_repository: DocumentChunkRepository,
        document_repository: DocumentRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        """
        Initialize the retrieval service.

        Args:
            chunk_repository: Repository used for pgvector similarity search.
            document_repository: Repository used to validate user-scoped
                document selections.
            embedding_service: Service used to generate query embeddings.
        """

        self._chunks = chunk_repository
        self._documents = document_repository
        self._embedding = embedding_service

    async def retrieve(
        self,
        *,
        user_id: int,
        document_ids: list[int],
        queries: list[str],
        limit: int = 5,
        candidate_limit: int | None = None,
        rrf_k: int = 60,
    ) -> RetrievalResult:
        """
        Retrieve and rank chunks from the selected user-owned documents.

        Retrieval queries are normalized and deduplicated before execution.
        Each query is independently embedded and searched against the selected
        documents.

        When only one query remains, chunks are ranked directly by dense
        similarity. With multiple queries, each ranked candidate list
        contributes to a Reciprocal Rank Fusion score. Dense similarity is
        retained separately as an observability metric and deterministic
        secondary ranking criterion.

        Args:
            user_id: Identifier of the user performing retrieval.
            document_ids: Identifiers of documents that define the search
                scope.
            queries: One or more semantic retrieval queries.
            limit: Maximum number of final retrieval hits.
            candidate_limit: Maximum candidates retrieved per query before
                final ranking or RRF fusion. Defaults to at least ten while
                never falling below ``limit``.
            rrf_k: Rank constant used by Reciprocal Rank Fusion.

        Returns:
            Retrieval result containing ranked hits, validated source
            documents, normalized queries, and the retrieval strategy used.

        Raises:
            ValueError: If no document is selected, no valid retrieval query
                is supplied, ranking limits are invalid, one or more documents
                do not belong to the user, or any selected document is not
                ready for retrieval.
        """

        retrieval_queries = self._normalize_queries(queries)

        if not document_ids:
            raise ValueError("At least one document must be selected")

        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        if candidate_limit is None:
            candidate_limit = max(limit, 10)

        if candidate_limit < limit:
            raise ValueError("Candidate limit cannot be smaller than final limit")

        if rrf_k <= 0:
            raise ValueError("RRF rank constant must be greater than zero")

        unique_document_ids = list(dict.fromkeys(document_ids))

        documents = await self._documents.list_by_ids(
            document_ids=unique_document_ids,
            user_id=user_id,
        )

        if len(documents) != len(unique_document_ids):
            raise ValueError("One or more selected documents are invalid")

        if any(document.status != "ready" for document in documents):
            raise ValueError("All selected documents must be ready")

        ranked_lists: list[list[tuple[DocumentChunk, float]]] = []

        for query in retrieval_queries:
            query_embedding = await self._embedding.embed_query(query)

            results = await self._chunks.search_similar(
                user_id=user_id,
                document_ids=unique_document_ids,
                query_embedding=query_embedding,
                limit=candidate_limit,
            )

            ranked_lists.append(
                [(chunk, 1.0 - distance) for chunk, distance in results]
            )

        if len(ranked_lists) == 1:
            hits = [
                RetrievalHit(
                    chunk=chunk,
                    score=similarity,
                    best_similarity=similarity,
                    matched_query_indexes=(1,),
                )
                for chunk, similarity in ranked_lists[0][:limit]
            ]

            return RetrievalResult(
                hits=hits,
                documents=documents,
                retrieval_queries=retrieval_queries,
                strategy="dense",
            )

        fused_candidates: dict[int, _FusedCandidate] = {}

        for query_index, ranked_results in enumerate(ranked_lists, start=1):
            for rank, (chunk, similarity) in enumerate(ranked_results, start=1):
                rrf_contribution = 1.0 / (rrf_k + rank)
                candidate = fused_candidates.get(chunk.id)

                if candidate is None:
                    fused_candidates[chunk.id] = _FusedCandidate(
                        chunk=chunk,
                        rrf_score=rrf_contribution,
                        best_similarity=similarity,
                        matched_query_indexes=[query_index],
                    )
                    continue

                candidate.rrf_score += rrf_contribution
                candidate.best_similarity = max(
                    candidate.best_similarity,
                    similarity,
                )
                candidate.matched_query_indexes.append(query_index)

        ordered_candidates = sorted(
            fused_candidates.values(),
            key=lambda candidate: (
                -candidate.rrf_score,
                -candidate.best_similarity,
                candidate.chunk.id,
            ),
        )

        hits = [
            RetrievalHit(
                chunk=candidate.chunk,
                score=candidate.rrf_score,
                best_similarity=candidate.best_similarity,
                matched_query_indexes=tuple(candidate.matched_query_indexes),
            )
            for candidate in ordered_candidates[:limit]
        ]

        return RetrievalResult(
            hits=hits,
            documents=documents,
            retrieval_queries=retrieval_queries,
            strategy="multi_query_rrf",
        )

    def _normalize_queries(self, queries: list[str]) -> list[str]:
        """
        Normalize and deduplicate retrieval queries while preserving order.

        Queries are stripped of surrounding whitespace and compared
        case-insensitively. The first occurrence of each unique query is kept
        so query priority remains stable.

        Args:
            queries: Retrieval-query candidates.

        Returns:
            Non-empty unique queries in their original order.

        Raises:
            ValueError: If no queries are provided or any supplied query is
                empty or whitespace-only.
        """

        if not queries:
            raise ValueError("At least one retrieval query is required")

        unique_queries: list[str] = []
        seen: set[str] = set()

        for query in queries:
            normalized = query.strip()

            if not normalized:
                raise ValueError("Retrieval queries cannot be empty")

            key = normalized.casefold()

            if key in seen:
                continue

            unique_queries.append(normalized)
            seen.add(key)

        return unique_queries
