"""Orchestrate retrieval, context expansion, bounded context assembly, and generation."""

from dataclasses import dataclass

from document_rag_service.services.context_builder_service import (
    ContextBuilderService,
)
from document_rag_service.services.context_expansion_service import (
    ContextExpansionService,
)
from document_rag_service.services.generation_service import (
    GenerationService,
)
from document_rag_service.services.query_transformation_service import (
    QueryTransformationService,
)
from document_rag_service.services.retrieval_service import (
    RetrievalService,
)


@dataclass(frozen=True)
class RagRetrievalHit:
    """
    Describe one retrieval anchor selected by the retrieval pipeline.

    Attributes:
        document_id: Identifier of the document containing the retrieved chunk.
        document_title: Human-readable title of the source document.
        chunk_index: Zero-based position of the retrieved chunk in the document.
        score: Final score used to rank the retrieval hit.
        best_similarity: Highest semantic similarity observed for the chunk
            across the executed retrieval queries.
        matched_query_indexes: One-based indexes of retrieval queries that
            matched the chunk.
    """

    document_id: int
    document_title: str
    chunk_index: int
    score: float
    best_similarity: float
    matched_query_indexes: tuple[int, ...]


@dataclass(frozen=True)
class RagContextChunk:
    """
    Describe a chunk produced by context expansion.

    The metadata identifies whether a chunk was retrieved directly as an
    anchor or included because it neighbors an anchor. These entries represent
    the expanded context candidates available to context assembly; the final
    prompt budget may include only a subset of them.

    Attributes:
        document_id: Identifier of the document containing the chunk.
        document_title: Human-readable title of the source document.
        chunk_index: Zero-based position of the chunk in the document.
        role: Context role, either ``anchor`` for directly retrieved chunks or
            ``neighbor`` for chunks added during context expansion.
    """

    document_id: int
    document_title: str
    chunk_index: int
    role: str


@dataclass(frozen=True)
class RagResult:
    """
    Represent the complete result of a RAG question-answering operation.

    ``context_chunks`` exposes all candidates produced by context expansion for
    observability, while ``context_chunks_count`` reports how many of those
    candidates actually fit within the final prompt context budget.

    Attributes:
        answer: Grounded answer generated from the assembled context.
        sources: Unique document titles represented in the context actually
            provided to the language model.
        retrieval_strategy: Retrieval strategy used to rank evidence.
        retrieval_queries: Queries executed against the vector index.
        retrieval_hits: Ranked chunks selected directly by retrieval.
        context_chunks: Expanded anchor and neighboring chunks available to
            context assembly before prompt-budget truncation.
        context_chunks_count: Number of expanded chunks actually included in
            the final context sent for answer generation.
        context_characters: Character length of the final assembled context.
        max_context_characters: Maximum context-character budget available for
            the generation request.
    """

    answer: str
    sources: list[str]
    retrieval_strategy: str
    retrieval_queries: list[str]
    retrieval_hits: list[RagRetrievalHit]
    context_chunks: list[RagContextChunk]
    context_chunks_count: int
    context_characters: int
    max_context_characters: int


class RagService:
    """
    Coordinate the end-to-end retrieval-augmented generation workflow.

    The service optionally transforms the user's question into focused
    retrieval queries, retrieves semantically relevant anchor chunks, expands
    them with neighboring chunks, builds a context constrained by the
    generation prompt budget, and requests a grounded answer from the
    generation service.

    It also assembles retrieval and context metadata used for observability,
    including the distinction between directly retrieved anchors, neighboring
    chunks, and the subset that ultimately fits within the final context.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_expansion_service: ContextExpansionService,
        context_builder: ContextBuilderService,
        generation_service: GenerationService,
        query_transformation_service: QueryTransformationService | None = None,
    ) -> None:
        """
        Initialize the RAG orchestration service.

        Args:
            retrieval_service: Service responsible for semantic retrieval and
                retrieval-result ranking.
            context_expansion_service: Service that adds neighboring chunks
                around retrieved anchors.
            context_builder: Service that assembles expanded chunks into a
                bounded generation context.
            generation_service: Service that generates grounded answers from
                the assembled context.
            query_transformation_service: Optional service used to generate
                focused retrieval queries before semantic search.
        """

        self._retrieval = retrieval_service
        self._context_expansion = context_expansion_service
        self._context_builder = context_builder
        self._generation = generation_service
        self._query_transformation = query_transformation_service

    async def answer(
        self,
        *,
        user_id: int,
        document_ids: list[int],
        query: str,
        language: str,
        limit: int = 5,
        neighbor_window: int = 1,
        transform_query: bool = False,
    ) -> RagResult:
        """
        Execute the complete RAG workflow for a user question.

        The original query is used directly unless query transformation is
        enabled. Retrieved anchors are expanded with neighboring chunks and the
        resulting candidates are passed to the context builder, which keeps
        only the chunks that fit within the character budget available to the
        generation prompt.

        Context observability intentionally retains metadata for every expanded
        chunk, while source attribution and ``context_chunks_count`` reflect
        only the subset actually included in the final generated context.

        Args:
            user_id: Identifier of the user performing the query.
            document_ids: Identifiers of documents that define the retrieval
                scope.
            query: Original user question.
            language: Requested language for the generated answer.
            limit: Maximum number of retrieval anchors to select.
            neighbor_window: Number of adjacent chunks to include on each side
                of every retrieved anchor.
            transform_query: Whether to generate focused retrieval queries
                before semantic retrieval.

        Returns:
            Complete RAG result containing the answer, source attribution,
            retrieval diagnostics, expanded-context metadata, and context-budget
            information.

        Raises:
            RuntimeError: If query transformation is requested without a
                configured transformation service or if returned chunk metadata
                references an unknown document.
            ValueError: If an underlying retrieval, context, or generation
                operation rejects invalid input.
        """

        retrieval_queries = [query]

        if transform_query:
            if self._query_transformation is None:
                raise RuntimeError(
                    "Query transformation requested but service is not configured"
                )

            transformation_result = await self._query_transformation.transform(query)

            retrieval_queries = transformation_result.retrieval_queries

        retrieval_result = await self._retrieval.retrieve(
            user_id=user_id,
            document_ids=document_ids,
            queries=retrieval_queries,
            limit=limit,
        )

        context_chunks = await self._context_expansion.expand(
            user_id=user_id,
            retrieved_chunks=[hit.chunk for hit in retrieval_result.hits],
            neighbor_window=neighbor_window,
        )

        available_context_characters = (
            self._generation.get_available_context_characters(
                query=query,
                language=language,
            )
        )

        context_result = self._context_builder.build_result(
            context_chunks,
            max_characters=available_context_characters,
        )

        context = context_result.context

        answer = await self._generation.generate(
            query=query,
            context=context,
            language=language,
        )

        documents_by_id = {
            document.id: document for document in retrieval_result.documents
        }

        anchor_locations = {
            (hit.chunk.document_id, hit.chunk.chunk_index)
            for hit in retrieval_result.hits
        }

        rag_context_chunks: list[RagContextChunk] = []

        for chunk in context_chunks:
            document = documents_by_id.get(chunk.document_id)

            if document is None:
                raise RuntimeError("Context chunk references an unknown document")

            location = (chunk.document_id, chunk.chunk_index)

            rag_context_chunks.append(
                RagContextChunk(
                    document_id=chunk.document_id,
                    document_title=document.title,
                    chunk_index=chunk.chunk_index,
                    role="anchor" if location in anchor_locations else "neighbor",
                )
            )

        retrieval_hits: list[RagRetrievalHit] = []

        sources: list[str] = []
        seen_document_ids: set[int] = set()

        for hit in retrieval_result.hits:
            chunk = hit.chunk

            document = documents_by_id.get(chunk.document_id)

            if document is None:
                raise RuntimeError("Retrieved chunk references an unknown document")

            retrieval_hits.append(
                RagRetrievalHit(
                    document_id=chunk.document_id,
                    document_title=document.title,
                    chunk_index=chunk.chunk_index,
                    score=hit.score,
                    best_similarity=hit.best_similarity,
                    matched_query_indexes=hit.matched_query_indexes,
                )
            )

        for chunk in context_result.included_chunks:
            if chunk.document_id in seen_document_ids:
                continue

            document = documents_by_id.get(chunk.document_id)

            if document is None:
                raise RuntimeError("Context chunk references an unknown document")

            sources.append(document.title)
            seen_document_ids.add(chunk.document_id)

        return RagResult(
            answer=answer,
            sources=sources,
            retrieval_strategy=retrieval_result.strategy,
            retrieval_queries=retrieval_result.retrieval_queries,
            retrieval_hits=retrieval_hits,
            context_chunks=rag_context_chunks,
            context_chunks_count=len(context_result.included_chunks),
            context_characters=len(context),
            max_context_characters=available_context_characters,
        )
