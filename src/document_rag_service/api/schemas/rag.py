"""RAG request, response, and observability schemas exposed by the HTTP API."""

from pydantic import BaseModel, Field


class RagQueryRequest(BaseModel):
    """
    Validate an authenticated retrieval-augmented generation request.

    Attributes:
        document_ids: User-owned document identifiers defining the retrieval
            scope. At least one and at most one hundred documents may be
            selected per request.
        query: User question submitted to the RAG pipeline.
        language: Requested language for answer generation.
        limit: Maximum number of retrieval anchors returned by semantic search.
        neighbor_window: Number of adjacent chunks included on each side of
            every retrieved anchor during context expansion.
        transform_query: Whether query transformation should generate
            additional focused retrieval queries.
    """

    document_ids: list[int] = Field(
        min_length=1,
        max_length=100,
    )
    query: str = Field(
        min_length=1,
        max_length=4000,
    )
    language: str = Field(
        default="pt-BR",
        min_length=2,
        max_length=32,
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
    )
    neighbor_window: int = Field(
        default=1,
        ge=0,
        le=5,
    )
    transform_query: bool = False


class RetrievalHitResponse(BaseModel):
    """
    Represent observability metadata for one retrieval anchor.

    Attributes:
        document_id: Identifier of the document containing the retrieved chunk.
        document_title: Human-readable title of the source document.
        chunk_index: Zero-based position of the retrieved chunk.
        score: Final retrieval ranking score.
        best_similarity: Highest dense similarity observed for the chunk across
            the executed retrieval queries.
        matched_query_indexes: One-based indexes of retrieval queries in which
            the chunk appeared.
    """

    document_id: int
    document_title: str
    chunk_index: int
    score: float
    best_similarity: float
    matched_query_indexes: list[int]


class ContextChunkResponse(BaseModel):
    """
    Represent observability metadata for one expanded context candidate.

    Context chunks include directly retrieved anchors and neighboring chunks
    added during context expansion. The final prompt-character budget may cause
    only a subset of these candidates to be sent to the language model.

    Attributes:
        document_id: Identifier of the document containing the chunk.
        document_title: Human-readable title of the source document.
        chunk_index: Zero-based position of the chunk within its document.
        role: Context role identifying the chunk as an ``anchor`` or
            ``neighbor``.
    """

    document_id: int
    document_title: str
    chunk_index: int
    role: str


class RagQueryResponse(BaseModel):
    """
    Represent the complete result returned by the authenticated RAG pipeline.

    Retrieval diagnostics describe the semantic-search phase, while
    ``context_chunks`` preserves the complete expanded candidate set for
    observability. ``context_chunks_count`` and the character-budget fields
    describe the subset that actually fit within the final generation context.

    Attributes:
        answer: Grounded answer generated from the assembled context.
        sources: Unique document titles represented in the context actually
            supplied to the language model.
        retrieval_strategy: Ranking strategy used by retrieval.
        retrieval_queries: Queries executed against the vector index.
        retrieval_hits: Ranked chunks selected directly by retrieval.
        context_chunks: Expanded anchor and neighboring chunk candidates before
            final prompt-budget truncation.
        context_chunks_count: Number of expanded chunks actually included in
            the final LLM context.
        context_characters: Character length of the final assembled context.
        max_context_characters: Maximum character budget available for context.
    """

    answer: str
    sources: list[str]

    retrieval_strategy: str
    retrieval_queries: list[str]
    retrieval_hits: list[RetrievalHitResponse]

    context_chunks: list[ContextChunkResponse]
    context_chunks_count: int
    context_characters: int
    max_context_characters: int
