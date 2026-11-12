"""Build bounded LLM context from retrieved document chunks."""

from dataclasses import dataclass

from document_rag_service.models.document_chunk import DocumentChunk


@dataclass(frozen=True)
class ContextBuildResult:
    """
    Represent the result of building context within a character budget.

    Attributes:
        context: Formatted context sent to the language model.
        included_chunks: Document chunks that were fully included in the
            generated context.
    """

    context: str
    included_chunks: tuple[DocumentChunk, ...]


class ContextBuilderService:
    """
    Build structured LLM context from retrieved document chunks.

    The service formats chunks with stable source metadata and optionally
    enforces a maximum character budget. Chunks are included only when they
    fit completely, preventing partial source content from being sent to the
    language model.
    """

    def build(
        self,
        context_chunks: list[DocumentChunk],
        *,
        max_characters: int | None = None,
    ) -> str:
        """
        Build and return the formatted LLM context.

        Args:
            context_chunks: Document chunks ordered for context construction.
            max_characters: Optional maximum number of characters allowed in
                the final context.

        Returns:
            Formatted context containing the chunks that fit within the
            configured character budget.

        Raises:
            ValueError: If no chunks are provided, the character budget is
                invalid, or no complete chunk fits within the budget.
        """

        result = self.build_result(
            context_chunks,
            max_characters=max_characters,
        )

        return result.context

    def build_result(
        self,
        context_chunks: list[DocumentChunk],
        *,
        max_characters: int | None = None,
    ) -> ContextBuildResult:
        """
        Build context and report exactly which chunks were included.

        Each included chunk is formatted with an internal source label,
        document identifier, chunk index, and normalized content. When a
        character limit is configured, processing stops before the first chunk
        that would exceed the budget so no chunk is truncated.

        Args:
            context_chunks: Document chunks ordered for context construction.
            max_characters: Optional maximum number of characters allowed in
                the final context.

        Returns:
            ContextBuildResult containing the formatted context and the exact
            sequence of chunks included in it.

        Raises:
            ValueError: If no chunks are provided, ``max_characters`` is not
                positive, or the first complete chunk exceeds the configured
                budget.
        """

        if not context_chunks:
            raise ValueError("Context chunks cannot be empty")

        if max_characters is not None and max_characters <= 0:
            raise ValueError("Maximum context characters must be greater than zero")

        context_parts: list[str] = []
        included_chunks: list[DocumentChunk] = []
        current_length = 0

        for chunk in context_chunks:
            position = len(context_parts) + 1

            context_part = "\n".join(
                [
                    f"[SOURCE {position}]",
                    f"document_id={chunk.document_id}",
                    f"chunk_index={chunk.chunk_index}",
                    chunk.content.strip(),
                ]
            )

            separator_length = 2 if context_parts else 0

            candidate_length = current_length + separator_length + len(context_part)

            if max_characters is not None and candidate_length > max_characters:
                break

            context_parts.append(context_part)
            included_chunks.append(chunk)
            current_length = candidate_length

        if not context_parts:
            raise ValueError("No complete context chunk fits within the context budget")

        return ContextBuildResult(
            context="\n\n".join(context_parts),
            included_chunks=tuple(included_chunks),
        )
