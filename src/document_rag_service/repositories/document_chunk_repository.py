"""Repository for document chunk persistence and vector-similarity retrieval."""

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from document_rag_service.models.document import Document
from document_rag_service.models.document_chunk import DocumentChunk


class DocumentChunkRepository:
    """
    Provide persistence and retrieval operations for document chunks.

    The repository encapsulates SQLAlchemy and pgvector queries used to store
    chunk content, retrieve document neighborhoods, and perform semantic
    similarity searches.

    User-scoped retrieval operations join against the parent document so chunk
    access remains restricted to documents owned by the specified user.
    Transaction boundaries remain the responsibility of the calling service.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the document chunk repository.

        Args:
            session: Asynchronous SQLAlchemy session used for persistence and
                retrieval operations.
        """

        self._session = session

    async def create(
        self,
        *,
        document_id: int,
        chunk_index: int,
        content: str,
        embedding: list[float],
        token_count: int = 0,
    ) -> DocumentChunk:
        """
        Create and persist a document chunk.

        The entity is flushed and refreshed so database-generated values are
        available to the caller without committing the surrounding transaction.

        Args:
            document_id: Identifier of the parent document.
            chunk_index: Zero-based position of the chunk within the document.
            content: Text content stored for the chunk.
            embedding: Vector representation used for semantic retrieval.
            token_count: Number of embedding-model tokens contained in the
                chunk.

        Returns:
            Newly created and refreshed document chunk entity.
        """

        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            embedding=embedding,
        )

        self._session.add(chunk)
        await self._session.flush()
        await self._session.refresh(chunk)

        return chunk

    async def list_by_document_id(
        self,
        document_id: int,
    ) -> list[DocumentChunk]:
        """
        Return all chunks belonging to a document in natural reading order.

        Args:
            document_id: Identifier of the document whose chunks should be
                retrieved.

        Returns:
            Document chunks ordered by ascending ``chunk_index``.
        """

        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )

        result = await self._session.scalars(statement)

        return list(result.all())

    async def list_by_locations(
        self,
        *,
        user_id: int,
        locations: list[tuple[int, int]],
    ) -> list[DocumentChunk]:
        """
        Return user-owned chunks matching document and chunk locations.

        Each location is represented by a ``(document_id, chunk_index)`` pair.
        The query joins the parent document to enforce ownership before
        returning matching chunks.

        No database ordering is imposed because callers such as the context
        expansion service reconstruct the required semantic and reading order
        explicitly after retrieval.

        Args:
            user_id: Identifier of the user who must own the parent documents.
            locations: Document and chunk-index pairs to retrieve.

        Returns:
            Matching chunks owned by the specified user. An empty list is
            returned when no locations are supplied.
        """

        if not locations:
            return []

        statement = (
            select(DocumentChunk)
            .join(
                Document,
                Document.id == DocumentChunk.document_id,
            )
            .where(
                Document.user_id == user_id,
                tuple_(
                    DocumentChunk.document_id,
                    DocumentChunk.chunk_index,
                ).in_(locations),
            )
        )

        result = await self._session.scalars(statement)

        return list(result.all())

    async def search_similar(
        self,
        *,
        user_id: int,
        document_ids: list[int],
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Retrieve the nearest user-owned chunks by cosine distance.

        Semantic search is restricted to the selected documents and their
        owning user before results are ranked. Lower cosine-distance values
        indicate greater vector similarity.

        Args:
            user_id: Identifier of the user who must own the selected
                documents.
            document_ids: Document identifiers included in the search scope.
            query_embedding: Vector representation of the retrieval query.
            limit: Maximum number of nearest chunks to return.

        Returns:
            Pairs containing each retrieved chunk and its cosine-distance value,
            ordered from smallest to largest distance. An empty list is
            returned when no document identifiers are supplied.
        """

        if not document_ids:
            return []

        distance = DocumentChunk.embedding.cosine_distance(query_embedding)

        statement = (
            select(
                DocumentChunk,
                distance.label("distance"),
            )
            .join(
                Document,
                Document.id == DocumentChunk.document_id,
            )
            .where(
                Document.user_id == user_id,
                Document.id.in_(document_ids),
            )
            .order_by(distance)
            .limit(limit)
        )

        result = await self._session.execute(statement)

        return [
            (
                chunk,
                float(distance_value),
            )
            for chunk, distance_value in result.all()
        ]
