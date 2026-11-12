"""
Context expansion with relevance-prioritized contiguous document regions.
"""

from dataclasses import dataclass

from document_rag_service.models.document_chunk import DocumentChunk
from document_rag_service.repositories.document_chunk_repository import (
    DocumentChunkRepository,
)


@dataclass(frozen=True)
class _ContextRegion:
    """
    Represent one contiguous document region selected for context expansion.

    Attributes:
        document_id: Identifier of the document that owns the region.
        start_index: First chunk index included in the region.
        end_index: Last chunk index included in the region.
        priority: Semantic priority inherited from the highest-ranked
            retrieval anchor contained in the region.
    """

    document_id: int
    start_index: int
    end_index: int
    priority: int


class ContextExpansionService:
    """
    Expand retrieved anchor chunks with neighboring document chunks.

    Retrieval anchors arrive ordered by semantic relevance. Their local
    neighborhoods are merged when they overlap, while each merged region
    inherits the priority of its highest-ranked anchor.

    Regions are then returned in semantic-priority order, with chunks inside
    each region preserving their natural document order.
    """

    def __init__(
        self,
        chunk_repository: DocumentChunkRepository,
    ) -> None:
        """
        Initialize the context expansion service.

        Args:
            chunk_repository: Repository used to retrieve neighboring chunks.
        """

        self._chunks = chunk_repository

    async def expand(
        self,
        *,
        user_id: int,
        retrieved_chunks: list[DocumentChunk],
        neighbor_window: int = 1,
    ) -> list[DocumentChunk]:
        """
        Expand retrieved chunks into semantically prioritized context regions.

        Each retrieved chunk acts as an anchor. A window of neighboring chunks
        is created around every anchor. Overlapping windows from the same
        document are merged into one contiguous region.

        The input order of ``retrieved_chunks`` represents semantic retrieval
        priority. Each merged region therefore inherits the lowest positional
        priority value among the anchors it contains.

        Regions are returned in semantic-priority order. Chunks within each
        region remain ordered by ``chunk_index`` so the language model receives
        locally coherent document passages.

        Args:
            user_id: Identifier of the user who owns the documents.
            retrieved_chunks: Retrieval anchors ordered from most relevant to
                least relevant.
            neighbor_window: Number of adjacent chunks to include before and
                after each anchor.

        Returns:
            Expanded document chunks ordered first by semantic region priority
            and then by natural chunk order inside each region.

        Raises:
            ValueError: If no retrieved chunks are provided or if
                ``neighbor_window`` is negative.
        """

        if not retrieved_chunks:
            raise ValueError("Retrieved chunks cannot be empty")

        if neighbor_window < 0:
            raise ValueError("Neighbor window cannot be negative")

        regions_by_document: dict[int, list[_ContextRegion]] = {}

        for priority, chunk in enumerate(retrieved_chunks):
            region = _ContextRegion(
                document_id=chunk.document_id,
                start_index=max(
                    0,
                    chunk.chunk_index - neighbor_window,
                ),
                end_index=chunk.chunk_index + neighbor_window,
                priority=priority,
            )

            regions_by_document.setdefault(
                chunk.document_id,
                [],
            ).append(region)

        merged_regions: list[_ContextRegion] = []

        for document_regions in regions_by_document.values():
            ordered_regions = sorted(
                document_regions,
                key=lambda region: (
                    region.start_index,
                    region.end_index,
                ),
            )

            current_region = ordered_regions[0]

            for region in ordered_regions[1:]:
                if region.start_index <= current_region.end_index:
                    current_region = _ContextRegion(
                        document_id=current_region.document_id,
                        start_index=current_region.start_index,
                        end_index=max(
                            current_region.end_index,
                            region.end_index,
                        ),
                        priority=min(
                            current_region.priority,
                            region.priority,
                        ),
                    )
                    continue

                merged_regions.append(current_region)
                current_region = region

            merged_regions.append(current_region)

        merged_regions.sort(
            key=lambda region: region.priority,
        )

        requested_locations: list[tuple[int, int]] = []

        for region in merged_regions:
            for chunk_index in range(
                region.start_index,
                region.end_index + 1,
            ):
                requested_locations.append(
                    (
                        region.document_id,
                        chunk_index,
                    )
                )

        chunks = await self._chunks.list_by_locations(
            user_id=user_id,
            locations=requested_locations,
        )

        chunks_by_location = {
            (
                chunk.document_id,
                chunk.chunk_index,
            ): chunk
            for chunk in chunks
        }

        return [
            chunks_by_location[location]
            for location in requested_locations
            if location in chunks_by_location
        ]
