"""Document chunking with character overlap and embedding token-limit safety."""

from collections.abc import Callable


class ChunkingService:
    """
    Split document text into overlapping chunks suitable for embedding.

    The service first creates character-bounded chunks while preserving whole
    words and configurable overlap between adjacent chunks. A second optional
    stage enforces the actual token limit of the embedding model so oversized
    chunks are subdivided before embeddings are generated.
    """

    def __init__(
        self,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """
        Initialize the chunking service.

        Args:
            chunk_size: Maximum approximate chunk size in characters.
            chunk_overlap: Maximum character overlap between adjacent chunks.

        Raises:
            ValueError: If ``chunk_size`` is not positive, ``chunk_overlap`` is
                negative, or the overlap is greater than or equal to the chunk
                size.
        """

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(
        self,
        text: str,
    ) -> list[str]:
        """
        Split document text into overlapping character-bounded chunks.

        Text is grouped by whole words so normal chunk boundaries do not split
        words in the middle. Adjacent chunks reuse as many trailing words as
        possible without exceeding the configured character overlap.

        Whitespace is normalized because the input is tokenized with
        ``str.split()`` and reconstructed with single spaces.

        Args:
            text: Document text to split.

        Returns:
            Ordered chunks preserving the original word sequence and configured
            overlap. An empty list is returned when the input contains no
            non-whitespace text.
        """

        words = text.split()

        if not words:
            return []

        chunks: list[str] = []
        start = 0

        while start < len(words):
            current_words: list[str] = []
            current_length = 0
            end = start

            while end < len(words):
                word = words[end]

                additional_length = len(word)

                if current_words:
                    additional_length += 1

                if (
                    current_words
                    and current_length + additional_length > self._chunk_size
                ):
                    break

                current_words.append(word)
                current_length += additional_length
                end += 1

            chunks.append(" ".join(current_words))

            if end == len(words):
                break

            overlap_start = end
            overlap_length = 0

            while overlap_start > start:
                word = words[overlap_start - 1]

                additional_length = len(word)

                if overlap_length:
                    additional_length += 1

                if overlap_length + additional_length > self._chunk_overlap:
                    break

                overlap_start -= 1
                overlap_length += additional_length

            if overlap_start == start:
                start = end
            else:
                start = overlap_start

        return chunks

    def split_with_token_limit(
        self,
        text: str,
        *,
        count_tokens: Callable[[str], int],
        max_tokens: int,
    ) -> list[tuple[str, int]]:
        """
        Split document text while enforcing the embedding model token limit.

        Character-based chunks are created first. Each resulting chunk is then
        measured with the embedding provider's tokenizer. Chunks that exceed
        ``max_tokens`` are subdivided until every final fragment can be safely
        sent to the embedding model.

        Args:
            text: Document text to split.
            count_tokens: Callable that returns the token count for a text
                fragment using the embedding model's tokenizer.
            max_tokens: Maximum number of tokens allowed in each final chunk.

        Returns:
            Ordered pairs containing each token-safe chunk and its measured
            token count.

        Raises:
            ValueError: If ``max_tokens`` is not positive or an oversized
                fragment cannot be reduced to a valid token-safe chunk.
            RuntimeError: If token-aware subdivision produces an invalid
                oversized result or fails to make forward progress.
        """

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        chunks = self.split(text)
        chunks_with_token_count: list[tuple[str, int]] = []

        for chunk in chunks:
            token_count = count_tokens(chunk)

            if token_count <= max_tokens:
                chunks_with_token_count.append(
                    (
                        chunk,
                        token_count,
                    )
                )
                continue

            chunks_with_token_count.extend(
                self._split_oversized_chunk(
                    chunk,
                    count_tokens=count_tokens,
                    max_tokens=max_tokens,
                )
            )

        return chunks_with_token_count

    def _split_oversized_chunk(
        self,
        chunk: str,
        *,
        count_tokens: Callable[[str], int],
        max_tokens: int,
    ) -> list[tuple[str, int]]:
        """
        Subdivide one oversized chunk until every fragment fits the token limit.

        A binary search finds the longest prefix that satisfies the tokenizer
        limit. When possible, the split point is moved back to a word boundary.
        Subsequent fragments reuse text according to the configured character
        overlap while guaranteeing that each iteration advances through the
        source text.

        Args:
            chunk: Chunk whose token count exceeds the embedding model limit.
            count_tokens: Callable that returns the token count for a text
                fragment.
            max_tokens: Maximum number of tokens allowed in each final chunk.

        Returns:
            Ordered token-safe fragments paired with their measured token
            counts.

        Raises:
            ValueError: If no non-empty fragment can be produced within the
                configured token limit.
            RuntimeError: If a generated fragment still exceeds the token
                limit or the subdivision algorithm fails to make progress.
        """

        remaining = chunk.strip()
        safe_chunks: list[tuple[str, int]] = []

        while remaining:
            token_count = count_tokens(remaining)

            if token_count <= max_tokens:
                safe_chunks.append(
                    (
                        remaining,
                        token_count,
                    )
                )
                break

            low = 1
            high = len(remaining)
            best_end = 0

            while low <= high:
                middle = (low + high) // 2
                candidate = remaining[:middle].strip()

                if not candidate:
                    low = middle + 1
                    continue

                candidate_token_count = count_tokens(candidate)

                if candidate_token_count <= max_tokens:
                    best_end = middle
                    low = middle + 1
                else:
                    high = middle - 1

            if best_end == 0:
                raise ValueError(
                    "Unable to split document text within the "
                    "embedding model token limit"
                )

            split_end = best_end

            word_boundary = remaining.rfind(
                " ",
                0,
                split_end,
            )

            if word_boundary > 0:
                split_end = word_boundary

            safe_chunk = remaining[:split_end].strip()

            if not safe_chunk:
                raise ValueError("Unable to create a valid document chunk")

            safe_token_count = count_tokens(safe_chunk)

            if safe_token_count > max_tokens:
                raise RuntimeError("Token-aware chunking produced an oversized chunk")

            safe_chunks.append(
                (
                    safe_chunk,
                    safe_token_count,
                )
            )

            overlap_start = max(
                0,
                split_end - self._chunk_overlap,
            )

            if overlap_start == 0:
                overlap_start = split_end
            else:
                next_space = remaining.find(
                    " ",
                    overlap_start,
                    split_end,
                )

                if next_space == -1:
                    overlap_start = split_end
                else:
                    overlap_start = next_space + 1

            next_remaining = remaining[overlap_start:].strip()

            if next_remaining == remaining:
                raise RuntimeError("Token-aware chunking did not make progress")

            remaining = next_remaining

        return safe_chunks
