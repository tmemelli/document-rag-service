"""Embedding service and provider contract for document and query vectorization."""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """
    Define the embedding capabilities required by the application layer.

    Concrete providers may execute embeddings locally or delegate them to an
    external service, but must expose the same model metadata, token-counting
    behavior, vectorization operations, and asynchronous lifecycle interface.

    The protocol keeps the service layer independent from provider-specific
    implementations such as local sentence-transformer models or Cloudflare
    Workers AI.
    """

    @property
    def model_name(self) -> str:
        """
        Return the embedding model identifier.

        Returns:
            Name of the model used to generate embeddings.
        """
        ...

    @property
    def dimension(self) -> int:
        """
        Return the expected embedding vector dimension.

        Returns:
            Number of floating-point values in each embedding vector.
        """
        ...

    @property
    def max_sequence_length(self) -> int:
        """
        Return the maximum supported input sequence length.

        Returns:
            Maximum number of tokenizer tokens accepted by the provider.
        """
        ...

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for document texts.

        Args:
            texts: Document texts to embed.

        Returns:
            Embedding vectors in the same order as the input texts.
        """
        ...

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate an embedding for a retrieval query.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector representing the query.
        """
        ...

    def count_document_tokens(
        self,
        text: str,
    ) -> int:
        """
        Count document tokens using the provider-compatible tokenizer.

        Args:
            text: Document text whose tokens should be counted.

        Returns:
            Number of tokenizer tokens in the input.
        """
        ...

    async def aclose(self) -> None:
        """
        Release resources owned by the embedding provider.

        Providers without asynchronous resources may implement this operation
        as a no-op.
        """
        ...


class EmbeddingService:
    """
    Provide provider-agnostic embedding operations to the application layer.

    The service validates inputs and provider outputs while delegating the
    actual embedding and tokenization work to an ``EmbeddingProvider``.
    This prevents ingestion and retrieval workflows from depending directly on
    any specific embedding backend.

    Provider results are checked for expected vector counts, dimensions,
    sequence-length metadata, and token counts before they are exposed to the
    rest of the RAG pipeline.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        """
        Initialize the embedding service.

        Args:
            provider: Embedding backend implementing the application provider
                contract.
        """

        self._provider = provider

    @property
    def model_name(self) -> str:
        """
        Return the active embedding model identifier.

        Returns:
            Name of the model exposed by the configured provider.
        """

        return self._provider.model_name

    @property
    def dimension(self) -> int:
        """
        Return the expected embedding vector dimension.

        Returns:
            Number of values expected in every document and query embedding.
        """

        return self._provider.dimension

    @property
    def max_sequence_length(self) -> int:
        """
        Return the provider's validated maximum input sequence length.

        Returns:
            Maximum number of tokenizer tokens supported by the provider.

        Raises:
            RuntimeError: If the provider reports a non-positive sequence
                length.
        """

        max_sequence_length = self._provider.max_sequence_length

        if max_sequence_length <= 0:
            raise RuntimeError(
                "Embedding provider returned an invalid maximum sequence length"
            )

        return max_sequence_length

    async def aclose(self) -> None:
        """
        Release resources owned by the configured embedding provider.

        Resource cleanup is delegated to the provider so local and remote
        implementations can manage their own lifecycle requirements.
        """

        await self._provider.aclose()

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate and validate embeddings for document texts.

        Empty input collections are accepted and return an empty result without
        invoking the provider. Individual document texts must contain
        non-whitespace content.

        Provider output is validated to guarantee one embedding per input text
        and the vector dimension required by the configured retrieval pipeline.

        Args:
            texts: Document texts to embed.

        Returns:
            Validated embedding vectors in the same order as the input texts.

        Raises:
            ValueError: If any document text is empty or whitespace-only.
            RuntimeError: If the provider returns an unexpected number of
                embeddings or a vector with an unexpected dimension.
        """

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError("Document text cannot be empty")

        embeddings = await self._provider.embed_documents(texts)

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of embeddings"
            )

        for embedding in embeddings:
            if len(embedding) != self.dimension:
                raise RuntimeError(
                    "Embedding provider returned an unexpected embedding dimension"
                )

        return embeddings

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate and validate an embedding for a retrieval query.

        Args:
            text: Query text to embed.

        Returns:
            Query embedding with the configured provider dimension.

        Raises:
            ValueError: If the query is empty or whitespace-only.
            RuntimeError: If the provider returns a vector with an unexpected
                dimension.
        """

        if not text.strip():
            raise ValueError("Query text cannot be empty")

        embedding = await self._provider.embed_query(text)

        if len(embedding) != self.dimension:
            raise RuntimeError(
                "Embedding provider returned an unexpected embedding dimension"
            )

        return embedding

    def count_document_tokens(
        self,
        text: str,
    ) -> int:
        """
        Count and validate document tokens through the embedding provider.

        Token counting uses the tokenizer associated with the active embedding
        backend so chunk-size enforcement remains aligned with the model that
        will ultimately process the text.

        Args:
            text: Document text whose tokens should be counted.

        Returns:
            Positive number of tokenizer tokens in the document text.

        Raises:
            ValueError: If the document text is empty or whitespace-only.
            RuntimeError: If the provider reports a non-positive token count.
        """

        if not text.strip():
            raise ValueError("Document text cannot be empty")

        token_count = self._provider.count_document_tokens(text)

        if token_count <= 0:
            raise RuntimeError("Embedding provider returned an invalid token count")

        return token_count
