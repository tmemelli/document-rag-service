"""Local sentence-transformer embedding provider."""

import asyncio

from sentence_transformers import SentenceTransformer


class LocalEmbeddingProvider:
    """
    Generate document and query embeddings with a locally loaded model.

    The provider wraps a SentenceTransformer instance and exposes the embedding
    operations required by the RAG pipeline. Model execution is offloaded to a
    worker thread so CPU- or GPU-bound encoding does not block the application's
    asynchronous event loop.

    Document embeddings are generated without a retrieval prompt, while query
    embeddings use the model's ``query`` prompt when available. All embeddings
    are L2-normalized before being returned.
    """

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        dimension: int,
    ) -> None:
        """
        Initialize the local embedding model.

        The configured embedding dimension is passed to SentenceTransformer as
        ``truncate_dim`` and then verified against the dimension exposed by the
        loaded model. The provider also records the model tokenizer's maximum
        sequence length for downstream token-aware chunking.

        Args:
            model_name: Hugging Face model identifier used to load the
                SentenceTransformer model.
            device: Execution device passed to SentenceTransformer, such as
                ``cpu`` or ``cuda``.
            dimension: Expected output embedding dimension.

        Raises:
            ValueError: If the model name or device is empty, or ``dimension``
                is not positive.
            RuntimeError: If the loaded model does not expose a valid embedding
                dimension or maximum sequence length, or if its effective
                embedding dimension differs from the configured value.
        """

        if not model_name.strip():
            raise ValueError("Embedding model name cannot be empty")

        if not device.strip():
            raise ValueError("Embedding device cannot be empty")

        if dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero")

        self._model_name = model_name

        self._model = SentenceTransformer(
            model_name,
            device=device,
            truncate_dim=dimension,
        )

        model_dimension = self._model.get_embedding_dimension()

        if model_dimension is None:
            raise RuntimeError("Could not determine embedding dimension")

        if model_dimension != dimension:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected {dimension}, "
                f"got {model_dimension}"
            )

        self._dimension = model_dimension

        max_sequence_length = self._model.max_seq_length

        if max_sequence_length is None:
            raise RuntimeError("Could not determine maximum sequence length")

        if max_sequence_length <= 0:
            raise RuntimeError(
                "Embedding model returned an invalid maximum sequence length"
            )

        self._max_sequence_length = max_sequence_length

    @property
    def model_name(self) -> str:
        """
        Return the configured embedding model identifier.

        Returns:
            Hugging Face model name used by the provider.
        """

        return self._model_name

    @property
    def dimension(self) -> int:
        """
        Return the effective embedding vector dimension.

        Returns:
            Number of floating-point values produced for each embedding.
        """

        return self._dimension

    @property
    def max_sequence_length(self) -> int:
        """
        Return the maximum tokenizer sequence length supported by the model.

        Returns:
            Maximum number of input tokens accepted by the embedding model.
        """

        return self._max_sequence_length

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate normalized embeddings for document texts.

        Encoding is executed in a worker thread to avoid blocking the asyncio
        event loop during local model inference.

        Args:
            texts: Document texts to embed.

        Returns:
            L2-normalized embedding vectors in the same order as the supplied
            document texts.
        """

        embeddings = await asyncio.to_thread(
            self._model.encode,
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate a normalized embedding for a retrieval query.

        The model's ``query`` prompt is used so query embeddings follow the
        retrieval behavior expected by the configured SentenceTransformer
        model. Encoding runs in a worker thread to avoid blocking the asyncio
        event loop.

        Args:
            text: Query text to embed.

        Returns:
            L2-normalized query embedding.
        """

        embedding = await asyncio.to_thread(
            self._model.encode,
            text,
            prompt_name="query",
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def count_document_tokens(
        self,
        text: str,
    ) -> int:
        """
        Count model-tokenizer tokens for document text.

        Token counting uses the same tokenizer associated with the local
        embedding model and disables truncation so callers can detect content
        that exceeds the model's sequence-length limit before embedding.

        Args:
            text: Document text whose tokens should be counted.

        Returns:
            Number of tokenizer tokens, including special tokens.

        Raises:
            RuntimeError: If the loaded embedding model does not expose a
                tokenizer.
        """

        tokenizer = self._model.tokenizer

        if tokenizer is None:
            raise RuntimeError("Embedding model tokenizer is not available")

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )

        return len(token_ids)

    async def aclose(self) -> None:
        """
        Release provider-owned asynchronous resources.

        The local SentenceTransformer provider currently owns no asynchronous
        resources, so shutdown requires no explicit cleanup.
        """

        return None
