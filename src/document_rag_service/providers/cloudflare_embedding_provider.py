"""Cloudflare Workers AI embedding provider with retry and vector normalization."""

import asyncio
import math
from typing import Literal

import httpx
from transformers import AutoTokenizer

EmbeddingInputType = Literal["documents", "queries"]


class _CloudflareInputTooLargeError(RuntimeError):
    """Signal that Cloudflare rejected a request because its input was too large."""


class CloudflareEmbeddingProvider:
    """
    Generate document and query embeddings through Cloudflare Workers AI.

    The provider owns its HTTP client, batches document requests, adaptively
    splits batches rejected as too large, applies bounded retries for transient
    failures, and validates Cloudflare responses before exposing vectors to the
    application.

    Returned embeddings are truncated to the configured dimension and
    L2-normalized so they remain compatible with the vector representation
    expected by the retrieval pipeline. A matching local tokenizer is used for
    token counting before document chunks are sent to the remote model.
    """

    def __init__(
        self,
        *,
        account_id: str,
        api_token: str,
        model_name: str,
        tokenizer_model_name: str,
        dimension: int,
        max_sequence_length: int,
        batch_size: int,
        timeout_seconds: float,
        max_attempts: int,
        retry_base_delay_seconds: float,
    ) -> None:
        """
        Initialize the Cloudflare embedding provider.

        The remote embedding model is paired with a local tokenizer so the
        ingestion pipeline can enforce sequence-length limits before sending
        content to Cloudflare.

        Args:
            account_id: Cloudflare account identifier.
            api_token: API token used to authenticate Workers AI requests.
            model_name: Cloudflare Workers AI embedding model identifier.
            tokenizer_model_name: Hugging Face model identifier used to load
                the tokenizer corresponding to the embedding model.
            dimension: Target embedding dimension after vector truncation.
            max_sequence_length: Maximum supported input length in tokenizer
                tokens.
            batch_size: Maximum number of document texts attempted per request.
            timeout_seconds: HTTP timeout applied to Cloudflare requests.
            max_attempts: Maximum number of attempts for transient failures.
            retry_base_delay_seconds: Initial delay used for exponential retry
                backoff.

        Raises:
            ValueError: If any required string configuration is empty or any
                numeric configuration is not positive.
        """

        if not account_id.strip():
            raise ValueError("Cloudflare account ID cannot be empty")

        if not api_token.strip():
            raise ValueError("Cloudflare API token cannot be empty")

        if not model_name.strip():
            raise ValueError("Cloudflare embedding model name cannot be empty")

        if not tokenizer_model_name.strip():
            raise ValueError("Tokenizer model name cannot be empty")

        if dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero")

        if max_sequence_length <= 0:
            raise ValueError("Maximum sequence length must be greater than zero")

        if batch_size <= 0:
            raise ValueError(
                "Cloudflare embedding batch size must be greater than zero"
            )

        if timeout_seconds <= 0:
            raise ValueError("Cloudflare timeout must be greater than zero")

        if max_attempts <= 0:
            raise ValueError("Cloudflare maximum attempts must be greater than zero")

        if retry_base_delay_seconds <= 0:
            raise ValueError("Cloudflare retry base delay must be greater than zero")

        self._account_id = account_id
        self._api_token = api_token
        self._model_name = model_name
        self._dimension = dimension
        self._max_sequence_length = max_sequence_length
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._retry_base_delay_seconds = retry_base_delay_seconds

        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name,
        )

        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
        )

    @property
    def model_name(self) -> str:
        """
        Return the configured Cloudflare embedding model identifier.

        Returns:
            Cloudflare Workers AI model name used for embedding requests.
        """

        return self._model_name

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension exposed to the application.

        Returns:
            Number of floating-point values in each final embedding vector.
        """

        return self._dimension

    @property
    def max_sequence_length(self) -> int:
        """
        Return the configured maximum input sequence length.

        Returns:
            Maximum number of tokenizer tokens allowed for an embedding input.
        """

        return self._max_sequence_length

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate normalized embeddings for document texts.

        The configured batch size is treated as an optimistic upper bound.
        When Cloudflare rejects a document batch with error code ``3030``
        (``input too big``), the batch is split in half and each half is
        processed recursively. The left half is always completed before the
        right half, preserving the original text-to-embedding order.

        Args:
            texts: Document texts to embed.

        Returns:
            Truncated and L2-normalized vectors in the same order as the input
            texts.
        """

        embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            batch_embeddings = await self._embed_document_batch_adaptively(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    async def _embed_document_batch_adaptively(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Embed one document batch, recursively splitting provider-oversized input.

        Cloudflare error code ``3030`` represents a deterministic input-size
        rejection, so retrying the same payload would be wasteful. Instead, the
        batch is bisected and processed depth-first. This keeps request sizes as
        large as the provider accepts while preserving input order.

        Args:
            texts: Non-empty batch of document texts.

        Returns:
            Embeddings corresponding one-to-one with ``texts`` in original order.

        Raises:
            RuntimeError: If Cloudflare rejects a single text as too large, or if
                another provider/request failure occurs.
        """

        try:
            return await self._embed(
                input_type="documents",
                texts=texts,
            )
        except _CloudflareInputTooLargeError:
            if len(texts) == 1:
                raise

            midpoint = len(texts) // 2

            left_embeddings = await self._embed_document_batch_adaptively(
                texts[:midpoint]
            )
            right_embeddings = await self._embed_document_batch_adaptively(
                texts[midpoint:]
            )

            return left_embeddings + right_embeddings

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        Generate a normalized embedding for a retrieval query.

        Query text is sent through the Cloudflare model's query-specific input
        field so retrieval queries follow the model's expected embedding mode.

        Args:
            text: Retrieval query to embed.

        Returns:
            Truncated and L2-normalized query embedding.
        """

        embeddings = await self._embed(
            input_type="queries",
            texts=[text],
        )

        return embeddings[0]

    def count_document_tokens(
        self,
        text: str,
    ) -> int:
        """
        Count document tokens using the locally loaded matching tokenizer.

        Tokenization is performed without truncation so callers can determine
        whether a chunk exceeds the configured sequence-length limit before a
        remote embedding request is made.

        Args:
            text: Document text whose tokens should be counted.

        Returns:
            Number of tokenizer tokens, including special tokens.
        """

        token_ids = self._tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )

        return len(token_ids)

    async def aclose(self) -> None:
        """
        Close the provider-owned asynchronous HTTP client.

        The provider creates and owns its Cloudflare HTTP client, so its
        lifecycle must be explicitly terminated during application shutdown.
        """

        await self._client.aclose()

    async def _embed(
        self,
        *,
        input_type: EmbeddingInputType,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Execute a Cloudflare embedding request with bounded retries.

        HTTP 408, 429, and 5xx responses commonly associated with transient
        service conditions are retried. Transport failures are also retried
        until the configured attempt limit is reached. Cloudflare error code
        ``3030`` fails immediately with an internal signal so document batches
        can be adaptively split instead of retrying the same oversized payload.
        Other non-retryable HTTP errors fail immediately.

        Successful responses are validated for the expected Cloudflare payload
        structure and embedding count before vectors are truncated and
        normalized.

        Args:
            input_type: Cloudflare request field identifying document or query
                embedding mode.
            texts: Text inputs included in the request.

        Returns:
            Validated, truncated, and L2-normalized embedding vectors.

        Raises:
            RuntimeError: If all request attempts fail, a non-retryable HTTP
                error occurs, the response is invalid JSON, Cloudflare reports
                failure, response structure is malformed, or the number of
                returned embeddings does not match the number of inputs.
        """

        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/ai/run/{self._model_name}"
        )

        response: httpx.Response | None = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self._api_token}",
                    },
                    json={
                        input_type: texts,
                    },
                )

                response.raise_for_status()
                break

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code

                if self._is_input_too_large_response(exc.response):
                    raise _CloudflareInputTooLargeError(
                        "Cloudflare embedding input is too large: "
                        f"status={status_code} "
                        f"body={exc.response.text}"
                    ) from exc

                retryable = (
                    status_code == 408
                    or status_code == 429
                    or 500 <= status_code <= 504
                )

                if not retryable or attempt == self._max_attempts:
                    raise RuntimeError(
                        "Cloudflare embedding request failed: "
                        f"status={status_code} "
                        f"body={exc.response.text}"
                    ) from exc

                delay = self._get_retry_delay(
                    response=exc.response,
                    attempt=attempt,
                )

                await asyncio.sleep(delay)

            except httpx.RequestError as exc:
                if attempt == self._max_attempts:
                    raise RuntimeError(
                        f"Cloudflare embedding request failed: {exc}"
                    ) from exc

                await asyncio.sleep(self._calculate_backoff(attempt))

        if response is None:
            raise RuntimeError(
                "Cloudflare embedding request completed without a response"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Cloudflare embedding response is not valid JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Cloudflare embedding response has invalid structure")

        if not payload.get("success"):
            raise RuntimeError(
                "Cloudflare embedding request was not successful: "
                f"{payload.get('errors')}"
            )

        result = payload.get("result")

        if not isinstance(result, dict):
            raise RuntimeError("Cloudflare embedding response has no valid result")

        data = result.get("data")

        if not isinstance(data, list):
            raise RuntimeError("Cloudflare embedding response has no valid data")

        if len(data) != len(texts):
            raise RuntimeError("Cloudflare returned an unexpected number of embeddings")

        return [self._truncate_and_normalize(vector) for vector in data]

    @staticmethod
    def _is_input_too_large_response(response: httpx.Response) -> bool:
        """
        Return whether Cloudflare rejected the request for aggregate input size.

        The Workers AI embedding endpoint reports this condition as HTTP 400
        with provider error code ``3030``. Invalid or non-JSON error bodies are
        treated as ordinary HTTP failures.

        Args:
            response: Cloudflare HTTP response associated with a failed request.

        Returns:
            ``True`` only when error code ``3030`` is present.
        """

        if response.status_code != 400:
            return False

        try:
            payload = response.json()
        except ValueError:
            return False

        if not isinstance(payload, dict):
            return False

        errors = payload.get("errors")

        if not isinstance(errors, list):
            return False

        return any(
            isinstance(error, dict) and error.get("code") == 3030
            for error in errors
        )

    def _get_retry_delay(
        self,
        *,
        response: httpx.Response,
        attempt: int,
    ) -> float:
        """
        Determine the delay before retrying a transient HTTP failure.

        A numeric ``Retry-After`` response header takes precedence when
        Cloudflare provides one. Otherwise, the provider falls back to its
        configured exponential backoff policy.

        Args:
            response: HTTP response that triggered the retry.
            attempt: Current request attempt number.

        Returns:
            Delay in seconds before the next request attempt.
        """

        retry_after = response.headers.get("Retry-After")

        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass

        return self._calculate_backoff(attempt)

    def _calculate_backoff(
        self,
        attempt: int,
    ) -> float:
        """
        Calculate exponential retry backoff for a request attempt.

        Args:
            attempt: Current request attempt number, starting at one.

        Returns:
            Backoff delay in seconds derived from the configured base delay.
        """

        return self._retry_base_delay_seconds * (2 ** (attempt - 1))

    def _truncate_and_normalize(
        self,
        vector: list[float],
    ) -> list[float]:
        """
        Truncate a raw embedding and normalize it to unit length.

        Cloudflare may return vectors larger than the dimension used by the
        application and database schema. The vector is therefore truncated to
        the configured dimension and L2-normalized before it enters the
        retrieval pipeline.

        Args:
            vector: Raw embedding vector returned by Cloudflare Workers AI.

        Returns:
            Embedding truncated to the configured dimension and normalized to
            unit L2 length.

        Raises:
            RuntimeError: If the raw vector is smaller than the configured
                dimension or the truncated vector has zero norm.
        """

        if len(vector) < self._dimension:
            raise RuntimeError(
                "Cloudflare embedding dimension is smaller than expected: "
                f"expected at least {self._dimension}, "
                f"got {len(vector)}"
            )

        truncated = vector[: self._dimension]

        norm = math.sqrt(sum(value * value for value in truncated))

        if norm == 0:
            raise RuntimeError("Cloudflare returned a zero-length embedding vector")

        return [value / norm for value in truncated]
