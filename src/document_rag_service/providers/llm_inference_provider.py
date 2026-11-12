"""Provider for the external LLM inference service."""

import httpx


class LLMInferenceProvider:
    """
    Generate text through the configured external LLM inference service.

    The provider translates application-level generation requests into HTTP
    calls understood by the inference service. It uses an injected asynchronous
    HTTP client so connection pooling, base URL configuration, timeouts, and
    client lifecycle can be shared and managed by the application lifespan.
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        provider: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 500,
    ) -> None:
        """
        Initialize the LLM inference provider.

        Args:
            client: Shared asynchronous HTTP client configured for the inference
                service.
            provider: Inference provider identifier forwarded to the external
                service.
            model: Model identifier used for text generation.
            temperature: Sampling temperature used by the inference request.
            max_tokens: Maximum number of output tokens requested from the
                model.
        """

        self._client = client
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(
        self,
        *,
        system_prompt: str,
        prompt: str,
    ) -> str:
        """
        Generate text through the configured inference provider and model.

        The request is sent to the shared inference service using the provider,
        model, sampling configuration, system instructions, and user prompt
        configured for this provider instance.

        Successful responses must contain a non-empty string in the
        ``content`` field. Transport failures, unsuccessful HTTP responses,
        malformed JSON, and structurally invalid responses are converted into
        runtime errors for the calling service.

        Args:
            system_prompt: System-level instructions supplied to the language
                model.
            prompt: Application prompt containing the generation request.

        Returns:
            Generated text with surrounding whitespace removed.

        Raises:
            RuntimeError: If the inference service cannot be reached, returns
                an unsuccessful HTTP response, returns invalid JSON, omits a
                valid ``content`` field, or produces empty content.
        """

        payload = {
            "provider": self._provider,
            "model": self._model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        try:
            response = await self._client.post(
                "v1/inference",
                json=payload,
            )
        except httpx.RequestError as exc:
            raise RuntimeError("LLM inference service request failed") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                "LLM inference service request failed: "
                f"status={response.status_code} "
                f"body={response.text}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("LLM inference service returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise RuntimeError("LLM inference service returned an invalid response")

        content = data.get("content")

        if not isinstance(content, str):
            raise RuntimeError("LLM inference service returned an invalid response")

        if not content.strip():
            raise RuntimeError("LLM inference service returned empty content")

        return content.strip()
