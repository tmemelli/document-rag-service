"""
Grounded LLM answer generation with strict prompt-size enforcement.
"""

from typing import Protocol

SYSTEM_PROMPT = """
Answer the user's question using only the provided context.

Do not use external knowledge or make claims that are not supported
by the provided context.

When the context explicitly describes how an event happened, preserve
the concrete action, mechanism, cause, and sequence described in the source.

Do not replace a specific action with a more general, inferred, softened,
or alternative action.

Do not invent missing mechanisms or causal details.

If multiple context passages must be combined to answer the question,
combine them faithfully and preserve their factual relationship.

When paraphrasing, preserve the exact factual meaning of the source.

If the question asks how something happened, state the concrete action
described by the context.

If the context supports a specific answer, prefer that specific answer
over a broader or more generic description.

If the provided context is not sufficient to answer the question,
clearly state that there is not enough information in the selected
documents.

Always answer in the requested language, regardless of the language
used in the user's question or in the provided context.

The context may contain internal source labels such as [SOURCE 1],
[SOURCE 2], and so on. These labels exist only to separate retrieved
context internally.

Do not expose internal source labels in the final answer.

Do not mention chunks, retrieval, embeddings, source identifiers,
context construction, or any other internal RAG mechanism.

Answer naturally, clearly, and directly, as if responding to the
user based on the available documents.

Do not use Markdown or any other presentation-specific formatting.
Return plain natural text only.
""".strip()


class LLMProvider(Protocol):
    """
    Define the generation contract required by GenerationService.
    """

    async def generate(
        self,
        *,
        system_prompt: str,
        prompt: str,
    ) -> str:
        """
        Generate text from system and user prompts.

        Args:
            system_prompt: Instructions controlling generation behavior.
            prompt: User prompt containing the question and retrieved context.

        Returns:
            Generated natural-language response.
        """

        ...


class GenerationService:
    """
    Generate grounded answers from retrieved document context.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_prompt_characters: int,
    ) -> None:
        """
        Initialize the generation service.

        Args:
            provider: Provider used to execute LLM generation.
            max_prompt_characters: Maximum prompt size accepted by the
                upstream inference service.

        Raises:
            ValueError: If the configured prompt limit is invalid.
        """

        if max_prompt_characters <= 0:
            raise ValueError("max_prompt_characters must be greater than zero")

        self._provider = provider
        self._max_prompt_characters = max_prompt_characters

    async def generate(
        self,
        *,
        query: str,
        context: str,
        language: str,
    ) -> str:
        """
        Generate a grounded answer from a question and document context.

        Args:
            query: Original user question.
            context: Retrieved document context supplied to the LLM.
            language: Language requested for the final response.

        Returns:
            Generated answer stripped of surrounding whitespace.

        Raises:
            ValueError: If required input is empty or the resulting prompt
                exceeds the configured maximum size.
            RuntimeError: If the provider returns an empty response.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not context.strip():
            raise ValueError("Context cannot be empty")

        if not language.strip():
            raise ValueError("Response language cannot be empty")

        prompt = self._build_prompt(
            query=query,
            context=context,
            language=language,
        )

        if len(prompt) > self._max_prompt_characters:
            raise ValueError(
                "Generated prompt exceeds the maximum allowed size: "
                f"{len(prompt)} > {self._max_prompt_characters}"
            )

        answer = await self._provider.generate(
            system_prompt=SYSTEM_PROMPT,
            prompt=prompt,
        )

        if not answer.strip():
            raise RuntimeError("LLM provider returned an empty response")

        return answer.strip()

    def get_available_context_characters(
        self,
        *,
        query: str,
        language: str,
    ) -> int:
        """
        Calculate how many characters remain available for retrieved context.

        The calculation reserves space for the language, question, prompt
        labels, separators, and the newline preceding a non-empty context.

        Args:
            query: Original user question.
            language: Language requested for the final response.

        Returns:
            Maximum number of characters available for document context.

        Raises:
            ValueError: If required input is empty or prompt metadata alone
                exceeds the configured prompt limit.
        """

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not language.strip():
            raise ValueError("Response language cannot be empty")

        prompt_without_context = self._build_prompt(
            query=query,
            context="",
            language=language,
        )

        prompt_overhead = len(prompt_without_context) + 1

        available_characters = self._max_prompt_characters - prompt_overhead

        if available_characters <= 0:
            raise ValueError("Query and prompt metadata exceed the maximum prompt size")

        return available_characters

    def _build_prompt(
        self,
        *,
        query: str,
        context: str,
        language: str,
    ) -> str:
        """
        Build the user prompt sent to the inference provider.

        Args:
            query: Original user question.
            context: Retrieved document context.
            language: Language requested for the final response.

        Returns:
            Fully assembled generation prompt.
        """

        return f"""
Requested language:
{language.strip()}

Question:
{query.strip()}

Context:
{context.strip()}
""".strip()
