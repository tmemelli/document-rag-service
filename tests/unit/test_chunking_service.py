"""
Tests for character- and token-aware document chunking.
"""

import pytest

from document_rag_service.services.chunking_service import ChunkingService


def count_characters(text: str) -> int:
    """Use character count as a deterministic stand-in for token count."""
    return len(text)


def count_words(text: str) -> int:
    """Use whitespace-separated word count as a deterministic tokenizer."""
    return len(text.split())


@pytest.mark.parametrize("chunk_size", [0, -1, -100])
def test_chunking_service_rejects_non_positive_chunk_size(
    chunk_size: int,
) -> None:
    """Reject zero or negative character chunk sizes."""
    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than zero",
    ):
        ChunkingService(
            chunk_size=chunk_size,
            chunk_overlap=0,
        )


@pytest.mark.parametrize("chunk_overlap", [-1, -100])
def test_chunking_service_rejects_negative_chunk_overlap(
    chunk_overlap: int,
) -> None:
    """Reject negative character overlap values."""
    with pytest.raises(
        ValueError,
        match="chunk_overlap cannot be negative",
    ):
        ChunkingService(
            chunk_size=100,
            chunk_overlap=chunk_overlap,
        )


@pytest.mark.parametrize("chunk_overlap", [10, 11, 100])
def test_chunking_service_rejects_overlap_not_smaller_than_chunk_size(
    chunk_overlap: int,
) -> None:
    """Require overlap to remain strictly smaller than chunk size."""
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        ChunkingService(
            chunk_size=10,
            chunk_overlap=chunk_overlap,
        )


def test_chunking_service_accepts_zero_overlap() -> None:
    """Accept zero overlap as a valid chunking configuration."""
    service = ChunkingService(
        chunk_size=10,
        chunk_overlap=0,
    )

    assert service.split("one two three") == ["one two", "three"]


def test_split_returns_empty_list_for_empty_text() -> None:
    """Return no chunks when the input text is empty."""
    service = ChunkingService(
        chunk_size=20,
        chunk_overlap=5,
    )

    assert service.split("") == []


@pytest.mark.parametrize(
    "text",
    [
        " ",
        "   ",
        "\t",
        "\n",
        " \t\n ",
    ],
)
def test_split_returns_empty_list_for_whitespace_only_text(
    text: str,
) -> None:
    """Return no chunks when the input contains only whitespace."""
    service = ChunkingService(
        chunk_size=20,
        chunk_overlap=5,
    )

    assert service.split(text) == []


def test_split_returns_single_chunk_when_text_fits_chunk_size() -> None:
    """Keep text in one chunk when it fits within the character limit."""
    service = ChunkingService(
        chunk_size=20,
        chunk_overlap=5,
    )

    assert service.split("alpha beta gamma") == ["alpha beta gamma"]


def test_split_normalizes_whitespace() -> None:
    """Normalize input whitespace to single spaces in produced chunks."""
    service = ChunkingService(
        chunk_size=50,
        chunk_overlap=0,
    )

    chunks = service.split("  alpha\n\tbeta   gamma  ")

    assert chunks == ["alpha beta gamma"]


def test_split_accepts_chunk_exactly_at_character_limit() -> None:
    """Keep a chunk whose reconstructed length exactly matches the limit."""
    service = ChunkingService(
        chunk_size=10,
        chunk_overlap=0,
    )

    assert service.split("alpha beta gamma") == [
        "alpha beta",
        "gamma",
    ]
    assert len("alpha beta") == 10


def test_split_creates_multiple_chunks_without_splitting_words() -> None:
    """Split at word boundaries while preserving the original word order."""
    service = ChunkingService(
        chunk_size=10,
        chunk_overlap=0,
    )

    chunks = service.split("one two three four")

    assert chunks == [
        "one two",
        "three four",
    ]


def test_split_keeps_single_word_larger_than_chunk_size_intact() -> None:
    """Keep an oversized word intact instead of splitting it by characters."""
    service = ChunkingService(
        chunk_size=5,
        chunk_overlap=0,
    )

    chunks = service.split("abcdefghij x y")

    assert chunks == [
        "abcdefghij",
        "x y",
    ]


def test_split_reuses_trailing_words_within_overlap_limit() -> None:
    """Reuse trailing whole words when they fit the configured overlap."""
    service = ChunkingService(
        chunk_size=13,
        chunk_overlap=7,
    )

    chunks = service.split("one two three four five six")

    assert chunks == [
        "one two three",
        "three four",
        "four five six",
    ]


def test_split_does_not_exceed_overlap_to_include_another_word() -> None:
    """Do not reuse a trailing word when it would exceed overlap size."""
    service = ChunkingService(
        chunk_size=13,
        chunk_overlap=4,
    )

    chunks = service.split("one two three four five six")

    assert chunks == [
        "one two three",
        "four five six",
    ]


def test_split_advances_when_entire_previous_chunk_fits_overlap() -> None:
    """Advance to new text when reusing a whole chunk would prevent progress."""
    service = ChunkingService(
        chunk_size=10,
        chunk_overlap=9,
    )

    chunks = service.split("a abcdefghij z")

    assert chunks == [
        "a",
        "abcdefghij",
        "z",
    ]


@pytest.mark.parametrize("max_tokens", [0, -1, -100])
def test_split_with_token_limit_rejects_non_positive_max_tokens(
    max_tokens: int,
) -> None:
    """Reject zero or negative embedding token limits."""
    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=0,
    )

    with pytest.raises(
        ValueError,
        match="max_tokens must be greater than zero",
    ):
        service.split_with_token_limit(
            "alpha beta",
            count_tokens=count_words,
            max_tokens=max_tokens,
        )


def test_split_with_token_limit_returns_empty_list_for_empty_text() -> None:
    """Return no token-aware chunks when the source text is empty."""
    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=10,
    )

    chunks = service.split_with_token_limit(
        "",
        count_tokens=count_words,
        max_tokens=10,
    )

    assert chunks == []


def test_split_with_token_limit_keeps_safe_chunks_and_token_counts() -> None:
    """Preserve character chunks that already satisfy the token limit."""
    service = ChunkingService(
        chunk_size=10,
        chunk_overlap=0,
    )

    chunks = service.split_with_token_limit(
        "one two three four",
        count_tokens=count_words,
        max_tokens=2,
    )

    assert chunks == [
        ("one two", 2),
        ("three four", 2),
    ]


def test_split_with_token_limit_uses_supplied_token_counter() -> None:
    """Measure final chunks using the tokenizer callable from the caller."""
    measured_texts: list[str] = []

    def tracking_counter(text: str) -> int:
        measured_texts.append(text)
        return len(text.split())

    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=0,
    )

    chunks = service.split_with_token_limit(
        "alpha beta gamma",
        count_tokens=tracking_counter,
        max_tokens=10,
    )

    assert chunks == [
        ("alpha beta gamma", 3),
    ]
    assert measured_texts == [
        "alpha beta gamma",
    ]


def test_split_with_token_limit_subdivides_at_word_boundaries() -> None:
    """Subdivide token-oversized text while preferring word boundaries."""
    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=0,
    )

    chunks = service.split_with_token_limit(
        "one two three four five",
        count_tokens=count_characters,
        max_tokens=7,
    )

    assert chunks == [
        ("one two", 7),
        ("three", 5),
        ("four", 4),
        ("five", 4),
    ]


def test_split_with_token_limit_splits_long_word_when_required() -> None:
    """Split inside a word when no word boundary satisfies the token limit."""
    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=0,
    )

    chunks = service.split_with_token_limit(
        "abcdefghij",
        count_tokens=count_characters,
        max_tokens=4,
    )

    assert chunks == [
        ("abcd", 4),
        ("efgh", 4),
        ("ij", 2),
    ]


def test_split_with_token_limit_preserves_overlap_between_fragments() -> None:
    """Reuse trailing text at a word boundary during token subdivision."""
    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=5,
    )

    chunks = service.split_with_token_limit(
        "alpha beta gamma delta",
        count_tokens=count_characters,
        max_tokens=10,
    )

    assert chunks == [
        ("alpha beta", 10),
        ("beta gamma", 10),
        ("delta", 5),
    ]


def test_split_with_token_limit_avoids_overlap_inside_long_word() -> None:
    """Avoid character overlap when no usable word boundary exists."""
    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=2,
    )

    chunks = service.split_with_token_limit(
        "abcdefghij",
        count_tokens=count_characters,
        max_tokens=4,
    )

    assert chunks == [
        ("abcd", 4),
        ("efgh", 4),
        ("ij", 2),
    ]


def test_split_with_token_limit_handles_multiple_character_chunks() -> None:
    """Apply the token limit independently to each character chunk."""
    service = ChunkingService(
        chunk_size=10,
        chunk_overlap=0,
    )

    chunks = service.split_with_token_limit(
        "alpha beta gamma delta epsilon",
        count_tokens=count_characters,
        max_tokens=5,
    )

    assert chunks == [
        ("alpha", 5),
        ("beta", 4),
        ("gamma", 5),
        ("delta", 5),
        ("epsil", 5),
        ("on", 2),
    ]


def test_split_with_token_limit_rejects_when_no_fragment_can_fit() -> None:
    """Fail when every non-empty fragment exceeds the token limit."""

    def always_oversized(text: str) -> int:
        return 2 if text else 0

    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unable to split document text within the embedding model " "token limit"
        ),
    ):
        service.split_with_token_limit(
            "alpha",
            count_tokens=always_oversized,
            max_tokens=1,
        )


def test_split_with_token_limit_reports_final_measured_counts() -> None:
    """Return the measured token count alongside every final fragment."""
    service = ChunkingService(
        chunk_size=100,
        chunk_overlap=0,
    )

    chunks = service.split_with_token_limit(
        "one two three four",
        count_tokens=count_words,
        max_tokens=2,
    )

    assert all(token_count == count_words(chunk) for chunk, token_count in chunks)
    assert all(token_count <= 2 for _, token_count in chunks)
