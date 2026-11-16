"""
Tests for deterministic SHA-256 checksum generation.
"""

import hashlib
import string

import pytest

from document_rag_service.services.checksum_service import ChecksumService


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"hello",
        b"document content",
        b"\x00\x01\x02\xff",
        "conteúdo com acentuação".encode(),
    ],
)
def test_sha256_matches_standard_library_digest(
    content: bytes,
) -> None:
    """Return the same SHA-256 digest produced by Python's standard library."""
    service = ChecksumService()

    checksum = service.sha256(content)

    assert checksum == hashlib.sha256(content).hexdigest()


def test_sha256_returns_known_digest_for_empty_content() -> None:
    """Return the canonical SHA-256 digest for an empty byte sequence."""
    service = ChecksumService()

    checksum = service.sha256(b"")

    assert checksum == (
        "e3b0c44298fc1c149afbf4c8996fb924" "27ae41e4649b934ca495991b7852b855"
    )


def test_sha256_returns_known_digest_for_known_content() -> None:
    """Return the expected SHA-256 digest for a known byte sequence."""
    service = ChecksumService()

    checksum = service.sha256(b"hello")

    assert checksum == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e" "1b161e5c1fa7425e73043362938b9824"
    )


def test_sha256_is_deterministic() -> None:
    """Return the same checksum whenever identical content is hashed."""
    service = ChecksumService()
    content = b"same document content"

    first_checksum = service.sha256(content)
    second_checksum = service.sha256(content)

    assert first_checksum == second_checksum


def test_sha256_distinguishes_different_content() -> None:
    """Return different checksums when document content differs."""
    service = ChecksumService()

    first_checksum = service.sha256(b"document version one")
    second_checksum = service.sha256(b"document version two")

    assert first_checksum != second_checksum


def test_sha256_changes_when_single_byte_changes() -> None:
    """Change the digest when even one byte of the input changes."""
    service = ChecksumService()

    first_checksum = service.sha256(b"abcdefghij")
    second_checksum = service.sha256(b"abcdefghiJ")

    assert first_checksum != second_checksum


def test_sha256_returns_64_character_lowercase_hexadecimal_string() -> None:
    """Return a lowercase hexadecimal SHA-256 representation."""
    service = ChecksumService()

    checksum = service.sha256(b"document content")

    assert isinstance(checksum, str)
    assert len(checksum) == 64
    assert checksum == checksum.lower()
    assert all(character in string.hexdigits.lower() for character in checksum)


def test_sha256_supports_arbitrary_binary_content() -> None:
    """Hash arbitrary binary data without requiring text decoding."""
    service = ChecksumService()
    content = bytes(range(256))

    checksum = service.sha256(content)

    assert checksum == hashlib.sha256(content).hexdigest()


def test_sha256_supports_large_binary_content() -> None:
    """Hash larger document payloads without changing checksum semantics."""
    service = ChecksumService()
    content = b"document-data-" * 100_000

    checksum = service.sha256(content)

    assert checksum == hashlib.sha256(content).hexdigest()
    assert len(checksum) == 64
