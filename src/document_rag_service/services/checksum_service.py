"""Checksum generation utilities for document content."""

import hashlib


class ChecksumService:
    """
    Generate deterministic cryptographic checksums for binary content.

    The service centralizes checksum generation used by the document ingestion
    flow, allowing uploaded content to be compared reliably without depending
    on file names, titles, or metadata.
    """

    def sha256(self, content: bytes) -> str:
        """
        Generate the SHA-256 digest for binary content.

        Args:
            content: Raw binary content to hash.

        Returns:
            Lowercase hexadecimal SHA-256 digest containing 64 characters.
        """

        return hashlib.sha256(content).hexdigest()
