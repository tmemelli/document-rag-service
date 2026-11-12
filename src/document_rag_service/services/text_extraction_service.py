"""Extract and normalize plain text from supported document formats."""

from io import BytesIO
from pathlib import Path
from typing import cast

import pymupdf
from defusedxml import ElementTree
from docx import Document as DocxDocument
from odf import office, teletype
from odf.opendocument import load as load_odf
from striprtf.striprtf import rtf_to_text


class TextExtractionService:
    """
    Extract normalized plain text from supported uploaded document formats.

    The service provides a single format-independent entry point for document
    ingestion while delegating parsing to format-specific libraries.

    Supported formats include PDF, DOCX, ODT, RTF, and flat OpenDocument
    text (FODT). Extracted content is normalized before being returned so
    downstream chunking receives consistent text regardless of the original
    document format.
    """

    SUPPORTED_EXTENSIONS = frozenset(
        {
            ".pdf",
            ".docx",
            ".odt",
            ".rtf",
            ".fodt",
        }
    )

    def extract_text(
        self,
        *,
        content: bytes,
        file_name: str,
    ) -> str:
        """
        Extract normalized text from an uploaded document.

        The document parser is selected from the normalized file extension.
        Parser-specific exceptions are converted into a consistent
        ``ValueError`` so the ingestion layer does not need to depend on
        exceptions from individual document-processing libraries.

        Args:
            content: Raw binary content of the uploaded document.
            file_name: Original file name used to determine the document
                format.

        Returns:
            Normalized, non-empty plain text extracted from the document.

        Raises:
            ValueError: If the file name or content is empty, the extension is
                missing or unsupported, extraction fails, or the document
                contains no extractable text.
        """

        normalized_file_name = file_name.strip()

        if not normalized_file_name:
            raise ValueError("File name cannot be empty")

        if not content:
            raise ValueError("File content cannot be empty")

        extension = Path(normalized_file_name).suffix.lower()

        if not extension:
            raise ValueError("File extension is required")

        extractors = {
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".odt": self._extract_odt,
            ".rtf": self._extract_rtf,
            ".fodt": self._extract_fodt,
        }

        extractor = extractors.get(extension)

        if extractor is None:
            raise ValueError(f"Unsupported file type: {extension}")

        try:
            text = extractor(content)
        except Exception as exc:
            raise ValueError(
                f"Failed to extract text from {extension} document"
            ) from exc

        normalized_text = self._normalize_text(text)

        if not normalized_text:
            raise ValueError("Document contains no extractable text")

        return normalized_text

    def _extract_pdf(
        self,
        content: bytes,
    ) -> str:
        """
        Extract plain text from a PDF document.

        Text is extracted page by page in document order and separated by blank
        lines before global normalization.

        Args:
            content: Raw PDF bytes.

        Returns:
            Text extracted from all PDF pages.
        """

        with pymupdf.open(
            stream=content,
            filetype="pdf",
        ) as document:
            pages = [
                cast(
                    str,
                    page.get_text("text"),
                )
                for page in document
            ]

        return "\n\n".join(pages)

    def _extract_docx(
        self,
        content: bytes,
    ) -> str:
        """
        Extract text from paragraphs and tables in a DOCX document.

        Paragraphs are collected in document order. Table rows are converted
        into readable text by joining non-empty cell values with separators.

        Args:
            content: Raw DOCX bytes.

        Returns:
            Combined text extracted from document paragraphs and tables.
        """

        document = DocxDocument(BytesIO(content))

        parts: list[str] = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                parts.append(text)

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]

                row_text = " | ".join(cell for cell in cells if cell)

                if row_text:
                    parts.append(row_text)

        return "\n\n".join(parts)

    def _extract_odt(
        self,
        content: bytes,
    ) -> str:
        """
        Extract text from an OpenDocument Text document.

        Args:
            content: Raw ODT bytes.

        Returns:
            Text contained in the document's primary office text element, or an
            empty string when no text element is present.
        """

        document = load_odf(BytesIO(content))

        text_elements = document.getElementsByType(office.Text)

        if not text_elements:
            return ""

        return teletype.extractText(text_elements[0])

    def _extract_rtf(
        self,
        content: bytes,
    ) -> str:
        """
        Extract plain text from Rich Text Format content.

        RTF bytes are decoded with Latin-1 before being passed to the RTF
        parser, providing a lossless one-byte mapping for the raw input.

        Args:
            content: Raw RTF bytes.

        Returns:
            Plain text extracted from the RTF document.
        """

        rtf_content = content.decode("latin-1")

        return rtf_to_text(rtf_content)

    def _extract_fodt(
        self,
        content: bytes,
    ) -> str:
        """
        Extract paragraphs and headings from flat OpenDocument XML.

        XML parsing uses ``defusedxml`` to avoid unsafe XML features when
        processing user-supplied FODT content. Text nested inside elements such
        as spans is preserved through ``itertext``.

        Args:
            content: Raw FODT XML bytes.

        Returns:
            Text collected from OpenDocument paragraph and heading elements.
        """

        root = ElementTree.fromstring(content)

        text_namespace = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"

        paragraph_tag = f"{{{text_namespace}}}p"

        heading_tag = f"{{{text_namespace}}}h"

        parts: list[str] = []

        for element in root.iter():
            if element.tag not in {
                paragraph_tag,
                heading_tag,
            }:
                continue

            text = "".join(element.itertext()).strip()

            if text:
                parts.append(text)

        return "\n\n".join(parts)

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        """
        Normalize extracted text before chunking.

        Null characters are replaced with spaces, platform-specific line
        endings are normalized, surrounding whitespace is removed from each
        line, and empty lines are discarded.

        Args:
            text: Raw text produced by a format-specific extractor.

        Returns:
            Cleaned text suitable for downstream chunking and embedding.
        """

        normalized = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")

        lines = [line.strip() for line in normalized.split("\n")]

        return "\n".join(line for line in lines if line).strip()
