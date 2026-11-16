"""
Tests for supported document text extraction and normalization.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from document_rag_service.services import (
    text_extraction_service as text_extraction_module,
)
from document_rag_service.services.text_extraction_service import (
    TextExtractionService,
)


def test_supported_extensions_contains_expected_document_formats() -> None:
    """Expose exactly the document extensions supported by ingestion."""
    assert TextExtractionService.SUPPORTED_EXTENSIONS == frozenset(
        {
            ".pdf",
            ".docx",
            ".odt",
            ".rtf",
            ".fodt",
        }
    )


@pytest.mark.parametrize(
    "file_name",
    [
        "",
        " ",
        "   ",
        "\t",
        "\n",
    ],
)
def test_extract_text_rejects_empty_or_blank_file_name(
    file_name: str,
) -> None:
    """Reject missing meaningful file names before format detection."""
    service = TextExtractionService()

    with pytest.raises(
        ValueError,
        match="File name cannot be empty",
    ):
        service.extract_text(
            content=b"content",
            file_name=file_name,
        )


def test_extract_text_rejects_empty_content() -> None:
    """Reject an uploaded document whose binary content is empty."""
    service = TextExtractionService()

    with pytest.raises(
        ValueError,
        match="File content cannot be empty",
    ):
        service.extract_text(
            content=b"",
            file_name="document.pdf",
        )


@pytest.mark.parametrize(
    "file_name",
    [
        "document",
        "README",
        "file_without_extension",
    ],
)
def test_extract_text_rejects_missing_extension(
    file_name: str,
) -> None:
    """Require a file extension to choose the correct document parser."""
    service = TextExtractionService()

    with pytest.raises(
        ValueError,
        match="File extension is required",
    ):
        service.extract_text(
            content=b"content",
            file_name=file_name,
        )


@pytest.mark.parametrize(
    ("file_name", "extension"),
    [
        ("document.txt", ".txt"),
        ("document.md", ".md"),
        ("archive.zip", ".zip"),
        ("image.png", ".png"),
    ],
)
def test_extract_text_rejects_unsupported_extension(
    file_name: str,
    extension: str,
) -> None:
    """Reject formats outside the supported document ingestion set."""
    service = TextExtractionService()

    with pytest.raises(
        ValueError,
        match=rf"Unsupported file type: {extension}",
    ):
        service.extract_text(
            content=b"content",
            file_name=file_name,
        )


def test_extract_text_matches_extension_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize uppercase extensions before selecting an extractor."""
    service = TextExtractionService()

    monkeypatch.setattr(
        service,
        "_extract_pdf",
        lambda content: "Extracted PDF text",
    )

    result = service.extract_text(
        content=b"pdf-content",
        file_name="REPORT.PDF",
    )

    assert result == "Extracted PDF text"


def test_extract_text_strips_surrounding_file_name_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strip surrounding file-name whitespace before extension detection."""
    service = TextExtractionService()

    monkeypatch.setattr(
        service,
        "_extract_pdf",
        lambda content: "content",
    )

    result = service.extract_text(
        content=b"pdf-content",
        file_name="  report.pdf  ",
    )

    assert result == "content"


@pytest.mark.parametrize(
    ("file_name", "extractor_name"),
    [
        ("document.pdf", "_extract_pdf"),
        ("document.docx", "_extract_docx"),
        ("document.odt", "_extract_odt"),
        ("document.rtf", "_extract_rtf"),
        ("document.fodt", "_extract_fodt"),
    ],
)
def test_extract_text_dispatches_to_expected_format_extractor(
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    extractor_name: str,
) -> None:
    """Dispatch each supported extension to its format-specific parser."""
    service = TextExtractionService()
    content = b"binary-document-content"

    extractor = MagicMock(
        return_value=" extracted text ",
    )

    monkeypatch.setattr(
        service,
        extractor_name,
        extractor,
    )

    result = service.extract_text(
        content=content,
        file_name=file_name,
    )

    assert result == "extracted text"
    extractor.assert_called_once_with(content)


def test_extract_text_wraps_parser_failure_as_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose parser-specific failures through a stable service exception."""
    service = TextExtractionService()

    parser_error = RuntimeError("parser exploded")

    def failing_extractor(content: bytes) -> str:
        raise parser_error

    monkeypatch.setattr(
        service,
        "_extract_pdf",
        failing_extractor,
    )

    with pytest.raises(
        ValueError,
        match="Failed to extract text from .pdf document",
    ) as exc_info:
        service.extract_text(
            content=b"invalid-pdf",
            file_name="document.pdf",
        )

    assert exc_info.value.__cause__ is parser_error


@pytest.mark.parametrize(
    "raw_text",
    [
        "",
        " ",
        "\n",
        "\r\n",
        "\t\n ",
        "\x00",
        "\n\n\r\n",
    ],
)
def test_extract_text_rejects_document_without_extractable_text(
    monkeypatch: pytest.MonkeyPatch,
    raw_text: str,
) -> None:
    """Reject documents whose extracted content becomes empty after cleanup."""
    service = TextExtractionService()

    monkeypatch.setattr(
        service,
        "_extract_pdf",
        lambda content: raw_text,
    )

    with pytest.raises(
        ValueError,
        match="Document contains no extractable text",
    ):
        service.extract_text(
            content=b"pdf-content",
            file_name="document.pdf",
        )


def test_extract_text_normalizes_extracted_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize parser output before returning it to ingestion."""
    service = TextExtractionService()

    monkeypatch.setattr(
        service,
        "_extract_pdf",
        lambda content: (
            "  first line  \r\n" "\r\n" " second line\x00value \r" " third line "
        ),
    )

    result = service.extract_text(
        content=b"pdf-content",
        file_name="document.pdf",
    )

    assert result == ("first line\n" "second line value\n" "third line")


def test_normalize_text_normalizes_line_endings() -> None:
    """Convert CRLF and standalone carriage returns to newline characters."""
    service = TextExtractionService()

    result = service._normalize_text("alpha\r\nbeta\rgamma")

    assert result == "alpha\nbeta\ngamma"


def test_normalize_text_removes_empty_lines() -> None:
    """Discard blank lines from extracted document text."""
    service = TextExtractionService()

    result = service._normalize_text("alpha\n\n   \n beta\n\n gamma")

    assert result == "alpha\nbeta\ngamma"


def test_normalize_text_strips_each_line() -> None:
    """Remove surrounding whitespace independently from every text line."""
    service = TextExtractionService()

    result = service._normalize_text("  alpha  \n\tbeta\t\n   gamma   ")

    assert result == "alpha\nbeta\ngamma"


def test_normalize_text_replaces_null_characters() -> None:
    """Replace null bytes embedded in extracted text with ordinary spaces."""
    service = TextExtractionService()

    result = service._normalize_text("alpha\x00beta")

    assert result == "alpha beta"


def test_extract_pdf_collects_pages_in_document_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extract every PDF page in order and separate pages with blank lines."""
    service = TextExtractionService()

    first_page = MagicMock()
    first_page.get_text.return_value = "page one"

    second_page = MagicMock()
    second_page.get_text.return_value = "page two"

    document = MagicMock()
    document.__enter__.return_value = [
        first_page,
        second_page,
    ]

    open_pdf = MagicMock(
        return_value=document,
    )

    monkeypatch.setattr(
        text_extraction_module.pymupdf,
        "open",
        open_pdf,
    )

    result = service._extract_pdf(b"pdf-content")

    assert result == "page one\n\npage two"

    open_pdf.assert_called_once_with(
        stream=b"pdf-content",
        filetype="pdf",
    )

    first_page.get_text.assert_called_once_with("text")
    second_page.get_text.assert_called_once_with("text")


def test_extract_docx_collects_non_empty_paragraphs_and_table_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collect readable DOCX paragraphs and non-empty table cell values."""
    service = TextExtractionService()

    paragraphs = [
        SimpleNamespace(text=" First paragraph "),
        SimpleNamespace(text="   "),
        SimpleNamespace(text="Second paragraph"),
    ]

    rows = [
        SimpleNamespace(
            cells=[
                SimpleNamespace(text=" Name "),
                SimpleNamespace(text=" Value "),
            ]
        ),
        SimpleNamespace(
            cells=[
                SimpleNamespace(text=" "),
                SimpleNamespace(text="Second row"),
            ]
        ),
        SimpleNamespace(
            cells=[
                SimpleNamespace(text=" "),
                SimpleNamespace(text=" "),
            ]
        ),
    ]

    document = SimpleNamespace(
        paragraphs=paragraphs,
        tables=[
            SimpleNamespace(rows=rows),
        ],
    )

    docx_document = MagicMock(
        return_value=document,
    )

    monkeypatch.setattr(
        text_extraction_module,
        "DocxDocument",
        docx_document,
    )

    result = service._extract_docx(b"docx-content")

    assert result == (
        "First paragraph\n\n" "Second paragraph\n\n" "Name | Value\n\n" "Second row"
    )

    docx_document.assert_called_once()


def test_extract_odt_returns_empty_when_document_has_no_text_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return empty text when an ODT contains no primary office text element."""
    service = TextExtractionService()

    document = MagicMock()
    document.getElementsByType.return_value = []

    load_odf = MagicMock(
        return_value=document,
    )

    monkeypatch.setattr(
        text_extraction_module,
        "load_odf",
        load_odf,
    )

    result = service._extract_odt(b"odt-content")

    assert result == ""

    document.getElementsByType.assert_called_once_with(
        text_extraction_module.office.Text
    )


def test_extract_odt_extracts_primary_text_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extract plain text from the ODT document's primary text element."""
    service = TextExtractionService()

    first_text_element = object()
    second_text_element = object()

    document = MagicMock()
    document.getElementsByType.return_value = [
        first_text_element,
        second_text_element,
    ]

    load_odf = MagicMock(
        return_value=document,
    )

    extract_text = MagicMock(
        return_value="ODT document text",
    )

    monkeypatch.setattr(
        text_extraction_module,
        "load_odf",
        load_odf,
    )

    monkeypatch.setattr(
        text_extraction_module.teletype,
        "extractText",
        extract_text,
    )

    result = service._extract_odt(b"odt-content")

    assert result == "ODT document text"

    extract_text.assert_called_once_with(first_text_element)


def test_extract_rtf_decodes_bytes_with_latin1_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decode RTF bytes losslessly with Latin-1 before invoking the parser."""
    service = TextExtractionService()

    parser = MagicMock(
        return_value="RTF plain text",
    )

    monkeypatch.setattr(
        text_extraction_module,
        "rtf_to_text",
        parser,
    )

    content = b"{\\rtf1 caf\xe9}"

    result = service._extract_rtf(content)

    assert result == "RTF plain text"

    parser.assert_called_once_with(content.decode("latin-1"))


def test_extract_fodt_collects_paragraphs_and_headings() -> None:
    """Extract OpenDocument paragraphs and headings in XML document order."""
    service = TextExtractionService()

    content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <office:document
        xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
        xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
        <office:body>
            <office:text>
                <text:h>Document heading</text:h>
                <text:p>First paragraph</text:p>
                <text:p>Second paragraph</text:p>
            </office:text>
        </office:body>
    </office:document>
    """

    result = service._extract_fodt(content)

    assert result == ("Document heading\n\n" "First paragraph\n\n" "Second paragraph")


def test_extract_fodt_preserves_nested_span_text() -> None:
    """Preserve text nested inside OpenDocument paragraph child elements."""
    service = TextExtractionService()

    content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <office:document
        xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
        xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
        <office:body>
            <office:text>
                <text:p>
                    Sherlock <text:span>Holmes</text:span> observed the clue.
                </text:p>
            </office:text>
        </office:body>
    </office:document>
    """

    result = service._extract_fodt(content)

    assert "Sherlock " in result
    assert "Holmes" in result
    assert " observed the clue." in result


def test_extract_fodt_ignores_non_paragraph_text_elements() -> None:
    """Ignore XML text outside OpenDocument paragraphs and headings."""
    service = TextExtractionService()

    content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <office:document
        xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
        xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
        <office:body>
            <office:text>
                <text:section>Ignored section text</text:section>
                <text:p>Included paragraph</text:p>
            </office:text>
        </office:body>
    </office:document>
    """

    result = service._extract_fodt(content)

    assert result == "Included paragraph"
    assert "Ignored section text" not in result


def test_extract_text_wraps_malformed_fodt_xml_error() -> None:
    """Convert malformed FODT XML parsing errors into the service exception."""
    service = TextExtractionService()

    with pytest.raises(
        ValueError,
        match="Failed to extract text from .fodt document",
    ):
        service.extract_text(
            content=b"<invalid><xml>",
            file_name="document.fodt",
        )
