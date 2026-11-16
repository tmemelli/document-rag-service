"""
Tests for document upload validation and bounded file reading.
"""

from io import BytesIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile, status

from document_rag_service.api import routes


def _build_upload(
    *,
    filename: str | None,
    content: bytes = b"document content",
) -> UploadFile:
    """Create an in-memory upload for route-level validation tests."""
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
    )


def _patch_upload_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_size_bytes: int = 1024,
) -> None:
    """Replace application settings with a deterministic upload size limit."""
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            document_upload_max_size_bytes=max_size_bytes,
        ),
    )


async def _call_upload_document(
    *,
    file: UploadFile,
    document_service: Any,
    ingestion_service: Any | None = None,
    title: str | None = None,
) -> None:
    """Call the upload endpoint with lightweight test doubles."""
    current_user: Any = SimpleNamespace(id=42)
    session: Any = object()
    ingestion: Any = ingestion_service or SimpleNamespace(
        ingest_file_content=AsyncMock(),
    )

    await routes.upload_document(
        current_user=current_user,
        session=session,
        document_service=document_service,
        ingestion_service=ingestion,
        file=file,
        title=title,
    )


@pytest.mark.asyncio
async def test_read_upload_with_limit_returns_empty_bytes_for_empty_file() -> None:
    """Return empty bytes when the uploaded stream contains no data."""
    upload = _build_upload(
        filename="empty.pdf",
        content=b"",
    )

    content = await routes._read_upload_with_limit(
        upload,
        max_size_bytes=10,
    )

    assert content == b""


@pytest.mark.asyncio
async def test_read_upload_with_limit_accepts_content_below_limit() -> None:
    """Read and return content whose size remains below the configured limit."""
    upload = _build_upload(
        filename="document.pdf",
        content=b"12345",
    )

    content = await routes._read_upload_with_limit(
        upload,
        max_size_bytes=6,
    )

    assert content == b"12345"


@pytest.mark.asyncio
async def test_read_upload_with_limit_accepts_content_exactly_at_limit() -> None:
    """Accept an upload whose byte length exactly matches the size limit."""
    upload = _build_upload(
        filename="document.pdf",
        content=b"12345",
    )

    content = await routes._read_upload_with_limit(
        upload,
        max_size_bytes=5,
    )

    assert content == b"12345"


@pytest.mark.asyncio
async def test_read_upload_with_limit_rejects_content_above_limit() -> None:
    """Reject an upload as soon as its accumulated size exceeds the limit."""
    upload = _build_upload(
        filename="document.pdf",
        content=b"123456",
    )

    with pytest.raises(HTTPException) as exc_info:
        await routes._read_upload_with_limit(
            upload,
            max_size_bytes=5,
        )

    assert exc_info.value.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert "Uploaded document exceeds the maximum allowed size" in str(
        exc_info.value.detail
    )


@pytest.mark.asyncio
async def test_read_upload_with_limit_reads_incrementally_and_stops_on_overflow() -> (
    None
):
    """Stop consuming the stream once a newly read chunk exceeds the limit."""

    class UploadReader:
        def __init__(self) -> None:
            self._chunks = iter(
                [
                    b"1234",
                    b"5678",
                    b"unused",
                ]
            )
            self.requested_sizes: list[int] = []

        async def read(self, size: int) -> bytes:
            self.requested_sizes.append(size)
            return next(self._chunks, b"")

    upload: Any = UploadReader()

    with pytest.raises(HTTPException) as exc_info:
        await routes._read_upload_with_limit(
            upload,
            max_size_bytes=6,
        )

    assert exc_info.value.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert upload.requested_sizes == [
        routes.UPLOAD_READ_CHUNK_SIZE_BYTES,
        routes.UPLOAD_READ_CHUNK_SIZE_BYTES,
    ]


@pytest.mark.asyncio
async def test_upload_document_rejects_missing_file_name() -> None:
    """Reject an upload that does not provide an original file name."""
    upload = _build_upload(filename=None)

    document_service: Any = SimpleNamespace(
        register_uploaded_document=AsyncMock(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _call_upload_document(
            file=upload,
            document_service=document_service,
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "Uploaded file must have a file name"
    document_service.register_uploaded_document.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "expected_extension"),
    [
        ("document.txt", ".txt"),
        ("document.md", ".md"),
        ("document", "<missing>"),
    ],
)
async def test_upload_document_rejects_unsupported_extension(
    filename: str,
    expected_extension: str,
) -> None:
    """Reject file extensions unsupported by the ingestion pipeline."""
    upload = _build_upload(filename=filename)

    document_service: Any = SimpleNamespace(
        register_uploaded_document=AsyncMock(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _call_upload_document(
            file=upload,
            document_service=document_service,
        )

    assert exc_info.value.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    assert exc_info.value.detail == (
        f"Unsupported document extension: {expected_extension}"
    )
    document_service.register_uploaded_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_document_rejects_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a supported document whose uploaded content is empty."""
    _patch_upload_settings(monkeypatch)

    upload = _build_upload(
        filename="document.pdf",
        content=b"",
    )

    document_service: Any = SimpleNamespace(
        register_uploaded_document=AsyncMock(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _call_upload_document(
            file=upload,
            document_service=document_service,
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "Uploaded document cannot be empty"
    assert upload.file.closed
    document_service.register_uploaded_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_document_rejects_oversized_content_and_closes_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate size-limit rejection while always closing the upload stream."""
    _patch_upload_settings(
        monkeypatch,
        max_size_bytes=5,
    )

    upload = _build_upload(
        filename="document.pdf",
        content=b"123456",
    )

    document_service: Any = SimpleNamespace(
        register_uploaded_document=AsyncMock(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _call_upload_document(
            file=upload,
            document_service=document_service,
        )

    assert exc_info.value.status_code == status.HTTP_413_CONTENT_TOO_LARGE
    assert upload.file.closed
    document_service.register_uploaded_document.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "expected_mime_type"),
    [
        (
            "document.pdf",
            "application/pdf",
        ),
        (
            "document.docx",
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document",
        ),
        (
            "document.odt",
            "application/vnd.oasis.opendocument.text",
        ),
        (
            "document.rtf",
            "application/rtf",
        ),
        (
            "document.fodt",
            "application/vnd.oasis.opendocument.text-flat-xml",
        ),
    ],
)
async def test_upload_document_uses_canonical_mime_type_for_supported_extension(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    expected_mime_type: str,
) -> None:
    """Derive stored MIME type from the supported file extension."""
    _patch_upload_settings(monkeypatch)

    upload = _build_upload(
        filename=filename,
        content=b"content",
    )

    register = AsyncMock(
        side_effect=ValueError("validation stop"),
    )

    document_service: Any = SimpleNamespace(
        register_uploaded_document=register,
    )

    with pytest.raises(HTTPException) as exc_info:
        await _call_upload_document(
            file=upload,
            document_service=document_service,
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    register.assert_awaited_once_with(
        user_id=42,
        title="document",
        file_name=filename,
        mime_type=expected_mime_type,
        content=b"content",
    )

    assert upload.file.closed


@pytest.mark.asyncio
async def test_upload_document_matches_extension_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept uppercase extensions by normalizing them before validation."""
    _patch_upload_settings(monkeypatch)

    upload = _build_upload(
        filename="REPORT.PDF",
        content=b"content",
    )

    register = AsyncMock(
        side_effect=ValueError("validation stop"),
    )

    document_service: Any = SimpleNamespace(
        register_uploaded_document=register,
    )

    with pytest.raises(HTTPException):
        await _call_upload_document(
            file=upload,
            document_service=document_service,
        )

    register.assert_awaited_once_with(
        user_id=42,
        title="REPORT",
        file_name="REPORT.PDF",
        mime_type="application/pdf",
        content=b"content",
    )


@pytest.mark.asyncio
async def test_upload_document_strips_directory_components_from_file_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store only the basename when a supplied file name contains a path."""
    _patch_upload_settings(monkeypatch)

    upload = _build_upload(
        filename="../../private/report.pdf",
        content=b"content",
    )

    register = AsyncMock(
        side_effect=ValueError("validation stop"),
    )

    document_service: Any = SimpleNamespace(
        register_uploaded_document=register,
    )

    with pytest.raises(HTTPException):
        await _call_upload_document(
            file=upload,
            document_service=document_service,
        )

    register.assert_awaited_once_with(
        user_id=42,
        title="report",
        file_name="report.pdf",
        mime_type="application/pdf",
        content=b"content",
    )


@pytest.mark.asyncio
async def test_upload_document_strips_explicit_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trim surrounding whitespace from a caller-supplied document title."""
    _patch_upload_settings(monkeypatch)

    upload = _build_upload(
        filename="document.pdf",
        content=b"content",
    )

    register = AsyncMock(
        side_effect=ValueError("validation stop"),
    )

    document_service: Any = SimpleNamespace(
        register_uploaded_document=register,
    )

    with pytest.raises(HTTPException):
        await _call_upload_document(
            file=upload,
            document_service=document_service,
            title="  My document title  ",
        )

    assert register.await_args is not None
    assert register.await_args.kwargs["title"] == "My document title"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "title",
    [
        None,
        "",
        " ",
        "\t",
        "\n",
    ],
)
async def test_upload_document_uses_file_stem_when_title_is_missing_or_blank(
    monkeypatch: pytest.MonkeyPatch,
    title: str | None,
) -> None:
    """Fall back to the sanitized file stem when no meaningful title is given."""
    _patch_upload_settings(monkeypatch)

    upload = _build_upload(
        filename="my-document.pdf",
        content=b"content",
    )

    register = AsyncMock(
        side_effect=ValueError("validation stop"),
    )

    document_service: Any = SimpleNamespace(
        register_uploaded_document=register,
    )

    with pytest.raises(HTTPException):
        await _call_upload_document(
            file=upload,
            document_service=document_service,
            title=title,
        )

    assert register.await_args is not None
    assert register.await_args.kwargs["title"] == "my-document"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Document content already exists (document_id=1)",
        "Document content is already being processed (document_id=1)",
    ],
)
async def test_upload_document_maps_duplicate_registration_errors_to_conflict(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    """Translate duplicate or in-progress document errors into HTTP 409."""
    _patch_upload_settings(monkeypatch)

    upload = _build_upload(
        filename="document.pdf",
        content=b"content",
    )

    document_service: Any = SimpleNamespace(
        register_uploaded_document=AsyncMock(
            side_effect=ValueError(message),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _call_upload_document(
            file=upload,
            document_service=document_service,
        )

    assert exc_info.value.status_code == status.HTTP_409_CONFLICT
    assert exc_info.value.detail == message


@pytest.mark.asyncio
async def test_upload_document_maps_other_registration_errors_to_bad_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate other registration validation errors into HTTP 400."""
    _patch_upload_settings(monkeypatch)

    upload = _build_upload(
        filename="document.pdf",
        content=b"content",
    )

    document_service: Any = SimpleNamespace(
        register_uploaded_document=AsyncMock(
            side_effect=ValueError("Invalid document"),
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _call_upload_document(
            file=upload,
            document_service=document_service,
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "Invalid document"
