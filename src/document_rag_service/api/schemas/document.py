"""Request and response schemas for document API operations."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DeleteDocumentsRequest(BaseModel):
    """
    Validate a bulk document-deletion request.

    The request must contain at least one document identifier and is bounded to
    one hundred identifiers per operation to keep destructive bulk requests
    reasonably constrained.

    Attributes:
        document_ids: Identifiers of documents selected for deletion.
    """

    document_ids: list[int] = Field(
        min_length=1,
        max_length=100,
    )


class DocumentResponse(BaseModel):
    """
    Represent the public API view of a user-owned document.

    The schema reads values directly from ORM model attributes while exposing
    document metadata and its current ingestion lifecycle status.

    Attributes:
        id: Unique document identifier.
        title: Human-readable document title.
        source_type: Origin of the document, such as an uploaded file.
        source_url: Optional external source URL associated with the document.
        file_name: Original uploaded file name when applicable.
        mime_type: Stored MIME type when applicable.
        status: Current document processing status.
        created_at: Timestamp when the document record was created.
        updated_at: Timestamp of the most recent document update.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )

    id: int
    title: str
    source_type: str
    source_url: str | None
    file_name: str | None
    mime_type: str | None
    status: str
    created_at: datetime
    updated_at: datetime
