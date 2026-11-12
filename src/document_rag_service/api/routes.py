"""HTTP routes exposed by the Document RAG API."""

from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from document_rag_service.api.dependencies import (
    AuthTokenServiceDependency,
    CurrentUser,
    DatabaseSession,
    DocumentServiceDependency,
    IngestionServiceDependency,
    RagServiceDependency,
    UserServiceDependency,
    enforce_login_rate_limit,
    enforce_rag_rate_limit,
    enforce_register_rate_limit,
)
from document_rag_service.api.schemas.auth import (
    AuthResponse,
    DeleteAccountRequest,
    LoginRequest,
    RegisterRequest,
    UserResponse,
)
from document_rag_service.api.schemas.document import (
    DeleteDocumentsRequest,
    DocumentResponse,
)
from document_rag_service.api.schemas.rag import (
    ContextChunkResponse,
    RagQueryRequest,
    RagQueryResponse,
    RetrievalHitResponse,
)
from document_rag_service.core.settings import get_settings
from document_rag_service.repositories.document_repository import (
    DocumentRepository,
)

api_router = APIRouter()

SUPPORTED_UPLOAD_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    ".odt": "application/vnd.oasis.opendocument.text",
    ".rtf": "application/rtf",
    ".fodt": "application/vnd.oasis.opendocument.text-flat-xml",
}
UPLOAD_READ_CHUNK_SIZE_BYTES = 1024 * 1024


async def _read_upload_with_limit(
    file: UploadFile,
    *,
    max_size_bytes: int,
) -> bytes:
    """
    Read an uploaded file without exceeding the configured size limit.

    The upload is consumed incrementally so oversized files can be rejected
    before their complete contents are loaded into application memory.

    Args:
        file: Uploaded file supplied by the client.
        max_size_bytes: Maximum number of bytes accepted for one document.

    Returns:
        Complete uploaded content when it remains within the configured limit.

    Raises:
        HTTPException: If the uploaded document exceeds the configured maximum
            size.
    """

    content = bytearray()

    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_SIZE_BYTES)

        if not chunk:
            break

        if len(content) + len(chunk) > max_size_bytes:
            max_size_mb = max_size_bytes / (1024 * 1024)

            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    "Uploaded document exceeds the maximum allowed size "
                    f"of {max_size_mb:g} MB"
                ),
            )

        content.extend(chunk)

    return bytes(content)


@api_router.get(
    "/health",
    tags=["Health"],
    summary="Check API health",
)
async def health_check() -> dict[str, str]:
    """
    Return the current API health status.

    Returns:
        Minimal health response confirming that the application process is
        available.
    """

    return {"status": "ok"}


@api_router.post(
    "/v1/auth/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
    summary="Register a new user",
    dependencies=[Depends(enforce_register_rate_limit)],
)
async def register_user(
    payload: RegisterRequest,
    user_service: UserServiceDependency,
    token_service: AuthTokenServiceDependency,
) -> AuthResponse:
    """
    Register a user and immediately issue an access token.

    Args:
        payload: Registration credentials.
        user_service: Request-scoped user service.
        token_service: Shared authentication token service.

    Returns:
        Access token and public user information.

    Raises:
        HTTPException: If the email is already registered or registration
            data is invalid.
    """

    try:
        user = await user_service.register_user(
            email=str(payload.email),
            password=payload.password,
        )
    except ValueError as exc:
        if str(exc) == "Email is already registered":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email is already registered",
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    access_token = token_service.create_access_token(
        user_id=user.id,
    )

    return AuthResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user),
    )


@api_router.post(
    "/v1/auth/login",
    response_model=AuthResponse,
    tags=["Authentication"],
    summary="Authenticate a user",
    dependencies=[Depends(enforce_login_rate_limit)],
)
async def login_user(
    payload: LoginRequest,
    user_service: UserServiceDependency,
    token_service: AuthTokenServiceDependency,
) -> AuthResponse:
    """
    Authenticate a user and issue an access token.

    Args:
        payload: Login credentials.
        user_service: Request-scoped user service.
        token_service: Shared authentication token service.

    Returns:
        Access token and public user information.

    Raises:
        HTTPException: If authentication fails.
    """

    try:
        user = await user_service.authenticate_user(
            email=str(payload.email),
            password=payload.password,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        ) from exc

    access_token = token_service.create_access_token(
        user_id=user.id,
    )

    return AuthResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user),
    )


@api_router.get(
    "/v1/auth/me",
    response_model=UserResponse,
    tags=["Authentication"],
    summary="Get the authenticated user",
)
async def get_authenticated_user(
    current_user: CurrentUser,
) -> UserResponse:
    """
    Return the currently authenticated user.

    Args:
        current_user: User resolved from the Bearer access token.

    Returns:
        Public authenticated user information.
    """

    return UserResponse.model_validate(current_user)


@api_router.get(
    "/v1/documents",
    response_model=list[DocumentResponse],
    tags=["Documents"],
    summary="List the authenticated user's documents",
)
async def list_documents(
    current_user: CurrentUser,
    session: DatabaseSession,
) -> list[DocumentResponse]:
    """
    List documents owned by the authenticated user.

    Args:
        current_user: Authenticated user resolved from the Bearer token.
        session: Request-scoped database session.

    Returns:
        Documents owned exclusively by the authenticated user.
    """

    document_repository = DocumentRepository(session)

    documents = await document_repository.list_all(
        user_id=current_user.id,
    )

    return [DocumentResponse.model_validate(document) for document in documents]


@api_router.post(
    "/v1/documents/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload and ingest a document",
)
async def upload_document(
    current_user: CurrentUser,
    session: DatabaseSession,
    document_service: DocumentServiceDependency,
    ingestion_service: IngestionServiceDependency,
    file: Annotated[UploadFile, File()],
    title: Annotated[str | None, Form()] = None,
) -> DocumentResponse:
    """
    Upload and ingest a document owned by the authenticated user.

    The file extension determines the supported document format and canonical
    MIME type stored with the document. Registration occurs before ingestion so
    the complete processing lifecycle can be persisted and observed.

    Args:
        current_user: Authenticated user resolved from the Bearer token.
        session: Request-scoped database session.
        document_service: Request-scoped document service.
        ingestion_service: Request-scoped ingestion service.
        file: Uploaded document.
        title: Optional document title. When omitted or blank, the file stem is
            used.

    Returns:
        Document metadata after successful ingestion.

    Raises:
        HTTPException: If the upload is empty, unsupported, duplicated,
            contains invalid document content, or cannot be retrieved after
            ingestion.
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a file name",
        )

    file_name = Path(file.filename).name
    extension = Path(file_name).suffix.lower()

    mime_type = SUPPORTED_UPLOAD_MIME_TYPES.get(extension)

    if mime_type is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(f"Unsupported document extension: {extension or '<missing>'}"),
        )

    settings = get_settings()

    try:
        content = await _read_upload_with_limit(
            file,
            max_size_bytes=settings.document_upload_max_size_bytes,
        )
    finally:
        await file.close()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded document cannot be empty",
        )

    document_title = (
        title.strip() if title is not None and title.strip() else Path(file_name).stem
    )

    try:
        document = await document_service.register_uploaded_document(
            user_id=current_user.id,
            title=document_title,
            file_name=file_name,
            mime_type=mime_type,
            content=content,
        )
    except ValueError as exc:
        message = str(exc)

        if "already exists" in message or "already being processed" in message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=message,
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        ) from exc

    try:
        await ingestion_service.ingest_file_content(
            document_id=document.id,
            user_id=current_user.id,
            content=content,
            file_name=file_name,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    document_repository = DocumentRepository(session)

    refreshed_document = await document_repository.get_by_id(
        document_id=document.id,
        user_id=current_user.id,
    )

    if refreshed_document is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document could not be retrieved after ingestion",
        )

    return DocumentResponse.model_validate(refreshed_document)


@api_router.post(
    "/v1/rag/query",
    response_model=RagQueryResponse,
    tags=["RAG"],
    summary="Query authenticated user's documents",
    dependencies=[Depends(enforce_rag_rate_limit)],
)
async def query_rag(
    payload: RagQueryRequest,
    current_user: CurrentUser,
    rag_service: RagServiceDependency,
) -> RagQueryResponse:
    """
    Answer a question using documents owned by the authenticated user.

    Args:
        payload: RAG query configuration and selected document IDs.
        current_user: Authenticated user resolved from the Bearer token.
        rag_service: Request-scoped RAG orchestration service.

    Returns:
        Generated answer and RAG observability information.

    Raises:
        HTTPException: If selected documents are unavailable to the user or
            the RAG request is otherwise invalid.
    """

    try:
        result = await rag_service.answer(
            user_id=current_user.id,
            document_ids=payload.document_ids,
            query=payload.query,
            language=payload.language,
            limit=payload.limit,
            neighbor_window=payload.neighbor_window,
            transform_query=payload.transform_query,
        )
    except ValueError as exc:
        message = str(exc)

        if message == "One or more selected documents are invalid":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more selected documents were not found",
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        ) from exc

    return RagQueryResponse(
        answer=result.answer,
        sources=result.sources,
        retrieval_strategy=result.retrieval_strategy,
        retrieval_queries=result.retrieval_queries,
        retrieval_hits=[
            RetrievalHitResponse(
                document_id=hit.document_id,
                document_title=hit.document_title,
                chunk_index=hit.chunk_index,
                score=hit.score,
                best_similarity=hit.best_similarity,
                matched_query_indexes=list(
                    hit.matched_query_indexes,
                ),
            )
            for hit in result.retrieval_hits
        ],
        context_chunks=[
            ContextChunkResponse(
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                chunk_index=chunk.chunk_index,
                role=chunk.role,
            )
            for chunk in result.context_chunks
        ],
        context_chunks_count=result.context_chunks_count,
        context_characters=result.context_characters,
        max_context_characters=result.max_context_characters,
    )


@api_router.get(
    "/v1/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["Documents"],
    summary="Get one authenticated user's document",
)
async def get_document(
    document_id: int,
    current_user: CurrentUser,
    session: DatabaseSession,
) -> DocumentResponse:
    """
    Return one document owned by the authenticated user.

    Args:
        document_id: Requested document identifier.
        current_user: Authenticated user resolved from the Bearer token.
        session: Request-scoped database session.

    Returns:
        Requested document metadata.

    Raises:
        HTTPException: If the document does not exist for the current user.
    """

    document_repository = DocumentRepository(session)

    document = await document_repository.get_by_id(
        document_id=document_id,
        user_id=current_user.id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentResponse.model_validate(document)


@api_router.delete(
    "/v1/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Documents"],
    summary="Delete one authenticated user's document",
)
async def delete_document(
    document_id: int,
    current_user: CurrentUser,
    document_service: DocumentServiceDependency,
) -> Response:
    """
    Delete one document owned by the authenticated user.

    Args:
        document_id: Requested document identifier.
        current_user: Authenticated user resolved from the Bearer token.
        document_service: Request-scoped document service.

    Returns:
        Empty response after successful deletion.

    Raises:
        HTTPException: If the document does not exist for the current user.
    """

    try:
        await document_service.delete_document(
            document_id=document_id,
            user_id=current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        ) from exc

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
    )


@api_router.post(
    "/v1/documents/bulk-delete",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Documents"],
    summary="Delete multiple authenticated user's documents",
)
async def delete_documents(
    payload: DeleteDocumentsRequest,
    current_user: CurrentUser,
    document_service: DocumentServiceDependency,
) -> Response:
    """
    Delete multiple documents owned by the authenticated user.

    All requested documents are validated before any deletion occurs.

    Args:
        payload: Document identifiers selected for deletion.
        current_user: Authenticated user resolved from the Bearer token.
        document_service: Request-scoped document service.

    Returns:
        Empty response after successful deletion.

    Raises:
        HTTPException: If one or more documents are unavailable to the user or
            the request is invalid.
    """

    try:
        await document_service.delete_documents(
            document_ids=payload.document_ids,
            user_id=current_user.id,
        )
    except ValueError as exc:
        if str(exc) == "One or more documents not found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or more documents were not found",
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
    )


@api_router.post(
    "/v1/account/delete",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Account"],
    summary="Permanently delete the authenticated user's account",
)
async def delete_account(
    payload: DeleteAccountRequest,
    current_user: CurrentUser,
    user_service: UserServiceDependency,
) -> Response:
    """
    Permanently delete the authenticated user's account.

    The current password must be confirmed before deletion.

    Args:
        payload: Current password used to confirm account deletion.
        current_user: Authenticated user resolved from the Bearer token.
        user_service: Request-scoped user service.

    Returns:
        Empty response after successful account deletion.

    Raises:
        HTTPException: If password confirmation fails or the account becomes
            unavailable before deletion completes.
    """

    try:
        await user_service.delete_account(
            user_id=current_user.id,
            password=payload.password,
        )
    except ValueError as exc:
        if str(exc) == "Invalid password":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid password",
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is unavailable",
        ) from exc

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
    )
