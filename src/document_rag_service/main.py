"""
FastAPI application entry point for the Document RAG Service.

This module creates the HTTP application, applies cross-origin access rules,
attaches the application lifespan, and registers the public API router.
"""

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from document_rag_service.api.routes import api_router
from document_rag_service.core.lifespan import lifespan
from document_rag_service.core.settings import get_settings

settings = get_settings()

app = FastAPI(
    title="Document RAG Service",
    description=(
        "Production-oriented Retrieval-Augmented Generation API with document "
        "ingestion, vector retrieval, query transformation, reciprocal rank fusion, "
        "context expansion, deterministic context budgeting, and grounded LLM answers."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "DELETE",
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
    ],
    expose_headers=[
        "Retry-After",
    ],
)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Return basic service information."""
    return {
        "service": "document-rag-service",
        "status": "ok",
    }


@app.head("/", include_in_schema=False)
async def root_head() -> Response:
    """Respond to infrastructure probes against the application root."""
    return Response(status_code=204)


app.include_router(api_router)
