"""Tests for the FastAPI application entry point."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import Response

import document_rag_service.core.settings as settings_module


def _import_main(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the application entry point with isolated settings."""
    monkeypatch.setattr(
        settings_module,
        "get_settings",
        lambda: SimpleNamespace(
            cors_allowed_origins=["https://frontend.example.com"],
        ),
    )

    monkeypatch.delitem(
        sys.modules,
        "document_rag_service.main",
        raising=False,
    )

    return importlib.import_module("document_rag_service.main")


def test_main_composes_expected_fastapi_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the expected metadata and public routes from the real app."""
    main_module = _import_main(monkeypatch)

    assert main_module.app.title == "Document RAG Service"
    assert main_module.app.version == "0.1.0"

    app_paths = {
        path
        for route in main_module.app.routes
        if (path := getattr(route, "path", None)) is not None
    }

    openapi_paths = set(main_module.app.openapi()["paths"])

    assert "/" in app_paths
    assert "/health" in openapi_paths
    assert "/v1/auth/register" in openapi_paths
    assert "/v1/documents/upload" in openapi_paths
    assert "/v1/rag/query" in openapi_paths


@pytest.mark.asyncio
async def test_root_handlers_return_probe_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return stable body and status values from root probe handlers."""
    main_module = _import_main(monkeypatch)

    assert await main_module.root() == {
        "service": "document-rag-service",
        "status": "ok",
    }

    response = await main_module.root_head()

    assert isinstance(response, Response)
    assert response.status_code == 204
