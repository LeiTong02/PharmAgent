"""Tests for admin upload endpoint."""
from __future__ import annotations

import io
import pytest
from unittest.mock import MagicMock, patch


async def test_admin_page_renders_for_admin(admin_client):
    with patch("rag.vectorstore.list_uploaded_files", return_value=[]):
        resp = await admin_client.get("/admin")
    assert resp.status_code == 200
    assert b"Upload" in resp.content or b"upload" in resp.content


async def test_admin_page_403_for_researcher(researcher_client):
    resp = await researcher_client.get("/admin")
    assert resp.status_code == 403


async def test_upload_rejects_non_pdf(admin_client):
    fake_txt = io.BytesIO(b"This is a text file, not a PDF.")
    resp = await admin_client.post(
        "/admin/upload",
        files={"file": ("document.txt", fake_txt, "text/plain")},
    )
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]


async def test_upload_rejects_empty_file(admin_client):
    empty = io.BytesIO(b"")
    resp = await admin_client.post(
        "/admin/upload",
        files={"file": ("empty.pdf", empty, "application/pdf")},
    )
    assert resp.status_code == 400


async def test_upload_pdf_success(admin_client):
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake pdf content for testing purposes")
    mock_chunks = [MagicMock()]
    mock_image_tuples = []  # load_pdf_bytes now returns (text_docs, image_doc_tuples)
    mock_wiki_docs = [MagicMock()]

    with (
        patch("rag.loader.load_pdf_bytes", return_value=(mock_chunks, mock_image_tuples)),
        patch("rag.vectorstore.add_documents", return_value=3),
        patch("rag.vectorstore.add_image_documents", return_value=0),
        patch("rag.wiki_generator.docs_to_wiki_documents", return_value=mock_wiki_docs),
        patch("rag.vectorstore.add_wiki_documents", return_value=1),
    ):
        resp = await admin_client.post(
            "/admin/upload",
            files={"file": ("test_paper.pdf", fake_pdf, "application/pdf")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["chunks"] == 3
    assert data["wiki_pages"] == 1
    assert data["filename"] == "test_paper.pdf"


async def test_upload_403_for_researcher(researcher_client):
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake")
    resp = await researcher_client.post(
        "/admin/upload",
        files={"file": ("paper.pdf", fake_pdf, "application/pdf")},
    )
    assert resp.status_code == 403


async def test_list_files_for_admin(admin_client):
    with patch(
        "rag.vectorstore.list_uploaded_files",
        return_value=["paper1.pdf", "paper2.pdf"],
    ):
        resp = await admin_client.get("/admin/files")
    assert resp.status_code == 200
    assert resp.json()["files"] == ["paper1.pdf", "paper2.pdf"]


async def test_list_files_403_for_researcher(researcher_client):
    resp = await researcher_client.get("/admin/files")
    assert resp.status_code == 403


def test_load_pdf_bytes_rejects_string():
    """Passing a string as file_bytes (swapped args) must raise TypeError immediately."""
    from rag.loader import load_pdf_bytes
    with pytest.raises(TypeError, match="expected bytes"):
        load_pdf_bytes("filename.pdf", b"ignored")
