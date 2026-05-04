"""Admin routes: PDF upload, file listing, index rebuild."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse

from frontend.deps import require_admin

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("", response_class=HTMLResponse)
async def admin_page(request: Request, user: dict = Depends(require_admin)):
    try:
        from rag.vectorstore import list_uploaded_files
        uploaded = list_uploaded_files()
    except Exception:
        uploaded = []
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "admin.html",
        context={"user": user, "uploaded_files": uploaded},
    )


@router.post("/upload")
async def upload_pdf(
    request: Request,
    user: dict = Depends(require_admin),
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        from rag.loader import load_pdf_bytes
        from rag.vectorstore import add_documents, add_wiki_documents
        from rag.wiki_generator import docs_to_wiki_documents

        # Classic RAG
        chunks = load_pdf_bytes(file.filename, contents)
        n_classic = add_documents(chunks)

        # Wiki RAG
        wiki_docs = docs_to_wiki_documents(chunks)
        n_wiki = add_wiki_documents(wiki_docs)

        return {
            "ok": True,
            "filename": file.filename,
            "chunks": n_classic,
            "wiki_pages": n_wiki,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")


@router.get("/files")
async def list_files(user: dict = Depends(require_admin)):
    try:
        from rag.vectorstore import list_uploaded_files
        return {"files": list_uploaded_files()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
