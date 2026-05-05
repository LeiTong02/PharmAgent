import base64
import logging
import os
import re
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"
_FIGURES_ROOT = Path(__file__).parent.parent / "data" / "figures"

_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_CHAR_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

logger = logging.getLogger(__name__)

_IMAGE_MARKER_RE = re.compile(r"==> picture \[.+?\] intentionally omitted <==")
_TABLE_BLOCK_RE = re.compile(r"(?:\|[^\n]+\|\n){2,}", re.MULTILINE)
_FIG_CAP_RE = re.compile(r"^(?:Figure|Fig\.?)\s+([\w]+)[.:\s]", re.IGNORECASE)
_TAB_CAP_RE = re.compile(r"^Table\s+([\w]+)[.:\s]", re.IGNORECASE)


def _extract_text(content) -> str:
    """Extract plain text from Gemini content (list of blocks or plain str)."""
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content) if content else ""


def _parse_header(text: str) -> tuple[dict, str]:
    """Extract TITLE/AUTHORS/YEAR/JOURNAL header fields and return (metadata, body)."""
    metadata = {}
    lines = text.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "---":
            body_start = i + 1
            break
        for field in ("TITLE", "AUTHORS", "YEAR", "JOURNAL"):
            if line.startswith(f"{field}:"):
                metadata[field.lower()] = line[len(field) + 1:].strip()
    body = "\n".join(lines[body_start:]).strip()
    return metadata, body


def load_documents() -> list[Document]:
    """Load all .txt research papers from data/docs/, returning chunked Documents with metadata."""
    docs: list[Document] = []

    for txt_file in sorted(DOCS_DIR.glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8")
        metadata, body = _parse_header(raw)
        metadata["source_file"] = txt_file.name
        metadata["source_type"] = "mock"
        metadata["upload_timestamp"] = ""

        chunks = _CHAR_SPLITTER.create_documents(
            texts=[body],
            metadatas=[metadata] * 1,
        )
        docs.extend(chunks)

    return docs


def _get_vision_llm():
    """Return a ChatGoogleGenerativeAI instance for vision, or None if unavailable."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.environ.get("MODEL_NAME", "gemini-1.5-flash-latest")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    except Exception as exc:
        logger.warning("Could not initialize vision LLM: %s", exc)
        return None


def _describe_page_vision(img_b64: str, page_idx: int, llm) -> str | None:
    """
    Ask Gemini vision to describe figures/tables in a pre-rendered page image.
    Returns a formatted description string, or None if nothing significant found.
    """
    from langchain_core.messages import HumanMessage
    try:
        msg = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
            {
                "type": "text",
                "text": (
                    f"This is page {page_idx + 1} of a scientific paper. "
                    "Describe every figure, chart, graph, and data table visible. For each:\n"
                    "- State the figure/table number and caption text if visible\n"
                    "- Describe chart type (bar, line, scatter, confusion matrix, etc.), "
                    "axis labels, and key values or trends\n"
                    "- For tables: list column headers and 2-3 representative data rows\n"
                    "- For biological/microscopy images: describe specimens and annotations\n"
                    "Be specific and scientific. If no figures or tables are present, "
                    "respond with exactly: NO_FIGURES"
                ),
            },
        ])
        response = llm.invoke([msg])
        desc = _extract_text(response.content).strip()
        if not desc or desc == "NO_FIGURES" or len(desc) < 30:
            return None
        return f"[Page {page_idx + 1} figures and tables]\n{desc}"
    except Exception as exc:
        logger.warning("Vision description for page %d failed: %s", page_idx + 1, exc)
        return None


def _extract_table_documents(md_text: str, filename: str, ts: str, doc_id: str) -> list[Document]:
    """
    Extract each markdown pipe-table block as a dedicated Document so tables
    are retrievable even when not in the top-k regular text chunks.
    """
    docs = []
    for i, match in enumerate(_TABLE_BLOCK_RE.finditer(md_text), 1):
        table_text = match.group(0).strip()
        rows = [r for r in table_text.split("\n") if r.strip()]
        if len(rows) < 3:  # need header + separator + at least one data row
            continue
        docs.append(Document(
            page_content=f"[Table {i} from {filename}]\n{table_text}",
            metadata={
                "document_id":        doc_id,
                "source_file":        filename,
                "source_type":        "uploaded",
                "upload_timestamp":   ts,
                "chunk_type":         "table",
                "extraction_method":  "regex",
                "embedding_modality": "text",
                "page_number":        0,
                "figure_index":       "",
                "caption":            "",
                "nearby_text":        "",
                "image_path":         "",
                "figure_url":         "",
                "section":            "",
            },
        ))
    return docs


def _extract_figure_caption_blocks(
    page, page_idx: int, filename: str, ts: str, doc_id: str
) -> tuple[list[Document], list[dict]]:
    """
    Use fitz block-level extraction to find figure and table captions as dedicated chunks.

    Returns (caption_docs, fig_meta_list) where fig_meta_list is used for image cropping:
      [{"figure_index": str, "cap_x0": float, "cap_y0": float,
        "cap_x1": float, "cap_y1": float, "caption": str, "nearby_text": str}, ...]
    """
    caption_docs: list[Document] = []
    fig_meta_list: list[dict] = []

    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    text_blocks = sorted([b for b in blocks if b[6] == 0], key=lambda b: b[1])

    for bi, b in enumerate(text_blocks):
        text = b[4].strip()
        if not text:
            continue

        fig_m = _FIG_CAP_RE.match(text)
        tab_m = _TAB_CAP_RE.match(text)
        match = fig_m or tab_m
        if not match:
            continue

        fig_index = match.group(1)
        ctype = "figure_caption" if fig_m else "table_caption"

        # nearby_text: first paragraph block after this caption
        nearby = text_blocks[bi + 1][4].strip()[:300] if bi + 1 < len(text_blocks) else ""

        fig_url = f"/figures/{filename}/figure_{fig_index}.png" if fig_m else ""
        img_path = str(_FIGURES_ROOT / filename / f"figure_{fig_index}.png") if fig_m else ""

        caption_docs.append(Document(
            page_content=f"{text}\n{nearby}".strip(),
            metadata={
                "document_id":        doc_id,
                "source_file":        filename,
                "source_type":        "uploaded",
                "upload_timestamp":   ts,
                "chunk_type":         ctype,
                "extraction_method":  "fitz_blocks",
                "embedding_modality": "text",
                "page_number":        page_idx + 1,
                "figure_index":       fig_index,
                "caption":            text,
                "nearby_text":        nearby,
                "image_path":         img_path,
                "figure_url":         fig_url,
                "section":            "",
            },
        ))

        if fig_m:
            fig_meta_list.append({
                "figure_index": fig_index,
                "cap_x0": b[0], "cap_y0": b[1],
                "cap_x1": b[2], "cap_y1": b[3],
                "caption": text,
                "nearby_text": nearby,
            })

    return caption_docs, fig_meta_list


def _save_figure_crops(
    page, page_idx: int, filename: str, fig_meta_list: list[dict]
) -> tuple[dict[str, Path], Path]:
    """
    Render the full page PNG and attempt to crop individual figure regions.

    For each figure in fig_meta_list, the figure region is the vertical gap
    ABOVE the caption block.  Falls back to the full-page screenshot when
    the crop region is too narrow (<40 px at document resolution).

    Returns ({figure_index: Path}, page_png_path).
    """
    import fitz  # local import to avoid circular deps in tests

    figures_dir = _FIGURES_ROOT / filename
    figures_dir.mkdir(parents=True, exist_ok=True)

    mat = fitz.Matrix(150 / 72, 150 / 72)
    page_png_path = figures_dir / f"page_{page_idx + 1}.png"
    pix = page.get_pixmap(matrix=mat, alpha=False)
    page_png_path.write_bytes(pix.tobytes("png"))
    logger.info("PDF '%s': saved page_%d.png", filename, page_idx + 1)

    saved: dict[str, Path] = {}
    blocks = page.get_text("blocks")

    for meta in fig_meta_list:
        fig_index = meta["figure_index"]
        cap_y0 = meta["cap_y0"]

        # Figure region is ABOVE the caption; find the nearest preceding text block bottom
        text_blocks_above = [b for b in blocks if b[6] == 0 and b[3] < cap_y0 - 10]
        fig_top = max((b[3] for b in text_blocks_above), default=10) + 4
        fig_bottom = cap_y0 - 4

        if fig_bottom - fig_top < 40:
            logger.info(
                "PDF '%s' figure %s: crop region too small (%.0f px), using page screenshot",
                filename, fig_index, fig_bottom - fig_top,
            )
            saved[fig_index] = page_png_path
            continue

        clip = fitz.Rect(0, fig_top, page.rect.width, fig_bottom)
        pix_crop = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        fig_path = figures_dir / f"figure_{fig_index}.png"
        fig_path.write_bytes(pix_crop.tobytes("png"))
        logger.info(
            "PDF '%s': saved figure_%s.png (page %d, y=%.0f-%.0f)",
            filename, fig_index, page_idx + 1, fig_top, fig_bottom,
        )
        saved[fig_index] = fig_path

    return saved, page_png_path


def _create_image_chunk_docs(
    fig_meta_list: list[dict],
    fig_paths: dict[str, Path],
    page_png_path: Path,
    page_idx: int,
    filename: str,
    ts: str,
    doc_id: str,
) -> list[tuple[Document, bytes]]:
    """
    Create figure_image and page_screenshot Documents paired with their PNG bytes.
    These are returned separately from text docs for image embedding.
    """
    image_docs: list[tuple[Document, bytes]] = []

    for meta in fig_meta_list:
        fig_index = meta["figure_index"]
        fig_path = fig_paths.get(fig_index, page_png_path)
        try:
            png_bytes = fig_path.read_bytes()
        except Exception as exc:
            logger.warning("Could not read figure PNG %s: %s", fig_path, exc)
            continue

        doc = Document(
            page_content=f"[Figure {fig_index} image from {filename}, page {page_idx + 1}]",
            metadata={
                "document_id":        doc_id,
                "source_file":        filename,
                "source_type":        "uploaded",
                "upload_timestamp":   ts,
                "chunk_type":         "figure_image",
                "extraction_method":  "fitz_render",
                "embedding_modality": "image",
                "page_number":        page_idx + 1,
                "figure_index":       fig_index,
                "caption":            meta["caption"],
                "nearby_text":        meta["nearby_text"],
                "image_path":         str(fig_path),
                "figure_url":         f"/figures/{filename}/{fig_path.name}",
                "section":            "",
            },
        )
        image_docs.append((doc, png_bytes))

    # Always add a page_screenshot chunk for the full page
    try:
        page_png_bytes = page_png_path.read_bytes()
        page_doc = Document(
            page_content=f"[Page {page_idx + 1} screenshot from {filename}]",
            metadata={
                "document_id":        doc_id,
                "source_file":        filename,
                "source_type":        "uploaded",
                "upload_timestamp":   ts,
                "chunk_type":         "page_screenshot",
                "extraction_method":  "fitz_render",
                "embedding_modality": "image",
                "page_number":        page_idx + 1,
                "figure_index":       "",
                "caption":            "",
                "nearby_text":        "",
                "image_path":         str(page_png_path),
                "figure_url":         f"/figures/{filename}/{page_png_path.name}",
                "section":            "",
            },
        )
        image_docs.append((page_doc, page_png_bytes))
    except Exception as exc:
        logger.warning("Could not read page PNG %s: %s", page_png_path, exc)

    return image_docs


def load_pdf_bytes(
    file_bytes: bytes, filename: str
) -> tuple[list[Document], list[tuple[Document, bytes]]]:
    """Parse a digital PDF and return (text_docs, image_doc_tuples).

    text_docs:       text chunks + table chunks + figure/table caption chunks (text-embedded)
    image_doc_tuples: [(Document, png_bytes), ...] for figure_image and page_screenshot chunks
                     (image-embedded by add_image_documents in vectorstore.py)

    Extraction strategy:
    1. pymupdf4llm per-page markdown — preserves section headers and pipe tables
    2. Per-page fitz raw text fallback — used when a page yields no markdown
    3. Dedicated table chunks — each markdown table gets its own Document
    4. Layout-aware figure extraction — fitz block-level caption detection
    5. Figure crops and page screenshots — saved as PNGs, returned as image_doc_tuples
    6. Gemini vision descriptions — optional, describes figures in page screenshots

    Raises TypeError if file_bytes is not bytes (catches swapped-argument mistakes).
    Raises ValueError only when ALL pages produce empty text.
    """
    if not isinstance(file_bytes, (bytes, bytearray)):
        raise TypeError(
            f"load_pdf_bytes: expected bytes, got {type(file_bytes).__name__}. "
            "Check argument order: load_pdf_bytes(bytes, filename)."
        )

    try:
        import pymupdf4llm
    except ImportError as e:
        raise ImportError("pymupdf4llm is required for PDF parsing: pip install pymupdf4llm") from e

    try:
        import fitz
    except ImportError as e:
        raise ImportError("PyMuPDF is required: pip install pymupdf") from e

    try:
        fitz_doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open PDF '{filename}': {e}") from e

    n_pages = len(fitz_doc)
    doc_id = str(_uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    logger.info("PDF '%s': %d pages, doc_id=%s — starting extraction", filename, n_pages, doc_id)

    # --- Text extraction (primary: pymupdf4llm, per-page fitz fallback) ---
    md_pages: list[dict] | None = None
    method = "pymupdf4llm"
    try:
        md_pages = pymupdf4llm.to_markdown(fitz_doc, page_chunks=True)
    except Exception as exc:
        logger.warning(
            "PDF '%s': pymupdf4llm failed (%s), using fitz fallback for all pages",
            filename, exc,
        )
        method = "fitz"

    page_texts: list[str] = []
    n_skipped = 0
    for i, fitz_page in enumerate(fitz_doc):
        text = ""
        if md_pages is not None and i < len(md_pages):
            try:
                text = md_pages[i].get("text", "") or ""
            except Exception:
                pass

        if not text.strip():
            try:
                text = fitz_page.get_text()
                if text.strip() and method == "pymupdf4llm":
                    method = "pymupdf4llm+fitz-fallback"
            except Exception as exc:
                logger.warning(
                    "PDF '%s' page %d: fitz fallback also failed (%s)", filename, i + 1, exc
                )

        if text.strip():
            page_texts.append(text)
        else:
            logger.warning("PDF '%s' page %d: no text extracted (skipping)", filename, i + 1)
            n_skipped += 1

    logger.info(
        "PDF '%s': method=%s, %d/%d pages extracted, %d skipped",
        filename, method, n_pages - n_skipped, n_pages, n_skipped,
    )

    if not page_texts:
        fitz_doc.close()
        raise ValueError(
            f"No text extracted from '{filename}'. "
            "The file may be a scanned image-only PDF (no embedded text)."
        )

    # --- Layout-aware figure and page screenshot extraction ---
    all_caption_docs: list[Document] = []
    all_image_doc_tuples: list[tuple[Document, bytes]] = []
    vision_llm = _get_vision_llm()
    vision_docs: list[Document] = []
    n_vision = 0
    max_figure_pages = 15

    if vision_llm:
        logger.info("PDF '%s': vision model available — describing pages with figures", filename)
    else:
        logger.info(
            "PDF '%s': vision unavailable (set OPENAI_API_KEY/GOOGLE_API_KEY to enable). "
            "Page screenshots will still be saved for UI display and image embedding.",
            filename,
        )

    for page_idx in range(n_pages):
        fitz_page = fitz_doc[page_idx]
        images = fitz_page.get_images(full=True)
        sig_images = [img for img in images if img[2] >= 100 and img[3] >= 100]

        # Extract figure captions from block layout (all pages, not just those with images)
        cap_docs, fig_meta_list = _extract_figure_caption_blocks(
            fitz_page, page_idx, filename, ts, doc_id
        )
        all_caption_docs.extend(cap_docs)

        if not sig_images and not fig_meta_list:
            continue  # no visual content on this page

        if n_vision >= max_figure_pages:
            continue

        try:
            fig_paths, page_png_path = _save_figure_crops(
                fitz_page, page_idx, filename, fig_meta_list
            )

            # Build image chunk docs (figure_image + page_screenshot)
            img_docs = _create_image_chunk_docs(
                fig_meta_list, fig_paths, page_png_path, page_idx, filename, ts, doc_id
            )
            all_image_doc_tuples.extend(img_docs)

            # Optionally get a vision description for RAG retrieval (text chunk)
            if vision_llm and sig_images:
                img_b64 = base64.b64encode(page_png_path.read_bytes()).decode()
                desc = _describe_page_vision(img_b64, page_idx, vision_llm)
                if desc:
                    vision_docs.append(Document(
                        page_content=desc,
                        metadata={
                            "document_id":        doc_id,
                            "source_file":        filename,
                            "source_type":        "uploaded",
                            "upload_timestamp":   ts,
                            "chunk_type":         "vision_description",
                            "extraction_method":  "gemini_vision",
                            "embedding_modality": "text",
                            "page_number":        page_idx + 1,
                            "figure_index":       "",
                            "caption":            "",
                            "nearby_text":        "",
                            "image_path":         str(page_png_path),
                            "figure_url":         f"/figures/{filename}/{page_png_path.name}",
                            "section":            "",
                        },
                    ))
                    n_vision += 1
                    logger.info("PDF '%s': vision described page %d", filename, page_idx + 1)
        except Exception as exc:
            logger.warning(
                "PDF '%s' page %d: figure processing failed: %s", filename, page_idx + 1, exc
            )

    fitz_doc.close()

    if vision_docs:
        logger.info("PDF '%s': added %d vision description chunks", filename, len(vision_docs))
    if all_caption_docs:
        logger.info("PDF '%s': added %d figure/table caption chunks", filename, len(all_caption_docs))
    if all_image_doc_tuples:
        logger.info("PDF '%s': produced %d image chunk tuples for image embedding", filename, len(all_image_doc_tuples))

    # --- Build text chunks from extracted page text ---
    md_text = "\n\n".join(page_texts)

    md_text = _IMAGE_MARKER_RE.sub(
        "[Figure: embedded image — see caption and surrounding context]",
        md_text,
    )

    n_table_rows = len(re.findall(r"^\|", md_text, re.MULTILINE))
    if n_table_rows:
        logger.info("PDF '%s': %d table rows in extracted markdown", filename, n_table_rows)

    # Pass 1: split at markdown section headers
    header_splitter = MarkdownHeaderTextSplitter(_HEADERS_TO_SPLIT_ON, strip_headers=False)
    header_chunks = header_splitter.split_text(md_text)

    # Pass 2: subdivide chunks that are still too large
    docs = _CHAR_SPLITTER.split_documents(header_chunks)

    for doc in docs:
        doc.metadata.update({
            "document_id":        doc_id,
            "source_file":        filename,
            "source_type":        "uploaded",
            "upload_timestamp":   ts,
            "chunk_type":         "text",
            "extraction_method":  method,
            "embedding_modality": "text",
            "page_number":        doc.metadata.get("page", 0),
            "figure_index":       "",
            "caption":            "",
            "nearby_text":        "",
            "image_path":         "",
            "figure_url":         "",
            "section":            " > ".join(filter(None, [
                                      doc.metadata.get("h1", ""),
                                      doc.metadata.get("h2", ""),
                                      doc.metadata.get("h3", ""),
                                  ])),
        })

    # Dedicated table chunks
    table_docs = _extract_table_documents(md_text, filename, ts, doc_id)
    if table_docs:
        logger.info("PDF '%s': added %d dedicated table chunks", filename, len(table_docs))
        docs.extend(table_docs)

    docs.extend(all_caption_docs)
    docs.extend(vision_docs)

    logger.info(
        "PDF '%s': produced %d total text chunks (%d text, %d table, %d caption, %d vision)",
        filename, len(docs),
        len(docs) - len(table_docs) - len(all_caption_docs) - len(vision_docs),
        len(table_docs), len(all_caption_docs), len(vision_docs),
    )

    return docs, all_image_doc_tuples
