import logging
import os
import struct as _struct
import uuid as _uuid

import redis as _redis
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore, RedisConfig

logger = logging.getLogger(__name__)

INDEX_NAME = "pharma_ra"
WIKI_INDEX_NAME = "pharma_wiki"

_METADATA_SCHEMA = [
    {"name": "title",             "type": "text"},
    {"name": "authors",           "type": "text"},
    {"name": "year",              "type": "text"},
    {"name": "journal",           "type": "text"},
    {"name": "source_file",       "type": "text"},
    {"name": "source_type",       "type": "text"},
    {"name": "upload_timestamp",  "type": "text"},
    # P1/P2 fields
    {"name": "document_id",       "type": "text"},
    {"name": "chunk_type",        "type": "text"},
    {"name": "extraction_method", "type": "text"},
    {"name": "embedding_modality","type": "text"},
    {"name": "page_number",       "type": "numeric"},
    {"name": "figure_index",      "type": "text"},
    {"name": "caption",           "type": "text"},
    {"name": "nearby_text",       "type": "text"},
    {"name": "image_path",        "type": "text"},
    {"name": "figure_url",        "type": "text"},
    {"name": "section",           "type": "text"},
]


def _get_redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:26379")


class _FixedGoogleEmbeddings:
    """Wrapper around GoogleGenerativeAIEmbeddings that fixes batch embed_documents.

    The underlying API only returns 1 vector per call regardless of batch size.
    This wrapper calls embed_query for each text individually.
    """

    def __init__(self, inner):
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._inner.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


class MultimodalGoogleEmbeddings:
    """
    Calls google.generativeai.embed_content directly, supporting text and (optionally) image.

    Image embedding falls back to caption text embedding if the model rejects image input,
    so indexing never fails due to embedding errors.
    """

    def __init__(self, model: str, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = model if model.startswith("models/") else f"models/{model}"
        self._text_dim: int | None = None  # cached after first text embed

    def embed_image(self, png_bytes: bytes) -> list[float] | None:
        """Embed a PNG image. Returns None if the model rejects image input."""
        import base64
        try:
            result = self._genai.embed_content(
                model=self.model,
                content={"parts": [{"inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(png_bytes).decode(),
                }}]},
            )
            vec = result.get("embedding") or []
            # Verify dimension consistency with text embeddings
            if self._text_dim and len(vec) != self._text_dim:
                logger.warning(
                    "Image embedding dim %d ≠ text dim %d; discarding", len(vec), self._text_dim
                )
                return None
            return list(vec)
        except Exception as exc:
            logger.warning("Image embedding failed (caption fallback will be used): %s", exc)
            return None

    def embed_query(self, text: str) -> list[float]:
        result = self._genai.embed_content(model=self.model, content=text)
        vec = list(result["embedding"])
        self._text_dim = len(vec)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


def _get_embeddings():
    """Return the appropriate embeddings client.

    Uses GoogleGenerativeAIEmbeddings when OPENAI_BASE_URL points to Google's API,
    otherwise falls back to OpenAIEmbeddings (standard OpenAI or any compatible provider).
    """
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if "google" in base_url.lower():
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        if not model.startswith("models/"):
            model = f"models/{model}"
        inner = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=os.getenv("OPENAI_API_KEY"),
        )
        return _FixedGoogleEmbeddings(inner)
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )


def _drop_index_if_exists(redis_url: str, index_name: str = INDEX_NAME) -> None:
    """Drop existing index and all its documents so rebuild is clean."""
    try:
        r = _redis.from_url(redis_url)
        r.execute_command("FT.DROPINDEX", index_name, "DD")
    except Exception:
        pass  # index didn't exist — fine


def _build_index_for(index_name: str, docs) -> RedisVectorStore:
    redis_url = _get_redis_url()
    _drop_index_if_exists(redis_url, index_name)
    config = RedisConfig(
        index_name=index_name,
        redis_url=redis_url,
        metadata_schema=_METADATA_SCHEMA,
    )
    return RedisVectorStore.from_documents(docs, _get_embeddings(), config=config)


def _load_index_for(index_name: str) -> RedisVectorStore:
    return RedisVectorStore.from_existing_index(
        index_name,
        _get_embeddings(),
        redis_url=_get_redis_url(),
    )


def build_index(docs) -> RedisVectorStore:
    """Embed documents and store in Redis pharma_ra. Drops existing index first."""
    return _build_index_for(INDEX_NAME, docs)


def load_index() -> RedisVectorStore:
    """Connect to the existing pharma_ra Redis vector index."""
    return _load_index_for(INDEX_NAME)


def add_documents(docs) -> int:
    """Append documents to the pharma_ra index. Returns the number of docs added."""
    vs = load_index()
    vs.add_documents(docs)
    return len(docs)


def build_wiki_index(docs) -> RedisVectorStore:
    """Embed wiki pages and store in Redis pharma_wiki. Drops existing index first."""
    return _build_index_for(WIKI_INDEX_NAME, docs)


def load_wiki_index() -> RedisVectorStore:
    """Connect to the existing pharma_wiki Redis vector index."""
    return _load_index_for(WIKI_INDEX_NAME)


def add_wiki_documents(docs) -> int:
    """Append wiki documents to the pharma_wiki index. Returns the number of docs added."""
    vs = load_wiki_index()
    vs.add_documents(docs)
    return len(docs)


def add_image_documents(image_doc_tuples: list[tuple]) -> int:
    """
    Embed PNG images with gemini-embedding-2 and store directly in the pharma_ra Redis index.

    Uses MultimodalGoogleEmbeddings.embed_image().  If the model rejects the image, falls back
    to embedding the caption/nearby_text as text so the chunk is still retrievable.
    Only runs when OPENAI_BASE_URL contains "google"; returns 0 otherwise.

    image_doc_tuples: [(Document, png_bytes), ...]  from load_pdf_bytes()
    """
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if "google" not in base_url.lower():
        logger.warning("add_image_documents: non-Google backend; skipping image embedding.")
        return 0

    if not image_doc_tuples:
        return 0

    model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    api_key = os.getenv("OPENAI_API_KEY", "")
    emb = MultimodalGoogleEmbeddings(model=model, api_key=api_key)
    r = _redis.from_url(_get_redis_url())

    # Probe text embedding dimension so we can validate image vectors
    try:
        probe = emb.embed_query("probe")
        text_dim = len(probe)
    except Exception as exc:
        logger.error("add_image_documents: cannot determine embedding dimension: %s", exc)
        return 0

    stored = 0
    for doc, png_bytes in image_doc_tuples:
        vector: list[float] | None = emb.embed_image(png_bytes)

        if vector is None:
            # Fallback: embed caption or nearby_text as a text vector
            fallback_text = (
                doc.metadata.get("caption")
                or doc.metadata.get("nearby_text")
                or doc.page_content
            )
            try:
                vector = emb.embed_query(fallback_text)
                doc.metadata["embedding_modality"] = "text_fallback"
            except Exception as exc:
                logger.warning(
                    "add_image_documents: both image and text fallback failed for %s: %s",
                    doc.metadata.get("image_path", "?"), exc,
                )
                continue

        if len(vector) != text_dim:
            logger.warning(
                "add_image_documents: dim mismatch %d vs %d, skipping %s",
                len(vector), text_dim, doc.metadata.get("image_path", "?"),
            )
            continue

        # Pack as float32 binary blob — matches LangChain-Redis internal format
        vec_bytes = _struct.pack(f"{len(vector)}f", *vector)
        key = f"{INDEX_NAME}:{_uuid.uuid4().hex}"

        mapping: dict = {"content": doc.page_content, "content_vector": vec_bytes}
        for field_def in _METADATA_SCHEMA:
            fname = field_def["name"]
            val = doc.metadata.get(fname, "")
            mapping[fname] = str(val) if val is not None else ""

        r.hset(key, mapping=mapping)
        stored += 1

    logger.info("add_image_documents: stored %d vectors (text_dim=%d) in %s", stored, text_dim, INDEX_NAME)
    return stored


def list_uploaded_files() -> list[str]:
    """Return distinct source_file names for uploaded (non-mock) documents."""
    try:
        r = _redis.from_url(_get_redis_url())
        keys = r.keys(f"{INDEX_NAME}:*")
        seen: set[str] = set()
        for key in keys:
            data = r.hgetall(key)
            source_type = data.get(b"source_type", b"").decode()
            source_file = data.get(b"source_file", b"").decode()
            if source_type == "uploaded" and source_file:
                seen.add(source_file)
        return sorted(seen)
    except Exception:
        return []
