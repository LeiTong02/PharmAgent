import os
import redis as _redis
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore, RedisConfig

INDEX_NAME = "pharma_ra"

_METADATA_SCHEMA = [
    {"name": "title", "type": "text"},
    {"name": "authors", "type": "text"},
    {"name": "year", "type": "text"},
    {"name": "journal", "type": "text"},
    {"name": "source_file", "type": "text"},
    {"name": "source_type", "type": "text"},
    {"name": "upload_timestamp", "type": "text"},
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


def _drop_index_if_exists(redis_url: str) -> None:
    """Drop existing index and all its documents so rebuild is clean."""
    try:
        r = _redis.from_url(redis_url)
        r.execute_command("FT.DROPINDEX", INDEX_NAME, "DD")
    except Exception:
        pass  # index didn't exist — fine


def build_index(docs) -> RedisVectorStore:
    """Embed documents and store in Redis. Drops existing index first."""
    redis_url = _get_redis_url()
    _drop_index_if_exists(redis_url)
    config = RedisConfig(
        index_name=INDEX_NAME,
        redis_url=redis_url,
        metadata_schema=_METADATA_SCHEMA,
    )
    return RedisVectorStore.from_documents(docs, _get_embeddings(), config=config)


def load_index() -> RedisVectorStore:
    """Connect to the existing Redis vector index."""
    return RedisVectorStore.from_existing_index(
        INDEX_NAME,
        _get_embeddings(),
        redis_url=_get_redis_url(),
    )


def add_documents(docs) -> int:
    """Append documents to the existing index. Returns the number of docs added."""
    vs = load_index()
    vs.add_documents(docs)
    return len(docs)


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
