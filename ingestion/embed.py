"""
ingestion/embed.py

Builds (or reopens) a Chroma vector store for a set of chunks, with an
embedding cache in front of a local sentence-transformers model so
re-ingesting the same repo (or overlapping chunks across branches) doesn't
re-pay for embeddings we've already computed.

Embeddings run locally via sentence-transformers (no API key, no external
embedding service) -- Anthropic does not offer an embeddings endpoint, so
this pairs with rag/chain.py's ChatAnthropic for generation.

Cache backend:
  - If REDIS_URL is set, embeddings are cached in Redis (shared across
    processes / restarts / horizontal scaling).
  - Otherwise, falls back to a local on-disk cache under .cache/embeddings
    so local development still benefits from caching without Redis running.
"""

from __future__ import annotations

import os
from functools import lru_cache

# Chroma's anonymized telemetry client is version-mismatched with the pinned
# chromadb release and logs a "Failed to send telemetry event" warning on
# every call -- harmless, but noisy. Must be set before chromadb is imported;
# passing client_settings to Chroma() is not reliable since clients are
# cached by persist path.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from langchain.embeddings import CacheBackedEmbeddings  # noqa: E402
from langchain.storage import LocalFileStore  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402

from ingestion.ingest import Document  # noqa: E402

PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "./.cache/embeddings")


@lru_cache(maxsize=1)
def _build_cached_embedder() -> CacheBackedEmbeddings:
    # Loading the underlying sentence-transformers model is expensive (disk
    # I/O + a real model load into memory) -- memoized so it happens once
    # per process instead of on every /ingest and /ask call, which was both
    # slow and a real contributor to OOM risk on memory-constrained hosts.
    underlying = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        # Imported lazily so Redis is an optional dependency for local dev.
        from langchain_community.storage import RedisStore
        import redis as redis_lib

        store = RedisStore(client=redis_lib.from_url(redis_url))
    else:
        store = LocalFileStore(CACHE_DIR)

    return CacheBackedEmbeddings.from_bytes_store(
        underlying,
        store,
        namespace=underlying.model_name,
        key_encoder="sha256",
    )


def get_or_create_vectorstore(chunks: list[Document], collection_name: str) -> Chroma:
    """Embed (with caching) and upsert chunks into a named Chroma collection."""
    embedder = _build_cached_embedder()

    texts = [c.content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=PERSIST_DIRECTORY,
    )

    if texts:
        vectorstore.add_texts(texts=texts, metadatas=metadatas)

    return vectorstore


def open_existing_vectorstore(collection_name: str) -> Chroma:
    """Reopen a previously built collection without re-embedding anything."""
    embedder = _build_cached_embedder()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=PERSIST_DIRECTORY,
    )


def delete_chunks_for_files(vectorstore: Chroma, file_paths: list[str]) -> int:
    """Delete all chunks belonging to the given file paths from a collection.

    Used before re-embedding a changed file (so stale chunks from its old
    contents don't linger) and for files removed outright between commits.
    langchain-chroma's Chroma.delete() doesn't forward a `where` filter to
    the underlying collection in this version, so we resolve matching ids
    via .get() first.
    """
    if not file_paths:
        return 0

    matches = vectorstore.get(where={"file_path": {"$in": file_paths}})
    ids = matches["ids"]
    if ids:
        vectorstore.delete(ids=ids)
    return len(ids)
