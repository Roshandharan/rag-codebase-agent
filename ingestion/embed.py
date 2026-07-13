"""
ingestion/embed.py

Builds (or reopens) a Chroma vector store for a set of chunks, with an
embedding cache in front of the OpenAI embeddings API so re-ingesting the
same repo (or overlapping chunks across branches) doesn't re-pay for
embeddings we've already computed.

Cache backend:
  - If REDIS_URL is set, embeddings are cached in Redis (shared across
    processes / restarts / horizontal scaling).
  - Otherwise, falls back to a local on-disk cache under .cache/embeddings
    so local development still benefits from caching without Redis running.
"""

from __future__ import annotations

import os

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ingestion.ingest import Document

PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR", "./.cache/embeddings")


def _build_cached_embedder() -> CacheBackedEmbeddings:
    underlying = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

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
        namespace=underlying.model,
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
