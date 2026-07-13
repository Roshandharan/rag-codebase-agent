"""
api/main.py

FastAPI backend for the RAG-powered codebase Q&A agent.

Endpoints:
  POST /ingest  {"repo_url": "...", "ref": "main"}   -> ingest a repo, return chunk stats
  POST /ask     {"repo_url": "...", "question": "..."} -> answer a question about a
                                                            previously-ingested repo
  GET  /health                                        -> liveness check

Design notes:
  - Each ingested repo gets its own Chroma collection (named from a hash
    of its URL), so multiple repos can be ingested and queried without
    their embeddings colliding.
  - Which repos have been ingested (and at which commit) is tracked in a
    small SQLite registry (api/db.py) rather than in-memory, so it
    survives process restarts.
  - A repeat /ingest call diffs the new commit against the last-ingested
    one and only re-chunks/re-embeds the files that actually changed,
    instead of re-ingesting the whole repo every time.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.db import get_repo, upsert_repo
from ingestion.chunking import chunk_documents
from ingestion.embed import delete_chunks_for_files, get_or_create_vectorstore, open_existing_vectorstore
from ingestion.ingest import (
    RepoTooLargeError,
    clone_repo,
    diff_changed_files,
    get_current_commit_sha,
    load_documents_for_files,
    load_repo_documents,
)
from rag.chain import build_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-agent")

app = FastAPI(title="RAG Codebase Q&A Agent", version="0.1.0")

# Guards so a huge monorepo can't hang /ingest indefinitely: cap how much
# of a *full* ingest we'll load (incremental diffs are naturally bounded
# by what changed and aren't subject to these), and bound every git
# clone/fetch subprocess call with a timeout.
MAX_INGEST_FILES = int(os.getenv("MAX_INGEST_FILES", "3000"))
MAX_INGEST_BYTES = int(os.getenv("MAX_INGEST_BYTES", str(200 * 1024 * 1024)))  # 200 MB


class IngestRequest(BaseModel):
    repo_url: str = Field(..., examples=["https://github.com/octocat/Hello-World"])
    ref: str | None = Field(default=None, description="Branch or tag to check out")


class IngestResponse(BaseModel):
    collection_name: str
    files_loaded: int
    chunks_created: int
    seconds_elapsed: float
    incremental: bool = Field(description="True if this was a diff-based re-ingest of an already-known repo")


class AskRequest(BaseModel):
    repo_url: str
    question: str


class Source(BaseModel):
    file_path: str | None
    chunk_index: int | None
    language: str | None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


def _collection_name(repo_url: str) -> str:
    return "repo-" + hashlib.sha256(repo_url.encode("utf-8")).hexdigest()[:16]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    start = time.time()
    collection_name = _collection_name(req.repo_url)
    existing = get_repo(req.repo_url)

    try:
        repo_path = clone_repo(req.repo_url, ref=req.ref)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504, detail=f"Cloning {req.repo_url} timed out after {exc.timeout}s"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface as a clean 400 to the client
        raise HTTPException(status_code=400, detail=f"Failed to clone repo: {exc}") from exc

    new_sha = get_current_commit_sha(repo_path)

    if existing and existing.commit_sha == new_sha:
        elapsed = time.time() - start
        logger.info("No changes for %s (still at %s)", req.repo_url, new_sha[:12])
        return IngestResponse(
            collection_name=collection_name,
            files_loaded=0,
            chunks_created=0,
            seconds_elapsed=round(elapsed, 1),
            incremental=True,
        )

    try:
        if existing:
            changed_paths = diff_changed_files(repo_path, existing.commit_sha, new_sha)
            docs = load_documents_for_files(repo_path, changed_paths)

            vectorstore = open_existing_vectorstore(collection_name)
            # Drop stale chunks for every changed path first -- covers both
            # edited files (about to be re-embedded below) and files that
            # were removed outright (nothing to re-add for those).
            delete_chunks_for_files(vectorstore, changed_paths)

            chunks = chunk_documents(docs)
            if chunks:
                vectorstore.add_texts(
                    texts=[c.content for c in chunks],
                    metadatas=[c.metadata for c in chunks],
                )
            files_loaded, chunks_created = len(docs), len(chunks)
        else:
            docs = load_repo_documents(repo_path, max_files=MAX_INGEST_FILES, max_bytes=MAX_INGEST_BYTES)
            if not docs:
                raise HTTPException(status_code=422, detail="No supported source files found in repo.")

            chunks = chunk_documents(docs)
            get_or_create_vectorstore(chunks, collection_name=collection_name)
            files_loaded, chunks_created = len(docs), len(chunks)
    except RepoTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    upsert_repo(req.repo_url, collection_name, new_sha)

    elapsed = time.time() - start
    logger.info(
        "Ingested %s: %d files, %d chunks in %.1fs (incremental=%s)",
        req.repo_url, files_loaded, chunks_created, elapsed, bool(existing),
    )

    return IngestResponse(
        collection_name=collection_name,
        files_loaded=files_loaded,
        chunks_created=chunks_created,
        seconds_elapsed=round(elapsed, 1),
        incremental=bool(existing),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not get_repo(req.repo_url):
        raise HTTPException(
            status_code=409,
            detail="Repo has not been ingested yet. Call /ingest first.",
        )

    collection_name = _collection_name(req.repo_url)
    vectorstore = open_existing_vectorstore(collection_name)
    chain = build_rag_chain(vectorstore)

    result = chain.invoke({"question": req.question})
    return AskResponse(answer=result["answer"], sources=[Source(**s) for s in result["sources"]])
