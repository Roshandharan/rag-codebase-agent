"""
api/main.py

FastAPI backend for the RAG-powered codebase Q&A agent.

Endpoints:
  POST /ingest  {"repo_url": "...", "ref": "main"}   -> ingest a repo, return chunk stats
  POST /ask     {"repo_url": "...", "question": "..."} -> answer a question about a
                                                            previously-ingested repo
  GET  /health                                        -> liveness check

Design note: each ingested repo gets its own Chroma collection (named from
a hash of its URL), so multiple repos can be ingested and queried without
their embeddings colliding, and re-ingesting a repo just upserts into the
same collection rather than duplicating it.
"""

from __future__ import annotations

import hashlib
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ingestion.chunking import chunk_documents
from ingestion.embed import get_or_create_vectorstore, open_existing_vectorstore
from ingestion.ingest import clone_and_load
from rag.chain import build_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-agent")

app = FastAPI(title="RAG Codebase Q&A Agent", version="0.1.0")

# In-memory registry of which repos have been ingested this process.
# Swap for a real DB/row in Postgres if you need this to survive restarts
# or run across multiple API replicas.
_INGESTED_REPOS: set[str] = set()


class IngestRequest(BaseModel):
    repo_url: str = Field(..., examples=["https://github.com/octocat/Hello-World"])
    ref: str | None = Field(default=None, description="Branch or tag to check out")


class IngestResponse(BaseModel):
    collection_name: str
    files_loaded: int
    chunks_created: int
    seconds_elapsed: float


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
    try:
        docs = clone_and_load(req.repo_url, ref=req.ref)
    except Exception as exc:  # noqa: BLE001 - surface as a clean 400 to the client
        raise HTTPException(status_code=400, detail=f"Failed to clone/load repo: {exc}") from exc

    if not docs:
        raise HTTPException(status_code=422, detail="No supported source files found in repo.")

    chunks = chunk_documents(docs)
    collection_name = _collection_name(req.repo_url)
    get_or_create_vectorstore(chunks, collection_name=collection_name)
    _INGESTED_REPOS.add(req.repo_url)

    elapsed = time.time() - start
    logger.info("Ingested %s: %d files, %d chunks in %.1fs", req.repo_url, len(docs), len(chunks), elapsed)

    return IngestResponse(
        collection_name=collection_name,
        files_loaded=len(docs),
        chunks_created=len(chunks),
        seconds_elapsed=round(elapsed, 1),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.repo_url not in _INGESTED_REPOS:
        raise HTTPException(
            status_code=409,
            detail="Repo has not been ingested yet in this session. Call /ingest first.",
        )

    collection_name = _collection_name(req.repo_url)
    vectorstore = open_existing_vectorstore(collection_name)
    chain = build_rag_chain(vectorstore)

    result = chain.invoke({"question": req.question})
    return AskResponse(answer=result["answer"], sources=[Source(**s) for s in result["sources"]])
