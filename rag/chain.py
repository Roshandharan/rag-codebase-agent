"""
rag/chain.py

Builds the retrieval + generation chain used to answer questions about an
ingested codebase. Retrieval uses MMR (Maximal Marginal Relevance) rather
than plain top-k similarity: codebases have a lot of near-duplicate chunks
(similar imports, boilerplate, repeated patterns), and MMR trades a little
pure similarity for diversity, which noticeably reduces the "all six
retrieved chunks are the same file" failure mode.

The answer prompt requires the model to cite the file path (and chunk
index) for every claim it makes, which is what lets the UI render
clickable/inspectable source citations instead of an unverifiable answer.
"""

from __future__ import annotations

import logging
import os
import time

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

logger = logging.getLogger("rag-agent.retrieval")

SYSTEM_PROMPT = """You are a senior engineer answering questions about a specific codebase.
Use ONLY the provided context chunks to answer -- do not rely on prior knowledge of \
similarly named libraries or frameworks unless the context confirms the codebase uses them.

Rules:
1. Every factual claim about the code must be followed by a citation in the \
   form (file_path, chunk N).
2. If the context does not contain enough information to answer, say so \
   explicitly rather than guessing.
3. Prefer quoting short, relevant snippets over paraphrasing large sections.
4. If multiple files are relevant, structure the answer by file.

Context:
{context}
"""

USER_PROMPT = "Question: {question}"


def _format_context(chunks) -> str:
    blocks = []
    for c in chunks:
        path = c.metadata.get("file_path", "unknown")
        idx = c.metadata.get("chunk_index", 0)
        blocks.append(f"--- {path} (chunk {idx}) ---\n{c.page_content}")
    return "\n\n".join(blocks)


def build_rag_chain(vectorstore: Chroma, k: int = 6, fetch_k: int = 20, lambda_mult: float = 0.75):
    """Return a Runnable that takes {"question": str} and returns
    {"answer": str, "sources": list[dict]}.

    Defaults (k=6, fetch_k=20, lambda_mult=0.75) come from
    scripts/eval_retrieval.py's hit-rate@k sweep against pallets/flask:
    k=4 was consistently worse (70% hit-rate across 3 runs) than k=6+
    (80-90%), and lambda_mult=0.75 was the only config that held 90% on
    every run -- lambda_mult=0.5 dipped to 80% once. See README for the
    full sweep results.
    """
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )

    # Opus 4.8 rejects non-default sampling params (temperature/top_p/top_k) --
    # steer determinism via the citation-required prompt instead.
    llm = ChatAnthropic(model=os.getenv("CHAT_MODEL", "claude-opus-4-8"))

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", USER_PROMPT)]
    )
    generation_chain = prompt | llm | StrOutputParser()

    def _retrieve_and_answer(inputs: dict) -> dict:
        question = inputs["question"]

        retrieval_start = time.perf_counter()
        docs = retriever.invoke(question)
        retrieval_seconds = time.perf_counter() - retrieval_start
        context = _format_context(docs)

        generation_start = time.perf_counter()
        answer = generation_chain.invoke({"context": context, "question": question})
        generation_seconds = time.perf_counter() - generation_start

        sources = [
            {
                "file_path": d.metadata.get("file_path"),
                "chunk_index": d.metadata.get("chunk_index"),
                "language": d.metadata.get("language"),
            }
            for d in docs
        ]

        logger.info(
            "retrieved %d chunks in %.3fs, generated answer in %.3fs",
            len(docs),
            retrieval_seconds,
            generation_seconds,
            extra={
                "event": "rag.retrieve",
                "k": k,
                "fetch_k": fetch_k,
                "chunks_retrieved": len(docs),
                "retrieved_files": [s["file_path"] for s in sources],
                "retrieval_seconds": round(retrieval_seconds, 4),
                "generation_seconds": round(generation_seconds, 4),
            },
        )

        return {"answer": answer, "sources": sources}

    return RunnableLambda(_retrieve_and_answer)
