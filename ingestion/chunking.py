"""
ingestion/chunking.py

Code-aware chunking: instead of splitting every file with the same
character-count rule, we use LangChain's language-aware recursive splitter
so that, wherever possible, chunk boundaries fall on function/class/block
edges rather than mid-statement. Falls back to plain recursive splitting
for languages LangChain doesn't have a dedicated separator set for.
"""

from __future__ import annotations

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from ingestion.ingest import Document

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Map our internal language tags to LangChain's Language enum where one exists.
_LANGCHAIN_LANGUAGE = {
    "python": Language.PYTHON,
    "js": Language.JS,
    "java": Language.JAVA,
    "go": Language.GO,
    "ruby": Language.RUBY,
    "rust": Language.RUST,
    "cpp": Language.CPP,
    "csharp": Language.CSHARP,
    "php": Language.PHP,
    "markdown": Language.MARKDOWN,
}


def _splitter_for(language: str) -> RecursiveCharacterTextSplitter:
    lc_language = _LANGCHAIN_LANGUAGE.get(language)
    if lc_language is not None:
        return RecursiveCharacterTextSplitter.from_language(
            language=lc_language,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    # SQL, YAML, JSON, shell, plain text, etc: no dedicated separator set,
    # fall back to a generic recursive splitter.
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split each Document into overlapping, language-aware chunks.

    Each resulting chunk keeps the parent file's metadata plus a chunk
    index, so downstream citations can point back to "file.py (chunk 3)".
    """
    chunks: list[Document] = []

    for doc in documents:
        splitter = _splitter_for(doc.metadata.get("language", ""))
        pieces = splitter.split_text(doc.content)

        for i, piece in enumerate(pieces):
            chunks.append(
                Document(
                    content=piece,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "chunk_count": len(pieces),
                    },
                )
            )

    return chunks
