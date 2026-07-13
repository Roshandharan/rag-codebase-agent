from ingestion.chunking import chunk_documents
from ingestion.ingest import Document


def test_chunk_documents_splits_large_python_file():
    long_function_body = "\n".join(f"    x{i} = {i}" for i in range(400))
    source = f"def big_function():\n{long_function_body}\n    return x0\n"

    doc = Document(content=source, metadata={"file_path": "big.py", "language": "python"})
    chunks = chunk_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["file_path"] == "big.py"
        assert "chunk_index" in chunk.metadata
        assert len(chunk.content) <= 1500 + 200  # chunk_size + some slack for splitter behavior


def test_chunk_documents_keeps_small_file_as_one_chunk():
    doc = Document(content="print('hello world')\n", metadata={"file_path": "hello.py", "language": "python"})
    chunks = chunk_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_count"] == 1


def test_chunk_documents_falls_back_for_unknown_language():
    doc = Document(content="SELECT * FROM patients;\n" * 200, metadata={"file_path": "q.sql", "language": "sql"})
    chunks = chunk_documents([doc])

    assert len(chunks) >= 1
    assert all(c.metadata["file_path"] == "q.sql" for c in chunks)
