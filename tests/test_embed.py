from pathlib import Path

import pytest

from ingestion.ingest import Document


@pytest.fixture
def isolated_stores(tmp_path: Path, monkeypatch):
    """Point Chroma persistence and the embedding cache at a tmp dir so
    tests don't touch (or depend on) the real ./chroma_db."""
    import ingestion.embed as embed_module

    monkeypatch.setattr(embed_module, "PERSIST_DIRECTORY", str(tmp_path / "chroma_db"))
    monkeypatch.setattr(embed_module, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("REDIS_URL", raising=False)
    return embed_module


def test_delete_chunks_for_files_removes_only_matching_paths(isolated_stores):
    embed_module = isolated_stores
    docs = [
        Document(content="a", metadata={"file_path": "a.py", "language": "python"}),
        Document(content="b", metadata={"file_path": "b.py", "language": "python"}),
        Document(content="c", metadata={"file_path": "c.py", "language": "python"}),
    ]
    vectorstore = embed_module.get_or_create_vectorstore(docs, collection_name="test-delete")

    deleted = embed_module.delete_chunks_for_files(vectorstore, ["a.py", "c.py"])

    assert deleted == 2
    remaining = vectorstore.get()
    remaining_paths = {m["file_path"] for m in remaining["metadatas"]}
    assert remaining_paths == {"b.py"}


def test_delete_chunks_for_files_empty_list_is_noop(isolated_stores):
    embed_module = isolated_stores
    docs = [Document(content="a", metadata={"file_path": "a.py", "language": "python"})]
    vectorstore = embed_module.get_or_create_vectorstore(docs, collection_name="test-delete-noop")

    deleted = embed_module.delete_chunks_for_files(vectorstore, [])

    assert deleted == 0
    assert len(vectorstore.get()["ids"]) == 1
