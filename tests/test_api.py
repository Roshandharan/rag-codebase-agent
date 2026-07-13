import subprocess
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.db as db
from api.main import _collection_name, app
from ingestion.ingest import Document, RepoTooLargeError

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    """Point the SQLite registry at a throwaway file per test instead of
    the real ./repo_registry.db, so tests don't share state."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "registry.db"))


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("api.main.get_or_create_vectorstore")
@patch("api.main.load_repo_documents")
@patch("api.main.get_current_commit_sha")
@patch("api.main.clone_repo")
def test_ingest_success(mock_clone_repo, mock_get_sha, mock_load_docs, mock_get_or_create_vectorstore):
    mock_clone_repo.return_value = "/tmp/fake-repo"
    mock_get_sha.return_value = "sha-initial"
    mock_load_docs.return_value = [
        Document(content="print('hi')", metadata={"file_path": "a.py", "language": "python"})
    ]
    mock_get_or_create_vectorstore.return_value = MagicMock()

    resp = client.post("/ingest", json={"repo_url": "https://github.com/example/repo"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["files_loaded"] == 1
    assert body["chunks_created"] >= 1
    assert body["incremental"] is False

    record = db.get_repo("https://github.com/example/repo")
    assert record is not None
    assert record.commit_sha == "sha-initial"


@patch("api.main.get_current_commit_sha")
@patch("api.main.clone_repo")
def test_ingest_no_supported_files_returns_422(mock_clone_repo, mock_get_sha):
    mock_clone_repo.return_value = "/tmp/fake-repo"
    mock_get_sha.return_value = "sha-initial"

    with patch("api.main.load_repo_documents", return_value=[]):
        resp = client.post("/ingest", json={"repo_url": "https://github.com/example/empty-repo"})

    assert resp.status_code == 422


@patch("api.main.clone_repo")
def test_ingest_clone_timeout_returns_504(mock_clone_repo):
    mock_clone_repo.side_effect = subprocess.TimeoutExpired(cmd=["git", "clone"], timeout=120)

    resp = client.post("/ingest", json={"repo_url": "https://github.com/example/slow-repo"})

    assert resp.status_code == 504


@patch("api.main.get_current_commit_sha")
@patch("api.main.clone_repo")
def test_ingest_too_large_returns_413(mock_clone_repo, mock_get_sha):
    mock_clone_repo.return_value = "/tmp/fake-repo"
    mock_get_sha.return_value = "sha-initial"

    with patch(
        "api.main.load_repo_documents",
        side_effect=RepoTooLargeError(file_count=5000, total_bytes=1, max_files=3000, max_bytes=None),
    ):
        resp = client.post("/ingest", json={"repo_url": "https://github.com/example/huge-repo"})

    assert resp.status_code == 413


@patch("api.main.get_current_commit_sha")
@patch("api.main.clone_repo")
def test_ingest_no_changes_skips_reembedding(mock_clone_repo, mock_get_sha):
    repo_url = "https://github.com/example/repo"
    db.upsert_repo(repo_url, _collection_name(repo_url), "sha-same")

    mock_clone_repo.return_value = "/tmp/fake-repo"
    mock_get_sha.return_value = "sha-same"

    with patch("api.main.load_repo_documents") as mock_load, patch(
        "api.main.load_documents_for_files"
    ) as mock_load_diff:
        resp = client.post("/ingest", json={"repo_url": repo_url})

        mock_load.assert_not_called()
        mock_load_diff.assert_not_called()

    body = resp.json()
    assert resp.status_code == 200
    assert body["incremental"] is True
    assert body["files_loaded"] == 0
    assert body["chunks_created"] == 0


@patch("api.main.open_existing_vectorstore")
@patch("api.main.load_documents_for_files")
@patch("api.main.diff_changed_files")
@patch("api.main.get_current_commit_sha")
@patch("api.main.clone_repo")
def test_ingest_incremental_reembeds_only_changed_files(
    mock_clone_repo, mock_get_sha, mock_diff, mock_load_diff, mock_open_vs
):
    repo_url = "https://github.com/example/repo"
    collection_name = _collection_name(repo_url)
    db.upsert_repo(repo_url, collection_name, "sha-old")

    mock_clone_repo.return_value = "/tmp/fake-repo"
    mock_get_sha.return_value = "sha-new"
    mock_diff.return_value = ["changed.py", "removed.py"]
    mock_load_diff.return_value = [
        Document(content="updated content", metadata={"file_path": "changed.py", "language": "python"})
    ]

    fake_vectorstore = MagicMock()
    fake_vectorstore.get.return_value = {"ids": ["id-1", "id-2"]}
    mock_open_vs.return_value = fake_vectorstore

    resp = client.post("/ingest", json={"repo_url": repo_url})

    assert resp.status_code == 200
    body = resp.json()
    assert body["incremental"] is True
    assert body["files_loaded"] == 1
    assert body["chunks_created"] >= 1

    mock_diff.assert_called_once_with("/tmp/fake-repo", "sha-old", "sha-new")
    fake_vectorstore.get.assert_called_once_with(
        where={"file_path": {"$in": ["changed.py", "removed.py"]}}
    )
    fake_vectorstore.delete.assert_called_once_with(ids=["id-1", "id-2"])
    fake_vectorstore.add_texts.assert_called_once()

    record = db.get_repo(repo_url)
    assert record.commit_sha == "sha-new"


def test_ask_without_ingest_returns_409():
    resp = client.post(
        "/ask",
        json={"repo_url": "https://github.com/never/ingested", "question": "what does this do?"},
    )
    assert resp.status_code == 409


@patch("api.main.build_rag_chain")
@patch("api.main.open_existing_vectorstore")
def test_ask_after_ingest_returns_answer(mock_open_vs, mock_build_chain):
    repo_url = "https://github.com/example/repo"
    db.upsert_repo(repo_url, _collection_name(repo_url), "sha-1")

    mock_open_vs.return_value = MagicMock()
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {
        "answer": "It adds two numbers (math.py, chunk 0).",
        "sources": [{"file_path": "math.py", "chunk_index": 0, "language": "python"}],
    }
    mock_build_chain.return_value = fake_chain

    resp = client.post("/ask", json={"repo_url": repo_url, "question": "What does add do?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "It adds two numbers (math.py, chunk 0)."
    assert body["sources"] == [{"file_path": "math.py", "chunk_index": 0, "language": "python"}]
