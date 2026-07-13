from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app
from ingestion.ingest import Document

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@patch("api.main.get_or_create_vectorstore")
@patch("api.main.clone_and_load")
def test_ingest_success(mock_clone_and_load, mock_get_or_create_vectorstore):
    mock_clone_and_load.return_value = [
        Document(content="print('hi')", metadata={"file_path": "a.py", "language": "python"})
    ]
    mock_get_or_create_vectorstore.return_value = MagicMock()

    resp = client.post("/ingest", json={"repo_url": "https://github.com/example/repo"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["files_loaded"] == 1
    assert body["chunks_created"] >= 1


@patch("api.main.clone_and_load")
def test_ingest_no_supported_files_returns_422(mock_clone_and_load):
    mock_clone_and_load.return_value = []

    resp = client.post("/ingest", json={"repo_url": "https://github.com/example/empty-repo"})

    assert resp.status_code == 422


def test_ask_without_ingest_returns_409():
    resp = client.post(
        "/ask",
        json={"repo_url": "https://github.com/never/ingested", "question": "what does this do?"},
    )
    assert resp.status_code == 409
