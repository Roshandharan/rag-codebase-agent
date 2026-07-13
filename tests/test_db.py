from pathlib import Path

import api.db as db


def test_repo_registry_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "registry.db"))

    assert db.get_repo("https://github.com/example/repo") is None
    assert not db.is_ingested("https://github.com/example/repo")

    db.upsert_repo("https://github.com/example/repo", "repo-abc123", "sha1")
    record = db.get_repo("https://github.com/example/repo")

    assert record is not None
    assert record.collection_name == "repo-abc123"
    assert record.commit_sha == "sha1"
    assert db.is_ingested("https://github.com/example/repo")


def test_repo_registry_upsert_overwrites_existing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "registry.db"))

    db.upsert_repo("https://github.com/example/repo", "repo-abc123", "sha1")
    db.upsert_repo("https://github.com/example/repo", "repo-abc123", "sha2")

    record = db.get_repo("https://github.com/example/repo")
    assert record.commit_sha == "sha2"


def test_repo_registry_persists_across_connections(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "registry.db"))

    db.upsert_repo("https://github.com/example/repo", "repo-abc123", "sha1")

    # Simulate a fresh process by re-reading straight from disk.
    record = db.get_repo("https://github.com/example/repo")
    assert record.repo_url == "https://github.com/example/repo"
