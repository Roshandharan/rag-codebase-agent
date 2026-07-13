"""
api/db.py

Lightweight SQLite-backed registry of ingested repos, replacing the
in-memory set that used to live in api/main.py and lost all state on
every process restart.

One row per repo_url: which Chroma collection it lives in, when it was
last ingested, and the commit SHA it was ingested at (so a repeat
/ingest call can diff against that SHA instead of re-embedding
everything -- see ingestion/ingest.py's diff_changed_files).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

DB_PATH = os.getenv("REGISTRY_DB_PATH", "./repo_registry.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    repo_url TEXT PRIMARY KEY,
    collection_name TEXT NOT NULL,
    last_ingested_at TEXT NOT NULL,
    commit_sha TEXT NOT NULL
);
"""


@dataclass
class RepoRecord:
    repo_url: str
    collection_name: str
    last_ingested_at: str
    commit_sha: str


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_repo(repo_url: str) -> RepoRecord | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT repo_url, collection_name, last_ingested_at, commit_sha "
            "FROM repos WHERE repo_url = ?",
            (repo_url,),
        ).fetchone()
    return RepoRecord(*row) if row else None


def upsert_repo(repo_url: str, collection_name: str, commit_sha: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO repos (repo_url, collection_name, last_ingested_at, commit_sha)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(repo_url) DO UPDATE SET
                collection_name = excluded.collection_name,
                last_ingested_at = excluded.last_ingested_at,
                commit_sha = excluded.commit_sha
            """,
            (repo_url, collection_name, datetime.now(timezone.utc).isoformat(), commit_sha),
        )


def is_ingested(repo_url: str) -> bool:
    return get_repo(repo_url) is not None
