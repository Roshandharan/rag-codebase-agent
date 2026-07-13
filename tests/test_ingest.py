import subprocess
from pathlib import Path

import pytest

from ingestion.ingest import (
    RepoTooLargeError,
    clone_repo,
    diff_changed_files,
    get_current_commit_sha,
    load_documents_for_files,
    load_repo_documents,
)


def _run_git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_local_repo(path: Path) -> Path:
    """Create a local (non-GitHub) git repo we can clone via a file:// URL,
    so incremental-ingestion tests don't depend on network access."""
    repo = path / "origin"
    repo.mkdir()
    _run_git("init", "-b", "main", cwd=repo)
    _run_git("config", "user.email", "test@example.com", cwd=repo)
    _run_git("config", "user.name", "Test", cwd=repo)
    (repo / "a.py").write_text("x = 1\n")
    _run_git("add", "a.py", cwd=repo)
    _run_git("commit", "-m", "initial", cwd=repo)
    return repo


def test_load_repo_documents_skips_excluded_dirs(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")

    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("var x = 1;\n")

    docs = load_repo_documents(tmp_path)
    file_paths = {d.metadata["file_path"] for d in docs}

    assert "src/main.py" in file_paths
    assert not any("node_modules" in p for p in file_paths)


def test_load_repo_documents_skips_unsupported_extensions(tmp_path: Path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "notes.py").write_text("# a note\n")

    docs = load_repo_documents(tmp_path)
    file_paths = {d.metadata["file_path"] for d in docs}

    assert "notes.py" in file_paths
    assert "image.png" not in file_paths


def test_load_repo_documents_skips_empty_files(tmp_path: Path):
    (tmp_path / "empty.py").write_text("")
    (tmp_path / "real.py").write_text("x = 1\n")

    docs = load_repo_documents(tmp_path)
    file_paths = {d.metadata["file_path"] for d in docs}

    assert "real.py" in file_paths
    assert "empty.py" not in file_paths


def test_load_repo_documents_raises_when_max_files_exceeded(tmp_path: Path):
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 1\n")
    (tmp_path / "c.py").write_text("c = 1\n")

    with pytest.raises(RepoTooLargeError):
        load_repo_documents(tmp_path, max_files=2)


def test_load_repo_documents_raises_when_max_bytes_exceeded(tmp_path: Path):
    (tmp_path / "a.py").write_text("x" * 100)
    (tmp_path / "b.py").write_text("y" * 100)

    with pytest.raises(RepoTooLargeError):
        load_repo_documents(tmp_path, max_bytes=150)


def test_load_repo_documents_under_limits_succeeds(tmp_path: Path):
    (tmp_path / "a.py").write_text("a = 1\n")

    docs = load_repo_documents(tmp_path, max_files=10, max_bytes=10_000)

    assert len(docs) == 1


def test_load_documents_for_files_loads_only_requested_paths(tmp_path: Path):
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 2\n")

    docs = load_documents_for_files(tmp_path, ["a.py"])

    assert len(docs) == 1
    assert docs[0].metadata["file_path"] == "a.py"


def test_load_documents_for_files_skips_deleted_paths(tmp_path: Path):
    (tmp_path / "a.py").write_text("a = 1\n")

    docs = load_documents_for_files(tmp_path, ["a.py", "gone.py"])

    assert {d.metadata["file_path"] for d in docs} == {"a.py"}


def test_load_documents_for_files_skips_excluded_dirs(tmp_path: Path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("var x = 1;\n")

    docs = load_documents_for_files(tmp_path, ["node_modules/junk.js"])

    assert docs == []


def test_diff_changed_files_detects_modified_file(tmp_path: Path):
    repo = _init_local_repo(tmp_path)
    old_sha = get_current_commit_sha(repo)

    (repo / "a.py").write_text("x = 2\n")
    (repo / "b.py").write_text("y = 1\n")
    _run_git("add", "-A", cwd=repo)
    _run_git("commit", "-m", "update", cwd=repo)
    new_sha = get_current_commit_sha(repo)

    changed = diff_changed_files(repo, old_sha, new_sha)

    assert set(changed) == {"a.py", "b.py"}


def test_diff_changed_files_same_sha_returns_empty(tmp_path: Path):
    repo = _init_local_repo(tmp_path)
    sha = get_current_commit_sha(repo)

    assert diff_changed_files(repo, sha, sha) == []


def test_clone_repo_propagates_timeout_on_clone(tmp_path: Path, monkeypatch):
    import ingestion.ingest as ingest_module

    def fake_run(cmd, **kwargs):
        assert kwargs.get("timeout") == 5
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=5)

    monkeypatch.setattr(ingest_module.subprocess, "run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired):
        ingest_module.clone_repo(
            "https://github.com/example/repo", workdir=str(tmp_path / "workdir"), timeout=5
        )


def test_clone_repo_refetch_without_ref_picks_up_new_commits(tmp_path: Path):
    origin = _init_local_repo(tmp_path)
    workdir = tmp_path / "workdir"

    dest = clone_repo(f"file://{origin}", workdir=str(workdir))
    first_sha = get_current_commit_sha(dest)

    (origin / "a.py").write_text("x = 999\n")
    _run_git("add", "-A", cwd=origin)
    _run_git("commit", "-m", "second commit", cwd=origin)

    # Re-fetch with no explicit ref -- this used to be a no-op bug where
    # the checkout never advanced past the first commit.
    dest_again = clone_repo(f"file://{origin}", workdir=str(workdir))
    second_sha = get_current_commit_sha(dest_again)

    assert first_sha != second_sha
    assert (dest_again / "a.py").read_text() == "x = 999\n"
