"""
ingestion/ingest.py

Clones a GitHub repository to a local scratch directory and loads every
source file we know how to chunk into a simple in-memory Document list.

We intentionally avoid a heavyweight loader framework here: for a codebase
Q&A agent, controlling exactly which files get read (and skipping vendored
directories, binaries, and lockfiles) matters more than generality.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Extensions we know how to chunk meaningfully. Anything else is skipped.
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "js",
    ".jsx": "js",
    ".ts": "js",
    ".tsx": "js",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".md": "markdown",
    ".sql": "sql",
    ".sh": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
}

# Directories we never want to walk into, regardless of repo.
EXCLUDED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build",
    ".next", ".idea", ".vscode", "vendor", "target", "coverage", ".pytest_cache",
    "site-packages", ".mypy_cache", ".tox",
}

MAX_FILE_BYTES = 400_000  # skip anything absurdly large (generated files, etc.)


class RepoTooLargeError(Exception):
    """Raised when a repo exceeds the configured ingest file-count or
    total-byte limits, so a huge monorepo can't hang /ingest indefinitely."""

    def __init__(self, file_count: int, total_bytes: int, max_files: int | None, max_bytes: int | None):
        self.file_count = file_count
        self.total_bytes = total_bytes
        self.max_files = max_files
        self.max_bytes = max_bytes
        super().__init__(
            f"Repo exceeds ingest limits after {file_count} files ({total_bytes} bytes): "
            f"max_files={max_files}, max_bytes={max_bytes}"
        )


@dataclass
class Document:
    """Minimal document container so we're not coupled to a specific
    LangChain loader's schema."""

    content: str
    metadata: dict = field(default_factory=dict)


def _repo_slug(repo_url: str) -> str:
    """Deterministic, filesystem-safe identifier for a repo URL."""
    digest = hashlib.sha256(repo_url.encode("utf-8")).hexdigest()[:12]
    name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
    return f"{name}-{digest}"


def _is_shallow(repo_path: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repo_path, capture_output=True, text=True,
    )
    return result.stdout.strip() == "true"


def _default_branch(repo_path: Path) -> str | None:
    """Best-effort lookup of origin's default branch from the local ref
    set during clone (no network call)."""
    result = subprocess.run(
        ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
        cwd=repo_path, capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout.strip().rsplit("/", 1)[-1]


CLONE_TIMEOUT_SECONDS = int(os.getenv("INGEST_CLONE_TIMEOUT_SECONDS", "120"))


def clone_repo(
    repo_url: str,
    workdir: str = "./repos",
    ref: str | None = None,
    timeout: int | None = CLONE_TIMEOUT_SECONDS,
) -> Path:
    """Clone (or reuse an already-cloned) repo, return its local path.

    If the repo is already present, we fetch + reset instead of re-cloning,
    so repeated ingestion of the same repo during development is fast.

    Re-fetches unshallow the checkout so previously-recorded commit SHAs
    stay reachable for `git diff` (see diff_changed_files), which a plain
    `--depth 1` re-fetch would otherwise prune out of history.

    `timeout` bounds every network-bound git subprocess call (clone/fetch)
    so a stalled connection can't hang /ingest indefinitely; raises
    subprocess.TimeoutExpired if exceeded. Pass None to disable.
    """
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    dest = workdir_path / _repo_slug(repo_url)

    if dest.exists():
        if _is_shallow(dest):
            subprocess.run(
                ["git", "fetch", "--unshallow"], cwd=dest, check=True, capture_output=True, timeout=timeout
            )
        else:
            subprocess.run(
                ["git", "fetch", "--all"], cwd=dest, check=True, capture_output=True, timeout=timeout
            )

        target = ref or _default_branch(dest) or "HEAD"
        subprocess.run(["git", "reset", "--hard", f"origin/{target}"],
                        cwd=dest, check=False, capture_output=True)
    else:
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        cmd += [repo_url, str(dest)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)

    return dest


def get_current_commit_sha(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def diff_changed_files(repo_path: Path, old_sha: str, new_sha: str) -> list[str]:
    """Repo-relative paths that differ (added, modified, or removed)
    between two commits. Requires both commits to be present locally --
    clone_repo unshallows re-fetched checkouts so a previously-recorded
    old_sha stays reachable."""
    if old_sha == new_sha:
        return []
    result = subprocess.run(
        ["git", "diff", "--name-only", old_sha, new_sha],
        cwd=repo_path, check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def load_documents_for_files(repo_path: Path, relative_paths: list[str]) -> list[Document]:
    """Load specific files (by repo-relative path) as Documents.

    Mirrors load_repo_documents' filtering (supported extensions, excluded
    dirs, size cap, non-empty) but for an explicit file list rather than a
    full directory walk -- used by incremental re-ingestion to re-embed
    only the files a diff says changed. Paths that no longer exist on disk
    (deleted files) are silently skipped; the caller is responsible for
    removing their stale chunks separately.
    """
    documents: list[Document] = []

    for rel_path in relative_paths:
        path_obj = Path(rel_path)
        ext = path_obj.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        if any(part in EXCLUDED_DIRS or part.startswith(".") for part in path_obj.parts[:-1]):
            continue

        fpath = repo_path / rel_path
        try:
            if not fpath.is_file() or fpath.stat().st_size > MAX_FILE_BYTES:
                continue
            text = fpath.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if not text.strip():
            continue

        documents.append(
            Document(
                content=text,
                metadata={
                    "file_path": rel_path,
                    "language": SUPPORTED_EXTENSIONS[ext],
                    "extension": ext,
                },
            )
        )

    return documents


def load_repo_documents(
    repo_path: Path,
    max_files: int | None = None,
    max_bytes: int | None = None,
) -> list[Document]:
    """Walk a cloned repo and return one Document per supported source file.

    If max_files/max_bytes are given, raises RepoTooLargeError as soon as
    the running total would exceed either one, so a huge monorepo can't
    hang this walk (and the /ingest endpoint) indefinitely.
    """
    documents: list[Document] = []
    total_bytes = 0

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]

        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            fpath = Path(root) / fname
            try:
                size = fpath.stat().st_size
                if size > MAX_FILE_BYTES:
                    continue
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            if not text.strip():
                continue

            if max_files is not None and len(documents) + 1 > max_files:
                raise RepoTooLargeError(len(documents), total_bytes, max_files, max_bytes)
            if max_bytes is not None and total_bytes + size > max_bytes:
                raise RepoTooLargeError(len(documents), total_bytes, max_files, max_bytes)

            total_bytes += size
            rel_path = str(fpath.relative_to(repo_path))
            documents.append(
                Document(
                    content=text,
                    metadata={
                        "file_path": rel_path,
                        "language": SUPPORTED_EXTENSIONS[ext],
                        "extension": ext,
                    },
                )
            )

    return documents


def clone_and_load(
    repo_url: str,
    workdir: str = "./repos",
    ref: str | None = None,
    max_files: int | None = None,
    max_bytes: int | None = None,
) -> list[Document]:
    """Convenience wrapper: clone a repo and return its loaded documents."""
    repo_path = clone_repo(repo_url, workdir=workdir, ref=ref)
    return load_repo_documents(repo_path, max_files=max_files, max_bytes=max_bytes)


def purge_repo(repo_url: str, workdir: str = "./repos") -> None:
    """Remove a previously cloned repo (used by tests / cache invalidation)."""
    dest = Path(workdir) / _repo_slug(repo_url)
    if dest.exists():
        shutil.rmtree(dest)
