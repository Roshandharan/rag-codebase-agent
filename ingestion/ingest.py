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


def clone_repo(repo_url: str, workdir: str = "./repos", ref: str | None = None) -> Path:
    """Clone (or reuse an already-cloned) repo, return its local path.

    If the repo is already present, we fetch + reset instead of re-cloning,
    so repeated ingestion of the same repo during development is fast.
    """
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    dest = workdir_path / _repo_slug(repo_url)

    if dest.exists():
        subprocess.run(["git", "fetch", "--all"], cwd=dest, check=True, capture_output=True)
        target = ref or "HEAD"
        subprocess.run(["git", "reset", "--hard", f"origin/{target}" if ref else target],
                        cwd=dest, check=False, capture_output=True)
    else:
        cmd = ["git", "clone", "--depth", "1"]
        if ref:
            cmd += ["--branch", ref]
        cmd += [repo_url, str(dest)]
        subprocess.run(cmd, check=True, capture_output=True)

    return dest


def load_repo_documents(repo_path: Path) -> list[Document]:
    """Walk a cloned repo and return one Document per supported source file."""
    documents: list[Document] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]

        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            fpath = Path(root) / fname
            try:
                if fpath.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            if not text.strip():
                continue

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


def clone_and_load(repo_url: str, workdir: str = "./repos", ref: str | None = None) -> list[Document]:
    """Convenience wrapper: clone a repo and return its loaded documents."""
    repo_path = clone_repo(repo_url, workdir=workdir, ref=ref)
    return load_repo_documents(repo_path)


def purge_repo(repo_url: str, workdir: str = "./repos") -> None:
    """Remove a previously cloned repo (used by tests / cache invalidation)."""
    dest = Path(workdir) / _repo_slug(repo_url)
    if dest.exists():
        shutil.rmtree(dest)
