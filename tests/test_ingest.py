from pathlib import Path

from ingestion.ingest import load_repo_documents


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
