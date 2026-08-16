from pathlib import Path

import pytest
from tiered_openrouter import (
    ContextLimitExceeded,
    ScopedMarkdownWorkspace,
    WorkspaceError,
    WorkspaceLimits,
)


def workspace(root: Path, max_total_chars: int = 10_000):
    return ScopedMarkdownWorkspace(
        root,
        limits=WorkspaceLimits(max_total_chars=max_total_chars),
    )


def test_read_is_confined_and_records_exact_source(tmp_path):
    permitted = tmp_path / "permitted"
    forbidden = tmp_path / "forbidden"
    permitted.mkdir()
    forbidden.mkdir()
    note = permitted / "note.md"
    note.write_text("permitted text", encoding="utf-8")
    (forbidden / "secret.md").write_text("secret text", encoding="utf-8")

    access = workspace(permitted)

    assert "permitted text" in access.read_file("note.md")
    assert access.source_paths == [str(note.resolve())]
    with pytest.raises(WorkspaceError):
        access.read_file("../forbidden/secret.md")


def test_symlink_escape_is_rejected(tmp_path):
    permitted = tmp_path / "permitted"
    forbidden = tmp_path / "forbidden"
    permitted.mkdir()
    forbidden.mkdir()
    secret = forbidden / "secret.md"
    secret.write_text("secret", encoding="utf-8")
    try:
        (permitted / "escape.md").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this host")

    with pytest.raises(WorkspaceError):
        workspace(permitted).read_file("escape.md")


def test_search_records_only_returned_sources(tmp_path):
    root = tmp_path / "brain"
    root.mkdir()
    match = root / "match.md"
    match.write_text("Elounda harbour", encoding="utf-8")
    (root / "other.md").write_text("unrelated", encoding="utf-8")
    access = workspace(root)

    assert "match.md:1" in access.search("elounda")
    assert access.source_paths == [str(match.resolve())]


def test_list_does_not_record_content_sources(tmp_path):
    root = tmp_path / "brain"
    root.mkdir()
    (root / "INDEX.md").write_text("index", encoding="utf-8")
    access = workspace(root)

    assert "INDEX.md" in access.list_files()
    assert access.source_paths == []


def test_context_budget_is_cumulative(tmp_path):
    root = tmp_path / "brain"
    root.mkdir()
    (root / "a.md").write_text("a" * 40, encoding="utf-8")
    (root / "b.md").write_text("b" * 40, encoding="utf-8")
    access = workspace(root, max_total_chars=100)

    access.read_file("a.md")
    with pytest.raises(ContextLimitExceeded):
        access.read_file("b.md")
