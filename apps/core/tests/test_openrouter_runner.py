"""Provider and containment tests for the PydanticAI OpenRouter runner."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic_ai import NativeOutput
from tiered_openrouter import (
    ContextLimitExceeded,
    OpenRouterAgent,
    ScopedMarkdownWorkspace,
    WorkspaceError,
    WorkspaceLimits,
)

from apps.reader.services import openrouter_runner


def _access(root, limit=10_000):
    return ScopedMarkdownWorkspace(
        root=root.resolve(),
        limits=WorkspaceLimits(max_total_chars=limit),
    )


def test_private_tiers_deny_provider_data_collection():
    for tier in ("agents-only", "private"):
        assert openrouter_runner._provider_policy(tier) == {
            "sort": "price",
            "allow_fallbacks": True,
            "require_parameters": True,
            "data_collection": "deny",
        }


def test_public_tier_explicitly_allows_free_model_policy():
    policy = openrouter_runner._provider_policy("public")
    assert policy["sort"] == "price"
    assert policy["data_collection"] == "allow"
    assert policy["require_parameters"] is True


def test_provider_error_label_fits_ledger_column():
    error = RuntimeError("x" * 500)

    label = openrouter_runner._error_label(error)

    assert label.startswith("RuntimeError: ")
    assert len(label) == 128


def test_json_schema_becomes_strict_native_output():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    output = OpenRouterAgent._output_type(schema)
    assert isinstance(output, NativeOutput)
    assert output.strict is True


def test_snapshot_reads_only_selected_root_and_records_source(tmp_path):
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.mkdir()
    private.mkdir()
    visible = public / "visible.md"
    visible.write_text("visible", encoding="utf-8")
    (private / "secret.md").write_text("never include", encoding="utf-8")
    access = _access(public)

    result = access.read_file("visible.md")

    assert "visible" in result
    assert "never include" not in result
    assert access.source_paths == [str(visible.resolve())]


def test_search_records_only_files_whose_text_entered_context(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    matching = root / "matching.md"
    matching.write_text("Crete\nElounda harbour", encoding="utf-8")
    (root / "other.md").write_text("unrelated", encoding="utf-8")
    access = _access(root)

    result = access.search("elounda")

    assert "matching.md:2" in result
    assert access.source_paths == [str(matching.resolve())]


def test_list_does_not_claim_file_contents_as_sources(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    (root / "INDEX.md").write_text("index", encoding="utf-8")
    (root / "note.md").write_text("note", encoding="utf-8")
    access = _access(root)

    assert "INDEX.md" in access.list_files()
    assert "note.md" in access.list_files()
    assert access.source_paths == []


def test_path_traversal_and_absolute_paths_are_refused(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    (root / "note.md").write_text("visible", encoding="utf-8")
    access = _access(root)

    with pytest.raises(WorkspaceError):
        access.read_file("../secret.md")
    with pytest.raises(WorkspaceError):
        access.read_file(str((root / "note.md").resolve()))


def test_snapshot_file_symlink_escape_is_refused(tmp_path):
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.mkdir()
    private.mkdir()
    secret = private / "secret.md"
    secret.write_text("secret", encoding="utf-8")
    try:
        (public / "escape.md").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this host")

    with pytest.raises(WorkspaceError):
        _access(public).read_file("escape.md")


def test_tool_context_ceiling_is_cumulative(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    (root / "a.md").write_text("a" * 40, encoding="utf-8")
    (root / "b.md").write_text("b" * 40, encoding="utf-8")
    access = _access(root, limit=100)

    access.read_file("a.md")
    with pytest.raises(ContextLimitExceeded):
        access.read_file("b.md")


def test_usage_limits_keep_cost_turn_tool_and_output_caps(monkeypatch):
    monkeypatch.setattr(openrouter_runner.config, "max_turns", lambda kind: 7)
    monkeypatch.setattr(openrouter_runner.config, "max_budget_usd", lambda kind: 0.25)
    monkeypatch.setattr(
        openrouter_runner.config, "openrouter_max_output_tokens", lambda kind: 900
    )
    monkeypatch.setattr(
        openrouter_runner.config,
        "openrouter_model_for",
        lambda kind, tier: "deepseek/test-model",
    )
    monkeypatch.setattr(openrouter_runner.config, "openrouter_site_url", lambda: "")
    monkeypatch.setattr(openrouter_runner.config, "app_name", lambda: "test")
    monkeypatch.setattr(openrouter_runner.config, "sdk_timeout_seconds", lambda: 120)

    agent_config = openrouter_runner._agent_config(
        "test-key", "reader", "agents-only", ""
    )

    assert agent_config.max_cost_usd == Decimal("0.25")
    assert agent_config.max_requests == 7
    assert agent_config.max_tool_calls == 21
    assert agent_config.max_output_tokens == 900
