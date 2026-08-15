"""Provider-boundary tests for the native OpenRouter runner.

No network calls: these pin privacy routing, schema translation and snapshot
containment independently from any provider endpoint.
"""
from __future__ import annotations

import pytest

from apps.reader.services import openrouter_runner


def test_private_tiers_deny_provider_data_collection():
    for tier in ("agents-only", "private"):
        policy = openrouter_runner._provider_policy(tier, structured=True)
        assert policy == {
            "sort": "price",
            "allow_fallbacks": True,
            "require_parameters": True,
            "data_collection": "deny",
        }


def test_public_tier_explicitly_allows_free_model_policy():
    policy = openrouter_runner._provider_policy("public", structured=False)
    assert policy["sort"] == "price"
    assert policy["data_collection"] == "allow"
    assert policy["require_parameters"] is True


def test_claude_schema_wrapper_is_translated():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    assert openrouter_runner._response_format(
        {"type": "json_schema", "schema": schema}
    ) == {
        "type": "json_schema",
        "json_schema": {
            "name": "brainoutside_result",
            "strict": True,
            "schema": schema,
        },
    }


def test_snapshot_bundle_reads_only_the_selected_tier(tmp_path, monkeypatch):
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.mkdir()
    private.mkdir()
    (public / "INDEX.md").write_text("public index", encoding="utf-8")
    (public / "public.md").write_text("visible", encoding="utf-8")
    (private / "secret.md").write_text("never include", encoding="utf-8")

    from apps.brain.services import snapshots

    monkeypatch.setattr(snapshots, "tier_dir", lambda tier: public)
    monkeypatch.setattr(
        openrouter_runner.config,
        "openrouter_context_max_chars",
        lambda: 10000,
    )

    bundle, paths = openrouter_runner._snapshot_bundle("public")

    assert "public index" in bundle
    assert "visible" in bundle
    assert "never include" not in bundle
    assert paths == [str(public / "INDEX.md"), str(public / "public.md")]


def test_snapshot_bundle_fails_instead_of_truncating(tmp_path, monkeypatch):
    public = tmp_path / "public"
    public.mkdir()
    (public / "large.md").write_text("x" * 1000, encoding="utf-8")

    from apps.brain.services import snapshots

    monkeypatch.setattr(snapshots, "tier_dir", lambda tier: public)
    monkeypatch.setattr(
        openrouter_runner.config,
        "openrouter_context_max_chars",
        lambda: 100,
    )

    with pytest.raises(openrouter_runner.ContextTooLarge):
        openrouter_runner._snapshot_bundle("public")


def test_snapshot_file_symlink_escape_is_refused(tmp_path, monkeypatch):
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

    from apps.brain.services import snapshots

    monkeypatch.setattr(snapshots, "tier_dir", lambda tier: public)
    monkeypatch.setattr(
        openrouter_runner.config,
        "openrouter_context_max_chars",
        lambda: 10000,
    )

    with pytest.raises(openrouter_runner.OpenRouterError):
        openrouter_runner._snapshot_bundle("public")
