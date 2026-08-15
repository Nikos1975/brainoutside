"""Provider and containment tests for the PydanticAI OpenRouter runner."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic_ai import ToolOutput
from tiered_openrouter import (
    ContextLimitExceeded,
    OpenRouterAgent,
    ScopedMarkdownWorkspace,
    WorkspaceError,
    WorkspaceLimits,
)

from apps.events.models import SdkOperation
from apps.reader.services import openrouter_runner, sdk_runner


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


def test_json_schema_becomes_portable_tool_output():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    output = OpenRouterAgent._output_type(schema)
    assert isinstance(output, ToolOutput)
    assert output.name == "agent_result"


def test_missing_snapshot_returns_a_ledgered_failure(monkeypatch):
    class RunResult:
        def __init__(
            self,
            *,
            ok,
            text,
            error_class,
            read_paths=None,
            duration_ms=None,
        ):
            self.ok = ok
            self.text = text
            self.error_class = error_class
            self.read_paths = read_paths or []
            self.duration_ms = duration_ms
            self.operation_id = None
            self.usage = {}

    op = SimpleNamespace(id=73)
    finished = []

    async def check_gates(**kwargs):
        return "key"

    async def create_operation(*args, **kwargs):
        return op

    async def finish_operation(operation, run):
        finished.append((operation, run))

    def missing_workspace(tier):
        raise openrouter_runner.OpenRouterError(
            f"{tier} snapshot is not available"
        )

    monkeypatch.setattr(openrouter_runner, "_check_gates", check_gates)
    monkeypatch.setattr(
        openrouter_runner,
        "_agent_config",
        lambda *args: SimpleNamespace(max_cost_usd=Decimal("0.50")),
    )
    monkeypatch.setattr(openrouter_runner, "_create_operation", create_operation)
    monkeypatch.setattr(openrouter_runner, "_finish_operation", finish_operation)
    monkeypatch.setattr(openrouter_runner, "_workspace", missing_workspace)
    monkeypatch.setattr(
        openrouter_runner,
        "_sdk",
        lambda: SimpleNamespace(RunResult=RunResult),
    )

    run = asyncio.run(
        openrouter_runner.run_agent_async(
            kind="feeder",
            tier="agents-only",
            prompt="test",
            append_system="",
        )
    )

    assert run.ok is False
    assert run.error_class == (
        "OpenRouterError: agents-only snapshot is not available"
    )
    assert run.operation_id == 73
    assert finished == [(op, run)]


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
    monkeypatch.setattr(
        openrouter_runner.config,
        "openrouter_fallback_prices",
        lambda model: {
            "input_per_million": "0.1",
            "output_per_million": "0.2",
        },
    )

    agent_config = openrouter_runner._agent_config(
        "test-key", "reader", "agents-only", ""
    )

    assert agent_config.max_cost_usd == Decimal("0.25")
    assert agent_config.max_requests == 7
    assert agent_config.max_tool_calls == 21
    assert agent_config.max_output_tokens == 900
    assert agent_config.fallback_prices.input_per_million == Decimal("0.1")


@pytest.mark.django_db(transaction=True)
def test_daily_cap_counts_existing_spend_and_refuses_oversized_reservation(
    monkeypatch,
):
    SdkOperation.objects.create(
        kind="chat",
        prompt_hash="a" * 64,
        finished_at=openrouter_runner.timezone.now(),
        ok=True,
        cost_usd=Decimal("0.80"),
        cost_source="provider",
    )
    monkeypatch.setattr(
        openrouter_runner.config, "daily_cost_cap", lambda: Decimal("1.00")
    )

    with pytest.raises(sdk_runner.DailyCapExceeded, match="cannot reserve"):
        asyncio.run(
            openrouter_runner._create_operation(
                "chat",
                "prompt",
                reserve_cost=Decimal("0.25"),
                enforce_daily_cap=True,
            )
        )

    assert SdkOperation.objects.count() == 1


@pytest.mark.django_db(transaction=True)
def test_running_reservations_are_part_of_the_daily_cap(monkeypatch):
    monkeypatch.setattr(
        openrouter_runner.config, "daily_cost_cap", lambda: Decimal("1.00")
    )
    first = asyncio.run(
        openrouter_runner._create_operation(
            "chat",
            "first",
            reserve_cost=Decimal("0.60"),
            enforce_daily_cap=True,
        )
    )

    with pytest.raises(sdk_runner.DailyCapExceeded):
        asyncio.run(
            openrouter_runner._create_operation(
                "chat",
                "second",
                reserve_cost=Decimal("0.50"),
                enforce_daily_cap=True,
            )
        )

    first.refresh_from_db()
    assert first.finished_at is None
    assert first.reserved_cost_usd == Decimal("0.600000")


@pytest.mark.django_db(transaction=True)
def test_final_cost_replaces_reservation_and_records_source():
    op = asyncio.run(
        openrouter_runner._create_operation(
            "feed_extraction",
            "prompt",
            reserve_cost=Decimal("1.00"),
        )
    )
    run = sdk_runner.RunResult(
        ok=True,
        text="done",
        cost_usd=0.0025,
        cost_source="fallback",
        usage={"input_tokens": 100, "output_tokens": 20},
    )

    asyncio.run(openrouter_runner._finish_operation(op, run))

    op.refresh_from_db()
    assert op.reserved_cost_usd == Decimal("0.000000")
    assert op.cost_usd == Decimal("0.002500")
    assert op.cost_source == "fallback"


@pytest.mark.django_db(transaction=True)
def test_unpriced_failure_consumes_reservation_fail_closed():
    op = asyncio.run(
        openrouter_runner._create_operation(
            "chat",
            "prompt",
            reserve_cost=Decimal("0.50"),
        )
    )
    run = sdk_runner.RunResult(ok=False, text="", error_class="Timeout")

    asyncio.run(openrouter_runner._finish_operation(op, run))

    op.refresh_from_db()
    assert op.reserved_cost_usd == Decimal("0.000000")
    assert op.cost_usd == Decimal("0.500000")
    assert op.cost_source == "reservation"
