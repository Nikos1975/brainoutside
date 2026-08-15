"""Structured event log — every dashboard is a query over this table.

Typed rows, not log lines (PLAN.md §3): indexed columns for what the UI
filters by (type, consumer, created_at), JSON `details` for the variable
parts, `entity_ids` for which-notes-were-touched queries.

PRIMARY data: unlike Entity/SyncRun this is not rebuildable from the
repo — it's covered by the DB backup policy (PLAN.md §10).
"""
from __future__ import annotations

from django.db import models


class Event(models.Model):
    TYPES = [
        ("read", "read"),
        ("feed", "feed"),
        ("drift", "drift"),
        ("sync", "sync"),
        ("auth_denied", "auth_denied"),
        ("settings_change", "settings_change"),
        ("degraded", "degraded"),
        # A request died, or a browser reported a CSP violation. Written
        # by `apps.events.sinks.record_error` via the `apps.core.error_hook`
        # bridge. Distinct from `degraded`, which means a subsystem
        # recovered (a feed extraction retried and went on).
        ("error", "error"),
    ]

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    type = models.CharField(max_length=24, choices=TYPES, db_index=True)
    # Nullable: UI/admin actions and system events have no consumer key.
    consumer = models.ForeignKey(
        "api_keys.APIKey", null=True, blank=True, on_delete=models.SET_NULL, related_name="events"
    )
    entity_ids = models.JSONField(default=list, blank=True)
    details = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["type", "created_at"])]

    def __str__(self) -> str:  # pragma: no cover - repr only
        return f"{self.type} @ {self.created_at:%Y-%m-%d %H:%M:%S}"


def emit(type: str, *, consumer=None, entity_ids: list[str] | None = None, **details: object) -> Event:
    """One-liner event writer: `emit("drift", entity_ids=[...], reason=...)`."""
    return Event.objects.create(
        type=type, consumer=consumer, entity_ids=entity_ids or [], details=details
    )


def credential_via(credential) -> dict:
    """Attribution detail for credentials the `consumer` FK cannot hold.

    `Event.consumer` is an FK to `api_keys.APIKey`, so a connector
    (URL-token) call lands with consumer NULL — and the activity feed's
    fallback then renders it as "ops", the operator's own name on a
    stranger's traffic. Spread the result into `emit(...)` alongside
    `consumer=`: `{}` for API keys (and no credential) so the common
    path's details JSON stays clean, `{"via": "connector:<name>"}` for
    URL tokens, the class name for anything future.
    """
    if credential is None:
        return {}
    # Lazy imports: this module is imported app-registry-early, and both
    # feature apps import it back.
    from apps.api_keys.models import APIKey

    if isinstance(credential, APIKey):
        return {}
    from apps.url_mcp_tokens import api as url_tokens

    if url_tokens.is_url_token(credential):
        # "Connector" is the ops UI's name for a URL token.
        return {"via": f"connector:{credential.name or credential.prefix}"}
    return {"via": type(credential).__name__}


class SdkOperation(models.Model):
    """The token ledger — one row per model invocation.

    Written BEFORE the run (`ok=None` = running) and finalized after, so a
    killed worker never produces an invisible spend (PLAN.md §3, grill C6).
    `cost_usd` is provider-reported when available and otherwise derived from
    the configured fallback token schedule. `reserved_cost_usd` makes the
    daily cap concurrency-safe before the provider call; raw token counts
    remain canonical and permit later repricing (grill C23).
    Usage fields stay NULL on errored/killed runs: those runs legitimately
    have no usage to report.
    """

    KINDS = [
        ("test_connection", "test_connection"),
        ("feed_extraction", "feed_extraction"),
        ("assemble_context", "assemble_context"),
        ("chat", "chat"),
    ]

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    kind = models.CharField(max_length=32, choices=KINDS, db_index=True)
    # Filled from the ResultMessage after the run (the SDK resolves aliases).
    model = models.CharField(max_length=64, blank=True, default="")
    # None = still running (or killed before finalize); True/False = outcome.
    ok = models.BooleanField(null=True, blank=True)
    error_class = models.CharField(max_length=128, blank=True, default="")
    input_tokens = models.BigIntegerField(null=True, blank=True)
    output_tokens = models.BigIntegerField(null=True, blank=True)
    cache_read_tokens = models.BigIntegerField(null=True, blank=True)
    cache_write_tokens = models.BigIntegerField(null=True, blank=True)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, null=True, blank=True)
    # Held before a provider call so concurrent workers cannot all pass the
    # daily breaker against the same stale total. Released on finalization.
    reserved_cost_usd = models.DecimalField(
        max_digits=10, decimal_places=6, default=0
    )
    cost_source = models.CharField(max_length=16, blank=True, default="")
    duration_ms = models.BigIntegerField(null=True, blank=True)
    num_turns = models.IntegerField(null=True, blank=True)
    # Hash of the composed prompt (system + user), for reproducibility (C7).
    prompt_hash = models.CharField(max_length=64, blank=True, default="")
    # What triggered the run: a Feed, a ChatMessage… (generic FK per PLAN.md
    # §3; NULL for admin-triggered runs like Test connection).
    subject_type = models.ForeignKey(
        "contenttypes.ContentType", null=True, blank=True, on_delete=models.SET_NULL
    )
    subject_id = models.CharField(max_length=64, blank=True, default="")

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["kind", "created_at"])]

    def __str__(self) -> str:  # pragma: no cover - repr only
        state = {None: "running", True: "ok", False: "error"}[self.ok]
        return f"{self.kind} [{state}] @ {self.created_at:%Y-%m-%d %H:%M:%S}"
