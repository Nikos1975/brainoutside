"""Ops Settings page (PLAN.md §8).

Session-authed, staff-only, mounted under the admin-panel prefix so the
IP-allowlist middleware and (in prod) the network boundary cover it.
Secrets are write-only: the form never echoes a stored secret — it shows
only set/unset and accepts a new value or an explicit clear.
"""
from __future__ import annotations

from django.contrib import messages
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from apps.core import endpoint_gating
from apps.core import maintenance as maintenance_mode
from apps.core.registry import registry
from apps.core.security.client_ip import client_ip
from apps.reader.services import sdk_runner

from . import services
from .models import AppSetting
from .nav import ops_context

# Display grouping for the Settings page — purely presentational, so it
# lives here rather than on SettingSpec (the registry is what SdkRunner
# and the wizard read; they have no use for page layout). `_save` iterates
# the registry, not this map, so a key missing here still saves — and
# `_sectioned` renders it in a trailing "Other settings" card, so it
# cannot silently drop out of the display path either.
_SECTIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Branding", "", ("APP_NAME",)),
    (
        "OpenRouter",
        "fallback prices are infrastructure-controlled and fail closed",
        (
            "AI_PROVIDER",
            "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL_PUBLIC",
            "OPENROUTER_MODEL_READER",
            "OPENROUTER_MODEL_FEEDER",
            "OPENROUTER_FALLBACK_PRICES",
            "OPENROUTER_CONTEXT_MAX_CHARS",
            "OPENROUTER_MAX_OUTPUT_TOKENS_READER",
            "OPENROUTER_MAX_OUTPUT_TOKENS_FEEDER",
            "OPENROUTER_SITE_URL",
        ),
    ),
    (
        "Claude SDK",
        "",
        (
            "ANTHROPIC_API_KEY",
            "CLAUDE_MODEL_READER",
            "CLAUDE_MODEL_FEEDER",
            "MAX_BUDGET_USD_READER",
            "MAX_BUDGET_USD_FEEDER",
            "MAX_TURNS_READER",
            "MAX_TURNS_FEEDER",
            "SDK_TIMEOUT_SECONDS",
            "DAILY_COST_CAP",
        ),
    ),
    (
        "Brain repo",
        "env/compose wins for these keys",
        ("BRAIN_REPO_URL", "GITHUB_WEBHOOK_SECRET", "BRAIN_GIT_WRITE_PAT"),
    ),
)


def _sectioned(rows: list[dict]) -> list[dict]:
    """Group registry rows into the page's cards, registry order kept."""
    by_key = {row["spec"].key: row for row in rows}
    placed: set[str] = set()
    sections = []
    for title, note, keys in _SECTIONS:
        section_rows = [by_key[k] for k in keys if k in by_key]
        placed.update(k for k in keys if k in by_key)
        if section_rows:
            sections.append({"title": title, "note": note, "rows": section_rows})
    leftovers = [row for row in rows if row["spec"].key not in placed]
    if leftovers:
        sections.append({"title": "Other settings", "note": "", "rows": leftovers})
    return sections


@staff_member_required(login_url="login")
@require_http_methods(["GET", "POST"])
def settings_page(request):
    if request.method == "POST":
        action = request.POST.get("action", "save")
        if action == "test_connection":
            return _test_connection(request)
        if action == "maintenance":
            return _set_maintenance(request)
        if action == "endpoint":
            return _set_endpoint(request)
        _save(request)
        return redirect(request.path)

    rows = []
    db_rows = {r.key: r for r in AppSetting.objects.all()}
    for spec in services.REGISTRY:
        row = db_rows.get(spec.key)
        db_set = bool(row is not None and row.value.strip())
        source = services.source_of(spec.key)
        rows.append(
            {
                "spec": spec,
                "db_set": db_set,
                # Secrets are never echoed; non-secrets show the DB value only
                # (the effective value may come from env — shown separately).
                "db_value": "" if spec.secret else (row.value if row else ""),
                "effective": "••••••••" if spec.secret and services.get(spec.key) else services.get(spec.key),
                "source": source,
                # An `env_wins` key with a stored value that env is beating:
                # the row exists but is inert, and saying so is the whole
                # point of showing provenance at all.
                "shadowed": db_set and source == "env",
                "updated_at": row.updated_at if row else None,
            }
        )
    return render(
        request,
        "ops/settings.html",
        {
            "sections": _sectioned(rows),
            "today_cost": sdk_runner.today_cost_usd(),
            "test_result": request.session.pop("test_result", None),
            "maintenance_on": maintenance_mode.is_enabled(),
            "maintenance_message": maintenance_mode.get_message(),
            "endpoint_rows": _endpoint_rows(),
            **ops_context(request),
        },
    )


def _set_maintenance(request):
    """Flip maintenance mode. The only way to reach the flag.

    It had a store, a cache, a middleware, a branded 503 page and an audit
    call, and no surface anywhere that could set it — so it was off
    forever. Lives on Settings rather than Health because it is a stored
    value, not an action on the world; Health is for re-clone and rotate.

    Turning it ON does not lock the operator out: `/ops/`, the Django
    admin, `LOGIN_URL`, logout and `/setup/` are all on the middleware's
    bypass list, so this page stays reachable to turn it back off.
    """
    enabled = request.POST.get("enabled") == "on"
    raw_message = (request.POST.get("maintenance_message") or "").strip()
    maintenance_mode.set_enabled(
        enabled,
        # Empty means "leave the current banner alone", matching the
        # store's own contract — not "blank the message".
        message=raw_message[:500] or None,
        actor=request.user,
        request_id=getattr(request, "request_id", "") or "",
        ip=client_ip(request),
    )
    if enabled:
        messages.success(
            request,
            "Maintenance mode is ON. Consumers get 503; you keep full access.",
        )
    else:
        messages.success(request, "Maintenance mode is OFF. The server is serving again.")
    return redirect(request.path)


def _endpoint_rows() -> list[dict]:
    """Registered endpoints merged with their persisted disable flags.

    One row per slug, not per (slug, version): `is_disabled` gates on the
    bare slug in both the REST view and the MCP proxy, so disabling
    `get-note` takes every version of it offline — the row says so.

    Flags whose slug is no longer in the registry (the endpoint was
    renamed or removed after the flag was written) still render, marked
    stale: a disabled=True orphan is invisible otherwise, yet springs
    back to life the moment the slug returns.
    """
    flags = {f.slug: f for f in endpoint_gating.list_flags()}
    by_slug: dict[str, list] = {}
    for spec in registry.all():
        by_slug.setdefault(spec.slug, []).append(spec)

    rows = []
    for slug, specs in sorted(by_slug.items()):
        flag = flags.pop(slug, None)
        rows.append(
            {
                "slug": slug,
                "versions": ", ".join(s.version for s in specs),
                "description": specs[0].description,
                "disabled": bool(flag and flag.disabled),
                "reason": flag.reason if flag else "",
                "updated_at": flag.updated_at if flag else None,
                "updated_by": flag.updated_by_email if flag else "",
                "registered": True,
            }
        )
    for slug, flag in sorted(flags.items()):
        rows.append(
            {
                "slug": slug,
                "versions": "",
                "description": "",
                "disabled": flag.disabled,
                "reason": flag.reason,
                "updated_at": flag.updated_at,
                "updated_by": flag.updated_by_email,
                "registered": False,
            }
        )
    return rows


def _set_endpoint(request):
    """Flip the runtime disable flag for one endpoint.

    Same story as maintenance mode one function up: the flag had a DB
    model, a Redis cache, an audit call and enforcement in the REST view
    and the MCP proxy — and `set_disabled` had zero callers, so the only
    way to take a buggy endpoint offline was a Django shell. This is the
    switch. Lives on Settings for the same reason maintenance does: it is
    a stored value, not an action on the world.
    """
    slug = (request.POST.get("slug") or "").strip()
    disable = request.POST.get("disabled") == "1"
    reason = (request.POST.get("reason") or "").strip()

    # Known = registered now, or carrying a flag row from when it was.
    # The second half is what lets an orphaned flag be switched back off;
    # anything else would let a typo mint junk rows.
    known = {spec.slug for spec in registry.all()}
    if slug not in known and not any(f.slug == slug for f in endpoint_gating.list_flags()):
        messages.error(request, f"Unknown endpoint {slug!r}.")
        return redirect(request.path)

    changed = endpoint_gating.set_disabled(
        slug,
        disable,
        reason=reason,
        actor=request.user,
        request_id=getattr(request, "request_id", "") or "",
        ip=client_ip(request),
    )
    if not changed:
        messages.info(request, f"{slug} was already {'disabled' if disable else 'enabled'}.")
    elif disable:
        messages.success(
            request,
            f"{slug} is disabled — REST calls get 503 and the MCP tool is "
            f"hidden. Cached gates catch up within 30 seconds.",
        )
    else:
        messages.success(request, f"{slug} is serving again.")
    return redirect(request.path)


def _save(request) -> None:
    changed: list[str] = []
    rejected: list[str] = []
    for spec in services.REGISTRY:
        clear = request.POST.get(f"clear__{spec.key}") == "on"
        raw = (request.POST.get(f"value__{spec.key}") or "").strip()
        if clear:
            if services.is_db_set(spec.key):
                services.set_value(spec.key, "", actor=request.user)
                changed.append(spec.key)
            continue
        if not raw:
            continue  # blank input = keep current (write-only secrets rely on this)
        if spec.max_len and len(raw) > spec.max_len:
            # Reject rather than truncate: a silently shortened app name
            # looks like the save worked.
            rejected.append(f"{spec.key} (max {spec.max_len} characters)")
            continue
        if spec.clean is not None:
            # The same normalisation the wizard applies — `myname/brain`
            # pasted here used to store verbatim, turn the clone check
            # red, and break the offered repair. Reject with the
            # cleaner's own message rather than store a value some later
            # surface will choke on less legibly.
            try:
                raw = spec.clean(raw)
            except ValueError as exc:
                rejected.append(f"{spec.key} ({exc})")
                continue
        if raw != services.get(spec.key) or not services.is_db_set(spec.key):
            services.set_value(spec.key, raw, actor=request.user)
            changed.append(spec.key)
    if rejected:
        messages.error(request, f"Not saved: {', '.join(rejected)}")
    if changed:
        messages.success(request, f"Saved: {', '.join(changed)}")
    elif not rejected:
        messages.info(request, "Nothing changed.")


def _test_connection(request):
    try:
        run = sdk_runner.test_connection()
    except sdk_runner.SdkRunnerError as exc:
        request.session["test_result"] = {"ok": False, "error": str(exc)}
        return redirect(request.path)
    request.session["test_result"] = {
        "ok": run.ok,
        "model": run.model,
        "duration_ms": run.duration_ms,
        "input_tokens": run.usage.get("input_tokens"),
        "output_tokens": run.usage.get("output_tokens"),
        "cost_usd": run.cost_usd,
        "text": run.text[:200],
        "error": run.error_class,
        "operation_id": run.operation_id,
    }
    return redirect(request.path)
