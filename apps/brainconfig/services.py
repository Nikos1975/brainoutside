"""Typed accessors over the encrypted AppSetting store.

Resolution order for every key: DB setting → env var of the same name →
registry default. Empty strings read as unset at every layer (Coolify
injects compose ``${VAR}`` as "" — never distinguish unset/empty).

The registry below is the single source of truth for what the Settings
page renders and what SdkRunner reads (PLAN.md §3 `apps/brainconfig`).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable, Optional

from django.conf import settings as dj_settings

from apps.events.models import emit

from .models import AppSetting

# SDK run kinds that carry their own model/budget knobs. `test_connection`
# deliberately reuses the reader settings — it is a ping, not a kind.
SDK_KINDS = ("reader", "feeder")


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    help: str
    default: str = ""
    secret: bool = False  # write-only in the UI; only "set/unset" is shown
    #: Flip the usual precedence for this key: env beats the stored value
    #: instead of losing to it. For anything an operator may pin in
    #: compose/Coolify — the repo URL, the webhook secret, git credentials
    #: — infrastructure-as-code has to win, or a UI edit would silently
    #: diverge from the declared deployment. Declared here rather than
    #: handled at each call site so the policy is visible in one table.
    #: The Settings page shows which source is live either way.
    env_wins: bool = False
    #: Reject longer values at save time; 0 means unbounded. Only set on
    #: free-text keys that land somewhere with a layout to break — the
    #: value column is a TextField, so nothing else stops a 5 KB paste.
    max_len: int = 0
    #: Normalise/validate a pasted value before it stores: takes the
    #: stripped input, returns the canonical form, raises ValueError with
    #: an operator-facing message. The Settings page enforces it; where
    #: the wizard validates the same key, the spec points at the SAME
    #: function (the repo URL uses `setup_services.normalise_repo_input`)
    #: so the two surfaces cannot drift. None = store as typed.
    clean: Optional[Callable[[str], str]] = None


def _clean_repo_url(raw: str) -> str:
    """The wizard's own normaliser — one definition, two surfaces.

    Before this, /ops/settings/ stored `myname/brain` verbatim: the
    clone check went red and the offered repair failed against a URL
    git cannot parse, while the wizard would have normalised the same
    paste to `git@github.com:myname/brain.git`.
    """
    from apps.brainconfig import setup_services

    try:
        return setup_services.normalise_repo_input(raw)
    except setup_services.SetupError as exc:
        raise ValueError(str(exc)) from exc


def _clean_decimal(raw: str) -> str:
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise ValueError("must be a number, like 10.00") from None
    if not value.is_finite():
        raise ValueError("must be a finite number — nan/inf is not a threshold")
    return str(value)


def _clean_positive_int(raw: str) -> str:
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("must be a whole number") from None
    if value < 1:
        raise ValueError("must be at least 1")
    return str(value)


def _clean_ai_provider(raw: str) -> str:
    value = raw.strip().lower()
    if value not in {"claude", "openrouter"}:
        raise ValueError("must be either claude or openrouter")
    return value


REGISTRY: tuple[SettingSpec, ...] = (
    SettingSpec(
        "APP_NAME",
        "App name",
        "Branding in the browser tab, sidebar, landing page, login and "
        "error pages. Applies on the next page load. The OpenAPI title "
        "and the MCP server name are read once per process, so those "
        "follow after a restart.",
        # Boot-time value = env/.env, falling back to the built-in name.
        # Frozen at import on purpose: that is exactly what an env var
        # means, and it keeps the Settings page honest about the value
        # in force when nothing is stored.
        default=dj_settings.APP_NAME,
        max_len=60,
    ),
    SettingSpec(
        "ANTHROPIC_API_KEY",
        "Anthropic credential",
        "Either a workspace-scoped API key (sk-ant-api…, console spend "
        "limit per PLAN.md §9) or a Claude subscription token from "
        "`claude setup-token` (sk-ant-oat…) — detected automatically.",
        secret=True,
    ),
    SettingSpec(
        "AI_PROVIDER",
        "AI provider",
        "Runtime used for reader, chat and feed extraction. Use openrouter "
        "for the native multi-model path; claude keeps the pinned Agent SDK.",
        default="claude",
        clean=_clean_ai_provider,
    ),
    SettingSpec(
        "OPENROUTER_API_KEY",
        "OpenRouter API key",
        "Secret used only by the native OpenRouter runner.",
        secret=True,
    ),
    SettingSpec(
        "OPENROUTER_MODEL_READER",
        "OpenRouter reader model",
        "Model for agents-only/private reading and chat.",
        default="deepseek/deepseek-v4-flash-0731",
    ),
    SettingSpec(
        "OPENROUTER_MODEL_FEEDER",
        "OpenRouter feeder model",
        "Model for feed extraction.",
        default="deepseek/deepseek-v4-flash-0731",
    ),
    SettingSpec(
        "OPENROUTER_MODEL_PUBLIC",
        "OpenRouter public model",
        "Optional public-tier reader model. The default free Laguna endpoint "
        "may use inputs and outputs for training, so it is never selected "
        "for agents-only or private requests.",
        default="poolside/laguna-s-2.1:free",
    ),
    SettingSpec(
        "OPENROUTER_CONTEXT_MAX_CHARS",
        "OpenRouter context ceiling",
        "Maximum cumulative characters returned by snapshot tools in one "
        "agent run. The run fails rather than silently truncating knowledge.",
        default="600000",
        clean=_clean_positive_int,
    ),
    SettingSpec(
        "OPENROUTER_MAX_OUTPUT_TOKENS_READER",
        "OpenRouter reader output tokens",
        "Maximum generated tokens for reader and chat requests.",
        default="4096",
        clean=_clean_positive_int,
    ),
    SettingSpec(
        "OPENROUTER_MAX_OUTPUT_TOKENS_FEEDER",
        "OpenRouter feeder output tokens",
        "Maximum generated tokens for extraction proposals.",
        default="16384",
        clean=_clean_positive_int,
    ),
    SettingSpec(
        "OPENROUTER_SITE_URL",
        "OpenRouter site URL",
        "Optional HTTP-Referer attribution header, for example the public "
        "BrainOutside URL.",
        default="",
    ),
    SettingSpec(
        "CLAUDE_MODEL_READER",
        "Reader model",
        "Model for assemble-context / chat. Retrieval defaults to a "
        "Sonnet-class model (PLAN.md §7).",
        default="claude-sonnet-5",
    ),
    SettingSpec(
        "CLAUDE_MODEL_FEEDER",
        "Feeder model",
        "Model for feed extraction — may be a bigger model than the reader.",
        default="claude-opus-5",
    ),
    SettingSpec(
        "MAX_BUDGET_USD_READER",
        "Reader budget (USD/run)",
        "Soft per-run cap, checked between turns by the SDK.",
        default="0.50",
        clean=_clean_decimal,
    ),
    SettingSpec(
        "MAX_BUDGET_USD_FEEDER",
        "Feeder budget (USD/run)",
        "Soft per-run cap for feed extraction.",
        default="1.00",
        clean=_clean_decimal,
    ),
    SettingSpec(
        "MAX_TURNS_READER",
        "Reader max turns",
        "Hard cap on agentic turns per reader run.",
        default="15",
        clean=_clean_positive_int,
    ),
    SettingSpec(
        "MAX_TURNS_FEEDER",
        "Feeder max turns",
        "Hard cap on agentic turns per feeder run.",
        default="25",
        clean=_clean_positive_int,
    ),
    SettingSpec(
        "SDK_TIMEOUT_SECONDS",
        "Run timeout (seconds)",
        "Wall-clock ceiling per SDK run; the subprocess is killed past it.",
        default="300",
        clean=_clean_positive_int,
    ),
    SettingSpec(
        "DAILY_COST_CAP",
        "Daily cost cap (USD)",
        "Circuit breaker: SdkRunner refuses to start once today's summed "
        "cost exceeds this (admin-exemptable, grill C16). Set 0 to "
        "disable — reasonable on subscription auth, where cost figures "
        "are synthetic CLI estimates, not billed spend. Clearing restores "
        "the default.",
        default="10.00",
        clean=_clean_decimal,
    ),
    SettingSpec(
        "BRAIN_REPO_URL",
        "Brain repo URL",
        "Git remote of the mind. Env/compose value wins when set. Changing "
        "this does NOT move the clone on disk — the server refuses to boot "
        "on a mismatch until the clone is replaced.",
        env_wins=True,
        clean=_clean_repo_url,
    ),
    SettingSpec(
        "GITHUB_WEBHOOK_SECRET",
        "GitHub webhook secret",
        "HMAC secret for POST /webhooks/github. Empty = webhook disabled "
        "(pushes are still picked up by the periodic pull, just slower).",
        secret=True,
        env_wins=True,
    ),
    SettingSpec(
        "BRAIN_GIT_WRITE_PAT",
        "Git write credential",
        "Fine-grained GitHub PAT with Contents: read+write on the brain "
        "repo only — what approvals push with. A mounted file "
        "(BRAIN_GIT_WRITE_PAT_PATH) or env var wins over this, and keeps "
        "the credential out of the database; see docs/SECURITY.md.",
        secret=True,
        env_wins=True,
    ),
)

_SPECS = {s.key: s for s in REGISTRY}


def get(key: str) -> str:
    """Effective value. "" reads as unset at every layer.

    Normally DB → env → default, so a value set in the browser sticks.
    For `env_wins` keys the first two swap: env → DB → default.
    """
    spec = _SPECS.get(key)
    env_val = (os.environ.get(key) or "").strip()
    if spec is not None and spec.env_wins and env_val:
        return env_val
    row = AppSetting.objects.filter(key=key).first()
    if row is not None and row.value.strip():
        return row.value.strip()
    if env_val:
        return env_val
    return spec.default if spec else ""


def source_of(key: str) -> str:
    """Where `get(key)` is currently reading from: env | db | default | unset."""
    spec = _SPECS.get(key)
    env_val = (os.environ.get(key) or "").strip()
    db_set = is_db_set(key)
    if spec is not None and spec.env_wins and env_val:
        return "env"
    if db_set:
        return "db"
    if env_val:
        return "env"
    return "default" if (spec and spec.default) else "unset"


def is_db_set(key: str) -> bool:
    row = AppSetting.objects.filter(key=key).first()
    return bool(row is not None and row.value.strip())


def set_value(key: str, value: str, *, actor=None) -> None:
    """Persist (encrypted) and emit a `settings_change` event.

    The event never carries the value — only which key changed (secrets!).
    """
    row, _ = AppSetting.objects.get_or_create(key=key)
    row.value = value.strip()
    row.updated_by = actor
    row.save()
    emit("settings_change", key=key, cleared=not value.strip())


# ---- typed accessors (what SdkRunner consumes) ---------------------------


def app_name() -> str:
    """Branding string for templates. Never raises.

    Read on every HTML render — including the 500 and maintenance pages,
    which exist precisely when the database is the broken thing. A failed
    query there must degrade to the boot-time value, not take the branded
    error page down with it (`apps/core/views.py::_render_error`). Hence
    the bare except: OperationalError when Postgres is gone,
    ProgrammingError before the first migrate, InvalidToken swallowed
    upstream in `AppSetting.value`.

    Not cached: `django_redis` is absent from the host test venv, so a
    cache round-trip here would fail every template test. One indexed
    lookup on a single-digit-row table per page render is the cheaper
    trade for a single-user server.
    """
    try:
        return get("APP_NAME") or dj_settings.APP_NAME
    except Exception:  # noqa: BLE001 - see docstring
        return dj_settings.APP_NAME


def anthropic_api_key() -> str:
    return get("ANTHROPIC_API_KEY")


def ai_provider() -> str:
    value = get("AI_PROVIDER").lower()
    return value if value in {"claude", "openrouter"} else "claude"


def openrouter_api_key() -> str:
    return get("OPENROUTER_API_KEY")


def openrouter_model_for(kind: str, tier: str) -> str:
    checked_kind = _kind(kind)
    if checked_kind == "READER" and tier == "public":
        return get("OPENROUTER_MODEL_PUBLIC")
    return get(f"OPENROUTER_MODEL_{checked_kind}")


def openrouter_context_max_chars() -> int:
    try:
        return max(10000, int(get("OPENROUTER_CONTEXT_MAX_CHARS")))
    except ValueError:
        return 600000


def openrouter_max_output_tokens(kind: str) -> int:
    fallback = 4096 if _kind(kind) == "READER" else 16384
    try:
        return max(1, int(get(f"OPENROUTER_MAX_OUTPUT_TOKENS_{_kind(kind)}")))
    except ValueError:
        return fallback


def openrouter_site_url() -> str:
    return get("OPENROUTER_SITE_URL")


def model_for(kind: str) -> str:
    return get(f"CLAUDE_MODEL_{_kind(kind)}")


def max_budget_usd(kind: str) -> float:
    return _as_float(get(f"MAX_BUDGET_USD_{_kind(kind)}"), 0.50)


def max_turns(kind: str) -> int:
    try:
        return max(1, int(get(f"MAX_TURNS_{_kind(kind)}")))
    except ValueError:
        return 15


def sdk_timeout_seconds() -> int:
    try:
        return max(30, int(get("SDK_TIMEOUT_SECONDS")))
    except ValueError:
        return 300


def daily_cost_cap() -> Decimal | None:
    """The breaker threshold, or None when disabled (value ≤ 0).

    Disabling is always explicit: an unparseable, non-finite or CLEARED
    value falls back to the safe default, never to "off"."""
    try:
        cap = Decimal(get("DAILY_COST_CAP"))
    except InvalidOperation:
        return Decimal("10.00")
    if not cap.is_finite():
        # Decimal("nan") constructs without complaint and then raises
        # InvalidOperation on its first comparison — which sat OUTSIDE
        # this guard, so one stored "nan" 500'd every SDK run and the
        # usage dashboard. NaN and ±Infinity are not thresholds.
        return Decimal("10.00")
    return None if cap <= 0 else cap


def _kind(kind: str) -> str:
    k = kind.upper()
    if k not in {s.upper() for s in SDK_KINDS}:
        raise ValueError(f"unknown SDK kind: {kind}")
    return k


def _as_float(raw: str, fallback: float) -> float:
    try:
        value = float(raw)
    except ValueError:
        return fallback
    # float("nan") parses, and NaN compares False with everything — a
    # NaN budget is a soft cap that simply never trips, with no error
    # anywhere. Infinity is "no cap" by stealth. Same rule as
    # daily_cost_cap: fall back to the safe default, never to "off".
    return value if math.isfinite(value) else fallback
