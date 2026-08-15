"""SdkRunner — the single gate every model-backed agent call goes through.

Responsibilities (PLAN.md §7):
- **Lockdown**: deny-by-default tool policy scoped to one tier snapshot.
  `Read` is allowed only inside the caller-tier's snapshot dir; `Grep`/
  `Glob` are confined by a PreToolUse hook, NOT by cwd — cwd does not
  constrain an absolute `path` argument, and an `allowed_tools` entry
  with no `(...)` specifier allows the whole tool outright. Everything
  else (Bash, Write, Edit, WebFetch, WebSearch, Task…) is disallowed BY
  NAME so the tools leave context entirely. The PydanticAI/OpenRouter
  path exposes equivalent read-only application tools instead of a raw
  filesystem. `setting_sources=[]` +
  `strict_mcp_config=True` mean nothing auto-loads from ~/.claude or any
  repo .claude/ (grill C4, C12a).
- **Ledger**: an SdkOperation row is written BEFORE the run (`ok=None`)
  and finalized after, so killed runs are never invisible (grill C6).
- **Budgets**: per-kind `max_budget_usd` + `max_turns` + wall-clock
  timeout, plus the daily circuit breaker (grill C5/C16). Claude SDK
  reaps its subprocess; PydanticAI cancels its direct HTTP agent run.
- **Degraded mode**: SDK/API failures map to `ok=False` + `error_class`;
  callers fail fast, never auto-retry chat/assemble (grill C21).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Awaitable

from asgiref.sync import sync_to_async
from django.utils import timezone

from apps.brainconfig import services as config
from apps.events.models import SdkOperation

log = logging.getLogger(__name__)

# Denied by name so their schemas never enter the agent's context (§7).
DENIED_TOOLS = [
    "Bash",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "Task",
    "Agent",
    "TodoWrite",
    "KillShell",
    "BashOutput",
]


class SdkRunnerError(Exception):
    """Base for run-refusals and failures the caller should surface."""


class NotConfigured(SdkRunnerError):
    """No ANTHROPIC_API_KEY set (DB setting or env)."""


class DailyCapExceeded(SdkRunnerError):
    """Circuit breaker: today's summed cost is over DAILY_COST_CAP."""


@dataclass
class RunResult:
    ok: bool
    text: str
    model: str = ""
    duration_ms: int | None = None
    num_turns: int | None = None
    cost_usd: float | None = None
    cost_source: str = ""
    usage: dict = field(default_factory=dict)
    error_class: str = ""
    operation_id: int | None = None
    # Schema-validated object when the run used `output_format` (C9).
    structured_output: object = None
    # Snapshot files the agent actually opened (observed Read tool calls,
    # streaming runs only) — the honest source list for chat (M3.3).
    read_paths: list[str] = field(default_factory=list)


def today_cost_usd() -> float:
    """Sum finalized costs plus reservations held by running operations."""
    from django.db.models import Case, DecimalField, Sum, Value, When
    from django.db.models.functions import Coalesce

    start = timezone.localtime().replace(hour=0, minute=0, second=0, microsecond=0)
    accounted = Case(
        When(
            finished_at__isnull=True,
            then=Coalesce("reserved_cost_usd", Value(Decimal(0))),
        ),
        default=Coalesce("cost_usd", Value(Decimal(0))),
        output_field=DecimalField(max_digits=10, decimal_places=6),
    )
    total = SdkOperation.objects.filter(created_at__gte=start).aggregate(
        s=Sum(accounted)
    )["s"]
    return float(total or 0)


def today_unpriced() -> dict[str, int]:
    """Today's finished runs that burned tokens but produced no cost.

    A run that timed out, or whose client hung up, never receives the
    ResultMessage carrying `total_cost_usd` — so the spend is real and
    `today_cost_usd()` cannot see any of it. The breaker was therefore
    measuring against incomplete data with nothing saying so.

    Reporting the gap is the honest alternative to inventing a number
    for it: pricing is per-model and changes, and a made-up dollar
    figure inside a spend breaker is worse than an admitted blind spot.

    **All four token columns count.** This used to sum `input_tokens` and
    `output_tokens` only, which on this workload is a rounding error:
    every run replays a system prompt and a snapshot through the prompt
    cache, so a real cut-off run measured here recorded `in=2, out=2,
    cache_read=11689, cache_write=2212` — reported as 4 tokens out of
    13,905. A blind-spot figure that understates by three orders of
    magnitude is not an admission, it is the same silence in smaller
    type. The two cache columns are also in the `> 0` filter, since a
    run can be entirely cache reads and would otherwise not be counted
    as a gap at all.
    """
    from django.db.models import Count, Q, Sum

    columns = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
    burned = Q()
    for column in columns:
        burned |= Q(**{f"{column}__gt": 0})

    start = timezone.localtime().replace(hour=0, minute=0, second=0, microsecond=0)
    agg = (
        SdkOperation.objects.filter(
            created_at__gte=start, cost_usd__isnull=True, finished_at__isnull=False
        )
        .filter(burned)
        .aggregate(runs=Count("id"), **{c: Sum(c) for c in columns})
    )
    out = {c: agg[c] or 0 for c in columns}
    out["runs"] = agg["runs"] or 0
    # Precomputed because the template cannot add four numbers, and the
    # gate's warning needs the same figure.
    out["total_tokens"] = sum(out[c] for c in columns)
    return out


#: Input-side counters the Anthropic `message_start` payload carries.
_PARTIAL_INPUT_KEYS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


class _PartialUsage:
    """Token counts accumulated from raw stream events as they arrive.

    `StreamEvent.event` is the raw Anthropic streaming payload, and those
    carry usage even though no `ResultMessage` has been seen yet:
    `message_start` has the turn's input and cache counts, `message_delta`
    carries the running output count for the turn in progress. A run cut
    short by a timeout or a disconnect used to record no usage at all, so
    the ledger showed zero tokens for work that really consumed them.

    Output counts are per-turn and cumulative WITHIN a turn, so each turn
    is banked when the next `message_start` arrives rather than summed as
    it grows. Cost is deliberately not derived from any of this.
    """

    def __init__(self) -> None:
        self.totals = dict.fromkeys(_PARTIAL_INPUT_KEYS, 0)
        self.output_tokens = 0
        self._turn_output = 0

    def observe(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "message_start":
            self._bank_turn()
            usage = ((event.get("message") or {}).get("usage")) or {}
            for key in _PARTIAL_INPUT_KEYS:
                self.totals[key] += int(usage.get(key) or 0)
            self._turn_output = int(usage.get("output_tokens") or 0)
        elif kind == "message_delta":
            usage = event.get("usage") or {}
            if usage.get("output_tokens") is not None:
                self._turn_output = int(usage["output_tokens"])

    def _bank_turn(self) -> None:
        self.output_tokens += self._turn_output
        self._turn_output = 0

    def as_usage(self) -> dict[str, int]:
        """Accumulated counts in the shape `ResultMessage.usage` uses."""
        self._bank_turn()
        out = dict(self.totals)
        out["output_tokens"] = self.output_tokens
        return out if any(out.values()) else {}


def _auth_env(api_key: str) -> dict[str, str]:
    """Map the stored credential to the env var the CLI expects.

    Console API keys (sk-ant-api…) belong in ANTHROPIC_API_KEY. Claude
    subscription tokens from `claude setup-token` (sk-ant-oat…) must go
    in CLAUDE_CODE_OAUTH_TOKEN instead — in the API-key slot the API
    401s them ("Invalid API key"), while the CLI accepts them natively
    through the OAuth path.
    """
    if api_key.startswith("sk-ant-oat"):
        return {"CLAUDE_CODE_OAUTH_TOKEN": api_key}
    return {"ANTHROPIC_API_KEY": api_key}


def _check_gates(*, exempt_daily_cap: bool) -> str:
    api_key = config.anthropic_api_key()
    if not api_key:
        raise NotConfigured("ANTHROPIC_API_KEY is not configured")
    cap = config.daily_cost_cap()  # None = breaker disabled (cap set to 0)
    if not exempt_daily_cap and cap is not None:
        unpriced = today_unpriced()
        if unpriced["runs"]:
            # Not a refusal: this is real spend the breaker is blind to,
            # and the operator has to know the cap is being measured
            # against incomplete data. /ops/logs/ shows the same figure.
            log.warning(
                "sdk runner: %s run(s) today consumed %s tokens that could not "
                "be priced (no ResultMessage) — the daily cap is being checked "
                "against incomplete spend",
                unpriced["runs"],
                unpriced["total_tokens"],
            )
        if today_cost_usd() >= float(cap):
            raise DailyCapExceeded(
                f"daily cost cap {cap} USD reached — raise DAILY_COST_CAP, "
                "set it to 0 to disable the breaker, or wait for tomorrow"
            )
    return api_key


# Tool inputs that name a filesystem location. `Read` uses `file_path`;
# `Grep` and `Glob` use `path`. Anything else here is belt-and-braces for
# tools that are denied by name today but might be allowed later.
_PATH_TOOL_INPUT_KEYS = ("file_path", "path", "notebook_path")


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def tier_path_guard(tier_path: Path):
    """PreToolUse hook: refuse any file tool call resolving outside `tier_path`.

    This is what actually confines `Grep`/`Glob`, and it is not optional.
    `cwd` does not constrain an absolute `path=` argument, and the SDK
    documents that an `allowed_tools` entry with no `(...)` specifier
    ("Grep") allows the whole tool outright — so neither mechanism can stop
    an agent reading a sibling tier. All three tier snapshots live on one
    mounted volume, so an unconfined Grep with `output_mode="content"` is a
    cross-tier read of note bodies. A PreToolUse hook is the SDK's
    documented way to gate every call (see `CanUseToolShadowedWarning`).

    Symlinks resolve before the check, so a symlink inside the snapshot
    pointing out of it is refused too.
    """
    root = tier_path.resolve()

    async def guard(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name", "tool")
        tool_input = input_data.get("tool_input") or {}
        for key in _PATH_TOOL_INPUT_KEYS:
            raw = tool_input.get(key)
            if raw is None or raw == "":
                # No path named. The tool's own default applies and `cwd`
                # is the tier, so there is nothing to contain.
                continue
            if not isinstance(raw, str):
                # Deny, do not skip. This used to `continue`, which means
                # allow — so any shape the SDK might start using for a
                # path (a list of them, most plausibly) would have walked
                # straight past the only check confining Grep/Glob to one
                # tier. The SDK is pinned and bumped deliberately; the
                # failure mode of a bump that changes this shape has to be
                # a visible refusal, not a silent widening.
                log.warning(
                    "tier guard refused %s %s: expected a path string, got %s",
                    tool_name, key, type(raw).__name__,
                )
                return _deny(
                    f"{tool_name}: {key} must be a single path string; "
                    f"got {type(raw).__name__}. Refusing rather than "
                    "skipping the tier check."
                )
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                resolved = candidate.resolve()
            except OSError:
                return _deny(f"{tool_name}: {key} could not be resolved.")
            if resolved != root and root not in resolved.parents:
                log.warning(
                    "tier guard refused %s %s=%r (outside %s)", tool_name, key, raw, root
                )
                return _deny(
                    f"{tool_name} may only touch the caller's tier snapshot. "
                    f"{key} resolves outside it."
                )
        return {}

    return guard


def _snapshot_options(
    tier: str,
    api_key: str,
    kind: str,
    append_prompt: str,
    output_format=None,
    partial_messages: bool = True,
):
    """ClaudeAgentOptions locked to one tier snapshot (§7 'the real config').

    `partial_messages` used to be `streaming`, on only for `stream_agent`,
    because raw stream events are what the chat SSE response is made of.
    But they are also the only place usage appears *before* the
    `ResultMessage` — and a run that times out never gets a
    `ResultMessage`. With it off, a timed-out `assemble-context` recorded
    no cost (expected: nothing can price it) and no tokens either, so it
    was invisible even to `today_unpriced()`, the figure whose whole job
    is to admit what the breaker cannot see. The extra events cost some
    IPC volume on a pipe that is already carrying the same content.
    """
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

    from apps.brain.services import snapshots

    tier_path = snapshots.tier_dir(tier)
    return ClaudeAgentOptions(
        cwd=str(tier_path),
        permission_mode="dontAsk",  # deny anything not pre-approved
        allowed_tools=[f"Read({tier_path.as_posix()}/**)", "Grep", "Glob"],
        # The hook — not cwd, not allowed_tools — is what confines Grep/Glob.
        hooks={
            "PreToolUse": [
                HookMatcher(
                    matcher="Read|Grep|Glob",
                    hooks=[tier_path_guard(tier_path)],
                )
            ]
        },
        disallowed_tools=DENIED_TOOLS,
        setting_sources=[],
        strict_mcp_config=True,
        system_prompt={"type": "preset", "preset": "claude_code", "append": append_prompt},
        model=config.model_for(kind),
        max_turns=config.max_turns(kind),
        max_budget_usd=config.max_budget_usd(kind),
        env=_auth_env(api_key),
        include_partial_messages=partial_messages,
        output_format=output_format,
    )


async def _drain(prompt: str, options, partial: "_PartialUsage | None" = None) -> RunResult:
    """Run one query() to completion and fold it into a RunResult.

    `partial` is filled as raw stream events arrive. It is the caller's
    object, not a return value, because the way this coroutine ends when
    it matters is `asyncio.wait_for` cancelling it — at which point
    nothing it returns is ever seen. `_run_ledgered` reads it afterwards.
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        StreamEvent,
        TextBlock,
        query,
    )

    chunks: list[str] = []
    model = ""
    result: ResultMessage | None = None
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, StreamEvent):
            if partial is not None:
                partial.observe(message.event or {})
        elif isinstance(message, AssistantMessage):
            model = message.model or model
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
        elif isinstance(message, ResultMessage):
            result = message

    if result is None:
        return RunResult(ok=False, text="", error_class="NoResultMessage")
    usage = result.usage or {}
    # ResultMessage has no `model` field; model_usage keys carry the
    # resolved model id(s) — prefer that over the assistant echo.
    if result.model_usage:
        model = ", ".join(sorted(result.model_usage))
    return RunResult(
        ok=not result.is_error,
        text="".join(chunks).strip() or str(result.result or ""),
        model=model,
        duration_ms=result.duration_ms,
        num_turns=result.num_turns,
        cost_usd=result.total_cost_usd,
        usage=usage,
        error_class="" if not result.is_error else (result.subtype or "sdk_error"),
        structured_output=result.structured_output,
    )


async def _aclose(stream) -> None:
    """Close an SDK stream, tolerating one that does not support it."""
    closer = getattr(stream, "aclose", None)
    if closer is None:
        return
    try:
        await closer()
    except Exception:  # pragma: no cover - teardown must not mask the run
        log.debug("sdk runner: closing the stream failed", exc_info=True)


async def _run_ledgered(
    *,
    kind: str,
    prompt: str,
    options,
    prompt_hash_input: str,
    timeout_s: int,
    subject=None,
) -> RunResult:
    """Row-before-run wrapper: SdkOperation is created before the SDK is
    touched and finalized in `finally`, whatever happens. `subject` (a
    model instance, e.g. the Feed being extracted) fills the generic FK."""

    def _create_op():
        extra = {}
        if subject is not None:
            from django.contrib.contenttypes.models import ContentType

            extra = {
                "subject_type": ContentType.objects.get_for_model(subject),
                "subject_id": str(subject.pk),
            }
        return SdkOperation.objects.create(
            kind=kind,
            prompt_hash=hashlib.sha256(prompt_hash_input.encode()).hexdigest(),
            **extra,
        )

    op = await sync_to_async(_create_op)()
    run = RunResult(ok=False, text="", error_class="Unknown")
    partial = _PartialUsage()
    try:
        run = await asyncio.wait_for(_drain(prompt, options, partial), timeout=timeout_s)
    except asyncio.TimeoutError:
        run = RunResult(ok=False, text="", error_class="Timeout")
    except Exception as exc:  # SDK/transport errors → degraded, never raise raw
        log.exception("sdk runner: %s run failed", kind)
        label = exc.__class__.__name__
        if label == "Exception":  # SDK raises bare Exception for CLI errors
            label = f"CliError: {str(exc)[:100]}"
        run = RunResult(ok=False, text="", error_class=label)
    finally:
        # Same rule as the streaming path: a completed ResultMessage is
        # authoritative and must never be overwritten by the partials
        # that led up to it. Cost is still left NULL — pricing is
        # per-model and changes, and an invented dollar figure inside a
        # spend breaker is worse than an admitted blind spot. What this
        # buys is that the blind spot is now *countable*.
        if not run.usage:
            run.usage = partial.as_usage()
        op.finished_at = timezone.now()
        op.ok = run.ok
        op.error_class = run.error_class
        op.model = run.model
        op.duration_ms = run.duration_ms
        op.num_turns = run.num_turns
        op.cost_usd = run.cost_usd
        op.input_tokens = run.usage.get("input_tokens")
        op.output_tokens = run.usage.get("output_tokens")
        op.cache_read_tokens = run.usage.get("cache_read_input_tokens")
        op.cache_write_tokens = run.usage.get("cache_creation_input_tokens")
        await sync_to_async(op.save)()
    run.operation_id = op.id
    return run


async def test_connection_async(
    *, exempt_daily_cap: bool = True, candidate_key: str = ""
) -> RunResult:
    """Minimal SDK ping for the Settings page 'Test connection' button.

    No tools, one turn, public-tier cwd — proves the whole chain works:
    key → CLI subprocess → API → usage accounting. Admin-triggered, so it
    is daily-cap-exempt by default (an operator debugging an outage must
    not be locked out by the breaker they're investigating).

    `candidate_key` probes a credential that is NOT stored: the setup
    wizard passes the pasted value so a failing key can be rejected
    before anything persists it. Without it the probe reads the stored
    key — the Settings-page behaviour.
    """
    if await sync_to_async(config.ai_provider)() == "openrouter":
        from apps.reader.services import openrouter_runner

        return await openrouter_runner.test_connection_async(
            exempt_daily_cap=exempt_daily_cap,
            candidate_key=candidate_key,
        )

    from claude_agent_sdk import ClaudeAgentOptions

    from apps.brain.services import snapshots

    if candidate_key:
        # No stored-key gate to check: nothing is stored yet, and the
        # wizard's probe is cap-exempt like every admin-triggered ping.
        api_key = candidate_key
    else:
        api_key = await sync_to_async(_check_gates)(exempt_daily_cap=exempt_daily_cap)
    model = await sync_to_async(config.model_for)("reader")
    tier_path = snapshots.tier_dir("public")
    cwd = str(tier_path if tier_path.is_dir() else tier_path.parent)
    options = ClaudeAgentOptions(
        cwd=cwd,
        permission_mode="dontAsk",
        allowed_tools=[],
        disallowed_tools=DENIED_TOOLS + ["Read", "Grep", "Glob"],
        setting_sources=[],
        strict_mcp_config=True,
        system_prompt="You are a connectivity probe. Reply with exactly: OK",
        model=model,
        max_turns=1,
        # Raw stream events carry usage before any ResultMessage does, so
        # a ping that times out still records what it burned.
        include_partial_messages=True,
        env={
            **_auth_env(api_key),
            # A ping must fail fast: one retry, short per-request timeout —
            # a bad key surfaces as an auth error, not a 2-minute hang.
            "API_TIMEOUT_MS": "30000",
            "CLAUDE_CODE_MAX_RETRIES": "1",
        },
    )
    prompt = "Reply with exactly: OK"
    return await _run_ledgered(
        kind="test_connection",
        prompt=prompt,
        options=options,
        prompt_hash_input=f"test_connection|{model}|{prompt}",
        timeout_s=90,
    )


def run_sync(coro: Awaitable[RunResult]) -> RunResult:
    """Run an SDK coroutine to completion from sync code — safely.

    A bare `asyncio.run()` breaks inside Django's middleware bridge: the
    vendored stack has async-only middlewares, so even under WSGI every
    sync view runs inside `async_to_sync`, which binds a
    CurrentThreadExecutor to the request thread. The coroutine's
    `sync_to_async(thread_sensitive=True)` hops then try to submit back
    onto that same thread while it's blocked driving the new loop →
    "You cannot submit onto CurrentThreadExecutor from its own thread".
    A fresh thread carries no executor binding, so the hops fall through
    to asgiref's global sync executor as intended.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def test_connection(**kwargs) -> RunResult:
    """Sync facade for classic Django views."""
    return run_sync(test_connection_async(**kwargs))


async def stream_agent(
    *,
    kind: str,
    tier: str,
    prompt: str,
    append_system: str,
    subject=None,
):
    """Tier-locked STREAMING run (M3.1/M3.3): an async generator that
    yields `("delta", text)` as tokens arrive and finally `("result",
    RunResult)` exactly once. Same gates, lockdown, and row-before-run
    ledger as `run_agent_async`; the wall-clock timeout bounds the WAIT
    for each event, not just the gap between events, and the stream is
    closed explicitly so the SDK reaps its subprocess.
    Observed `Read` tool calls land in RunResult.read_paths — the honest
    source list, not agent self-report."""
    if await sync_to_async(config.ai_provider)() == "openrouter":
        from apps.reader.services import openrouter_runner

        async for event in openrouter_runner.stream_agent(
            kind=kind,
            tier=tier,
            prompt=prompt,
            append_system=append_system,
            subject=subject,
        ):
            yield event
        return

    import time as _time

    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        StreamEvent,
        TextBlock,
        ToolUseBlock,
        query,
    )

    api_key = await sync_to_async(_check_gates)(exempt_daily_cap=False)
    options = await sync_to_async(_snapshot_options)(
        tier, api_key, kind, append_system, None, partial_messages=True
    )
    timeout_s = await sync_to_async(config.sdk_timeout_seconds)()
    ledger_kind = "chat" if kind == "reader" else "feed_extraction"

    def _create_op():
        extra = {}
        if subject is not None:
            from django.contrib.contenttypes.models import ContentType

            extra = {
                "subject_type": ContentType.objects.get_for_model(subject),
                "subject_id": str(subject.pk),
            }
        return SdkOperation.objects.create(
            kind=ledger_kind,
            prompt_hash=hashlib.sha256(f"{kind}|{tier}|{append_system}|{prompt}".encode()).hexdigest(),
            **extra,
        )

    op = await sync_to_async(_create_op)()
    run = RunResult(ok=False, text="", error_class="Unknown")
    chunks: list[str] = []
    read_paths: list[str] = []
    model = ""
    partial = _PartialUsage()
    started = _time.monotonic()
    stream = query(prompt=prompt, options=options)
    client_gone = False
    try:
        # Manual iteration with a per-event deadline, not `async for`.
        # The timeout used to be checked only AFTER a message arrived, so
        # the `await` on the next one was unbounded: a CLI that wedged
        # before producing anything — or between two events — hung the SSE
        # response forever. The Send button stayed disabled, the ledger row
        # stayed `ok=None`, and the subprocess kept spending where the
        # daily-cap breaker could not see it. The non-streaming path uses
        # `asyncio.wait_for` and was never exposed to this.
        events = stream.__aiter__()
        while True:
            remaining = timeout_s - (_time.monotonic() - started)
            if remaining <= 0:
                run = RunResult(ok=False, text="".join(chunks), error_class="Timeout")
                break
            try:
                message = await asyncio.wait_for(events.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                run = RunResult(ok=False, text="".join(chunks), error_class="Timeout")
                break
            if isinstance(message, StreamEvent):
                ev = message.event or {}
                # Bank usage as it arrives: a run cut short by a timeout
                # or a disconnect never sees the ResultMessage that
                # carries it, and used to record no tokens at all.
                partial.observe(ev)
                if ev.get("type") == "content_block_delta":
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield ("delta", delta["text"])
                continue
            if isinstance(message, AssistantMessage):
                model = message.model or model
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
                    elif isinstance(block, ToolUseBlock) and block.name == "Read":
                        p = str((block.input or {}).get("file_path") or "")
                        if p:
                            read_paths.append(p)
            elif isinstance(message, ResultMessage):
                usage = message.usage or {}
                if message.model_usage:
                    model = ", ".join(sorted(message.model_usage))
                run = RunResult(
                    ok=not message.is_error,
                    text="".join(chunks).strip() or str(message.result or ""),
                    model=model,
                    duration_ms=message.duration_ms,
                    num_turns=message.num_turns,
                    cost_usd=message.total_cost_usd,
                    usage=usage,
                    error_class="" if not message.is_error else (message.subtype or "sdk_error"),
                )
    except GeneratorExit:
        # Client disconnected mid-stream: finalize the ledger row in the
        # finally below, re-raise, and NEVER yield again.
        client_gone = True
        run = RunResult(ok=False, text="".join(chunks), error_class="ClientDisconnected")
        raise
    except Exception as exc:  # SDK/transport errors → degraded, never raise raw
        log.exception("sdk runner: streaming %s run failed", kind)
        label = exc.__class__.__name__
        if label == "Exception":
            label = f"CliError: {str(exc)[:100]}"
        run = RunResult(ok=False, text="".join(chunks), error_class=label)
    finally:
        # Close the stream so the SDK reaps its subprocess. Abandoning the
        # iterator only closes it whenever the generator is collected,
        # which for a wedged CLI means the process outlives the request.
        # Skipped while unwinding from GeneratorExit: awaiting there is
        # what turns a disconnect into "async generator ignored
        # GeneratorExit", and that teardown closes the stream anyway.
        if not client_gone:
            await _aclose(stream)
        run.read_paths = read_paths
        # Only when the run never produced its own totals — a completed
        # ResultMessage is authoritative and must not be overwritten by
        # the partials that led up to it.
        if not run.usage:
            run.usage = partial.as_usage()
        op.finished_at = timezone.now()
        op.ok = run.ok
        op.error_class = run.error_class
        op.model = run.model
        op.duration_ms = run.duration_ms
        op.num_turns = run.num_turns
        op.cost_usd = run.cost_usd
        op.input_tokens = run.usage.get("input_tokens")
        op.output_tokens = run.usage.get("output_tokens")
        op.cache_read_tokens = run.usage.get("cache_read_input_tokens")
        op.cache_write_tokens = run.usage.get("cache_creation_input_tokens")
        await sync_to_async(op.save)()
    run.operation_id = op.id
    yield ("result", run)


async def run_agent_async(
    *,
    kind: str,
    tier: str,
    prompt: str,
    append_system: str,
    output_format: dict | None = None,
    subject=None,
) -> RunResult:
    """Generic tier-locked agent run — the M2/M3 entry point.

    `kind` is an SDK kind from brainconfig ("reader"/"feeder"); the ledger
    row records the operational kind derived from it. `output_format`
    (Messages-API json_schema shape) makes the run return
    `RunResult.structured_output` (grill C9); `subject` links the ledger
    row to what triggered the run.
    """
    if await sync_to_async(config.ai_provider)() == "openrouter":
        from apps.reader.services import openrouter_runner

        return await openrouter_runner.run_agent_async(
            kind=kind,
            tier=tier,
            prompt=prompt,
            append_system=append_system,
            output_format=output_format,
            subject=subject,
        )

    api_key = await sync_to_async(_check_gates)(exempt_daily_cap=False)
    options = await sync_to_async(_snapshot_options)(
        tier, api_key, kind, append_system, output_format, partial_messages=True
    )
    ledger_kind = "assemble_context" if kind == "reader" else "feed_extraction"
    timeout_s = await sync_to_async(config.sdk_timeout_seconds)()
    return await _run_ledgered(
        kind=ledger_kind,
        prompt=prompt,
        options=options,
        prompt_hash_input=f"{kind}|{tier}|{append_system}|{prompt}",
        timeout_s=timeout_s,
        subject=subject,
    )
