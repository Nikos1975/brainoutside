"""Reusable PydanticAI/OpenRouter agent loop."""

from __future__ import annotations

import asyncio
import copy
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from openai import AsyncOpenAI
from pydantic_ai import (
    Agent,
    ModelRetry,
    NativeOutput,
    RunContext,
    StructuredDict,
    UsageLimits,
)
from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.run import AgentRunResultEvent

from .workspace import ScopedMarkdownWorkspace, WorkspaceError


def cheapest_provider_policy(
    *, data_collection: Literal["allow", "deny"], zero_data_retention: bool = False
) -> dict:
    """Return an explicit cheapest-compatible OpenRouter routing policy."""
    policy = {
        "sort": "price",
        "allow_fallbacks": True,
        "require_parameters": True,
        "data_collection": data_collection,
    }
    if zero_data_retention:
        policy["zdr"] = True
    return policy


@dataclass(frozen=True)
class AgentConfig:
    """Provider and run controls supplied by the host application."""

    api_key: str
    model: str
    instructions: str
    provider_policy: dict
    base_url: str | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    app_url: str | None = None
    app_title: str | None = None
    max_output_tokens: int = 4096
    max_requests: int = 15
    max_tool_calls: int = 45
    max_cost_usd: Decimal | None = None
    timeout_seconds: int = 300
    tool_retries: int = 1
    output_retries: int = 1

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key is required")
        if not self.model.strip():
            raise ValueError("model is required")
        for name in (
            "max_output_tokens",
            "max_requests",
            "max_tool_calls",
            "timeout_seconds",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_cost_usd is not None and self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive or None")


@dataclass
class AgentResult:
    """Framework-neutral completed-run record."""

    output: str | dict
    model: str
    duration_ms: int
    requests: int
    cost_usd: float | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    source_paths: list[str] = field(default_factory=list)


async def list_workspace_files(
    ctx: RunContext[ScopedMarkdownWorkspace], pattern: str = "*.md"
) -> str:
    """List Markdown paths in the permitted workspace.

    Args:
        pattern: Optional relative glob such as identity/*.md.
    """
    try:
        return await asyncio.to_thread(ctx.deps.list_files, pattern)
    except WorkspaceError as exc:
        raise ModelRetry(str(exc)) from exc


async def search_workspace(
    ctx: RunContext[ScopedMarkdownWorkspace],
    query: str,
    pattern: str = "*.md",
) -> str:
    """Search permitted files and return matching lines with paths and numbers.

    Args:
        query: Case-insensitive text to find.
        pattern: Optional relative Markdown glob limiting the search.
    """
    try:
        return await asyncio.to_thread(ctx.deps.search, query, pattern)
    except WorkspaceError as exc:
        raise ModelRetry(str(exc)) from exc


async def read_workspace_file(
    ctx: RunContext[ScopedMarkdownWorkspace], path: str
) -> str:
    """Read one permitted Markdown file by relative path.

    Args:
        path: Relative path returned by list_workspace_files or search_workspace.
    """
    try:
        return await asyncio.to_thread(ctx.deps.read_file, path)
    except WorkspaceError as exc:
        raise ModelRetry(str(exc)) from exc


def _provider(config: AgentConfig) -> OpenRouterProvider:
    """Build the direct provider or an OpenAI-compatible proxy transport."""
    if not config.base_url:
        return OpenRouterProvider(
            api_key=config.api_key,
            app_url=config.app_url,
            app_title=config.app_title,
        )

    headers = dict(config.default_headers)
    if config.app_url:
        headers.setdefault("HTTP-Referer", config.app_url)
    if config.app_title:
        headers.setdefault("X-Title", config.app_title)
    client = AsyncOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        default_headers=headers or None,
    )
    return OpenRouterProvider(openai_client=client)


class OpenRouterAgent:
    """A bounded, read-only agent that can be embedded in any Python project."""

    def __init__(self, config: AgentConfig):
        self.config = config
        provider = _provider(config)
        settings = OpenRouterModelSettings(
            max_tokens=config.max_output_tokens,
            parallel_tool_calls=False,
            openrouter_provider=config.provider_policy,
            openrouter_usage={"include": True},
        )
        self.model = OpenRouterModel(
            config.model,
            provider=provider,
            settings=settings,
        )

    def _limits(self) -> UsageLimits:
        return UsageLimits(
            cost_limit=self.config.max_cost_usd,
            request_limit=self.config.max_requests,
            tool_calls_limit=self.config.max_tool_calls,
            output_tokens_limit=self.config.max_output_tokens,
        )

    @staticmethod
    def _output_type(output_schema: dict | None):
        if output_schema is None:
            return str
        if not isinstance(output_schema, dict):
            raise TypeError("output_schema must be a JSON-schema object")
        output = StructuredDict(
            copy.deepcopy(output_schema),
            name="agent_result",
            description="Return the result using this exact schema.",
        )
        return NativeOutput(output, strict=True)

    def _agent(self, output_schema: dict | None = None) -> Agent:
        return Agent(
            self.model,
            deps_type=ScopedMarkdownWorkspace,
            instructions=self.config.instructions,
            output_type=self._output_type(output_schema),
            tools=[list_workspace_files, search_workspace, read_workspace_file],
            retries={
                "tools": self.config.tool_retries,
                "output": self.config.output_retries,
            },
        )

    @staticmethod
    def _result(raw, workspace: ScopedMarkdownWorkspace, started: float) -> AgentResult:
        usage = raw.usage
        output = (
            dict(raw.output)
            if isinstance(raw.output, dict)
            else str(raw.output or "").strip()
        )
        return AgentResult(
            output=output,
            model=str(raw.response.model_name or ""),
            duration_ms=int((time.monotonic() - started) * 1000),
            requests=usage.requests,
            cost_usd=float(usage.cost) if usage.cost is not None else None,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            source_paths=workspace.source_paths.copy(),
        )

    async def run(
        self,
        prompt: str,
        workspace: ScopedMarkdownWorkspace,
        *,
        output_schema: dict | None = None,
    ) -> AgentResult:
        """Run to completion with the configured timeout and budgets."""
        started = time.monotonic()
        raw = await asyncio.wait_for(
            self._agent(output_schema).run(
                prompt,
                deps=workspace,
                usage_limits=self._limits(),
            ),
            timeout=self.config.timeout_seconds,
        )
        return self._result(raw, workspace, started)

    async def stream(
        self,
        prompt: str,
        workspace: ScopedMarkdownWorkspace,
    ) -> AsyncIterator[
        tuple[Literal["delta"], str] | tuple[Literal["result"], AgentResult]
    ]:
        """Stream text deltas across a multi-request tool run, then one result."""
        started = time.monotonic()
        async with asyncio.timeout(self.config.timeout_seconds):
            async with self._agent().run_stream_events(
                prompt,
                deps=workspace,
                usage_limits=self._limits(),
            ) as events:
                async for event in events:
                    if isinstance(event, PartStartEvent) and isinstance(
                        event.part, TextPart
                    ):
                        if event.part.content:
                            yield ("delta", event.part.content)
                    elif isinstance(event, PartDeltaEvent) and isinstance(
                        event.delta, TextPartDelta
                    ):
                        if event.delta.content_delta:
                            yield ("delta", event.delta.content_delta)
                    elif isinstance(event, AgentRunResultEvent):
                        yield ("result", self._result(event.result, workspace, started))


async def probe(config: AgentConfig) -> AgentResult:
    """Test API, routing, and model access without exposing workspace tools."""
    started = time.monotonic()
    provider = _provider(config)
    model = OpenRouterModel(
        config.model,
        provider=provider,
        settings=OpenRouterModelSettings(
            max_tokens=8,
            openrouter_provider=config.provider_policy,
            openrouter_usage={"include": True},
        ),
    )
    raw = await asyncio.wait_for(
        Agent(
            model,
            instructions="You are a connectivity probe. Reply with exactly: OK",
        ).run(
            "Reply with exactly: OK",
            usage_limits=UsageLimits(request_limit=1),
        ),
        timeout=min(config.timeout_seconds, 90),
    )
    usage = raw.usage
    return AgentResult(
        output=str(raw.output or "").strip(),
        model=str(raw.response.model_name or ""),
        duration_ms=int((time.monotonic() - started) * 1000),
        requests=usage.requests,
        cost_usd=float(usage.cost) if usage.cost is not None else None,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
    )
