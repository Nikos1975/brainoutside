from decimal import Decimal

import pytest
from pydantic_ai import ToolOutput, UsageLimitExceeded
from pydantic_ai.usage import RunUsage
from tiered_openrouter import (
    AgentConfig,
    FallbackPricedUsageLimits,
    OpenRouterAgent,
    TokenPrices,
    cheapest_provider_policy,
)
from tiered_openrouter.agent import _provider


def test_cheapest_policy_keeps_privacy_explicit():
    assert cheapest_provider_policy(data_collection="deny") == {
        "sort": "price",
        "allow_fallbacks": True,
        "require_parameters": True,
        "data_collection": "deny",
    }


def test_config_rejects_unbounded_or_missing_values():
    with pytest.raises(ValueError):
        AgentConfig(
            api_key="",
            model="model",
            instructions="",
            provider_policy={},
        )
    with pytest.raises(ValueError):
        AgentConfig(
            api_key="key",
            model="model",
            instructions="",
            provider_policy={},
            max_cost_usd=Decimal(0),
        )


def test_fallback_pricing_does_not_double_count_cache_tokens():
    prices = TokenPrices(
        input_per_million=Decimal(2),
        output_per_million=Decimal(10),
        cache_read_per_million=Decimal("0.5"),
        cache_write_per_million=Decimal(3),
    )
    usage = RunUsage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=200_000,
        cache_write_tokens=100_000,
    )

    # 700k ordinary input + 200k cache read + 100k cache write + 100k output.
    assert prices.cost(usage) == Decimal("2.8")


def test_fallback_prices_default_cache_buckets_to_input_rate():
    prices = TokenPrices(
        input_per_million=Decimal(1),
        output_per_million=Decimal(2),
    )
    usage = RunUsage(
        input_tokens=1_000,
        output_tokens=100,
        cache_read_tokens=800,
    )

    assert prices.cost(usage) == Decimal("0.0012")


def test_fallback_cost_is_enforced_without_native_price():
    limits = FallbackPricedUsageLimits(
        cost_limit=Decimal("0.001"),
        fallback_prices=TokenPrices(
            input_per_million=Decimal(1),
            output_per_million=Decimal(2),
        ),
    )

    with pytest.raises(UsageLimitExceeded, match="source=fallback"):
        limits.check_cost(RunUsage(input_tokens=1_000, output_tokens=1))


def test_missing_native_and_fallback_prices_fails_closed():
    limits = FallbackPricedUsageLimits(cost_limit=Decimal(1))

    with pytest.raises(UsageLimitExceeded, match="no fallback token prices"):
        limits.check_cost(RunUsage(input_tokens=1))


def test_native_cost_takes_precedence_over_fallback_estimate():
    limits = FallbackPricedUsageLimits(
        cost_limit=Decimal(1),
        fallback_prices=TokenPrices(
            input_per_million=Decimal(999),
            output_per_million=Decimal(999),
        ),
    )
    usage = RunUsage(input_tokens=1_000_000, cost=Decimal("0.25"))

    assert limits.effective_cost(usage) == (Decimal("0.25"), "provider")
    limits.check_cost(usage)


def test_json_schema_uses_portable_tool_output():
    output = OpenRouterAgent._output_type(
        {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
    )

    assert isinstance(output, ToolOutput)
    assert output.name == "agent_result"


def test_tools_are_sequential_without_unsupported_provider_parameter(monkeypatch):
    for name in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    runner = OpenRouterAgent(
        AgentConfig(
            api_key="key",
            model="poolside/laguna-s-2.1:free",
            instructions="",
            provider_policy={},
        )
    )

    assert "parallel_tool_calls" not in runner.model.settings
    tools = runner._agent()._function_toolset.tools
    assert tools
    assert all(tool.sequential for tool in tools.values())


def test_compatible_proxy_is_configuration_not_a_dependency(monkeypatch):
    for name in (
        "ALL_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "all_proxy",
        "https_proxy",
        "http_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    provider = _provider(
        AgentConfig(
            api_key="upstream-key",
            model="deepseek/model",
            instructions="",
            provider_policy={},
            base_url="http://compressor.internal:8787/v1",
            default_headers={"X-Proxy-Key": "proxy-key"},
        )
    )

    assert str(provider.client.base_url) == "http://compressor.internal:8787/v1/"
    assert provider.client.default_headers["X-Proxy-Key"] == "proxy-key"
