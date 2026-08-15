from decimal import Decimal

import pytest
from pydantic_ai import ToolOutput
from tiered_openrouter import AgentConfig, OpenRouterAgent, cheapest_provider_policy
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
