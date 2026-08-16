"""Configuration contract for provider selection and tier routing."""
from __future__ import annotations

import pytest

from apps.brainconfig import services as config


def test_provider_rejects_unknown_values():
    with pytest.raises(ValueError):
        config._clean_ai_provider("mystery")


def test_provider_normalises_case():
    assert config._clean_ai_provider(" OpenRouter ") == "openrouter"


def test_public_reader_uses_public_model(monkeypatch):
    values = {
        "OPENROUTER_MODEL_PUBLIC": "poolside/laguna-s-2.1:free",
        "OPENROUTER_MODEL_READER": "deepseek/deepseek-v4-flash-0731",
    }
    monkeypatch.setattr(config, "get", values.__getitem__)

    assert (
        config.openrouter_model_for("reader", "public")
        == "poolside/laguna-s-2.1:free"
    )
    assert (
        config.openrouter_model_for("reader", "agents-only")
        == "deepseek/deepseek-v4-flash-0731"
    )
    assert (
        config.openrouter_model_for("reader", "private")
        == "deepseek/deepseek-v4-flash-0731"
    )


def test_feeder_never_uses_public_model(monkeypatch):
    values = {
        "OPENROUTER_MODEL_PUBLIC": "poolside/laguna-s-2.1:free",
        "OPENROUTER_MODEL_FEEDER": "deepseek/deepseek-v4-flash-0731",
    }
    monkeypatch.setattr(config, "get", values.__getitem__)

    assert (
        config.openrouter_model_for("feeder", "agents-only")
        == "deepseek/deepseek-v4-flash-0731"
    )


def test_fallback_price_map_is_exact_model_scoped(monkeypatch):
    monkeypatch.setattr(
        config,
        "get",
        lambda key: (
            '{"model/a":{"input_per_million":"1",'
            '"output_per_million":"2"}}'
        ),
    )

    assert config.openrouter_fallback_prices("model/a") == {
        "input_per_million": "1",
        "output_per_million": "2",
    }
    assert config.openrouter_fallback_prices("model/b") is None


def test_fallback_price_map_rejects_negative_and_unknown_rates():
    with pytest.raises(ValueError, match="non-negative"):
        config._clean_openrouter_fallback_prices(
            '{"model/a":{"input_per_million":"-1",'
            '"output_per_million":"2"}}'
        )
    with pytest.raises(ValueError, match="unknown keys"):
        config._clean_openrouter_fallback_prices(
            '{"model/a":{"input_per_million":"1",'
            '"output_per_million":"2","request":"3"}}'
        )
