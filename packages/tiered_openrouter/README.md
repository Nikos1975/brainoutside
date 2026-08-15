# tiered-openrouter

A small, framework-independent Python package for bounded OpenRouter agents.
It combines PydanticAI with a read-only Markdown workspace, explicit provider
routing, structured output, streaming, token and cost limits, timeouts, and
exact source-path tracking.

The package deliberately contains no Django or BrainOutside imports. A host
application remains responsible for authentication, selecting the permitted
workspace root, keeping secrets, recording usage, and applying business-level
rate or daily-spend limits.

## Install

From this repository:

    pip install ./packages/tiered_openrouter

From another project, copy this directory or depend on its Git URL and
subdirectory after the package is published on a stable branch or tag.

## Use in another project

    import asyncio
    import os
    from decimal import Decimal
    from pathlib import Path

    from tiered_openrouter import (
        AgentConfig,
        OpenRouterAgent,
        ScopedMarkdownWorkspace,
        cheapest_provider_policy,
    )

    async def main():
        workspace = ScopedMarkdownWorkspace(Path("./knowledge/public"))
        config = AgentConfig(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model="poolside/laguna-s-2.1:free",
            instructions=(
                "Answer from the workspace. Start with INDEX.md and read only "
                "the files needed for the question."
            ),
            provider_policy=cheapest_provider_policy(data_collection="allow"),
            max_cost_usd=Decimal("0.05"),
        )
        result = await OpenRouterAgent(config).run(
            "What are the main operating principles?",
            workspace,
        )
        print(result.output)
        print(result.source_paths)

    asyncio.run(main())

For private content, set data_collection to deny. If all eligible providers
support it, zero_data_retention can also be enabled.

## Optional proxies and compressors

AgentConfig accepts base_url, so a compatible local proxy can be inserted
without changing application code. Keep this optional: first run a direct
baseline, then compare cost, latency, and answer quality with the proxy.

Headroom is the better candidate for inline Python compression because it is
Apache-2.0 and exposes a Python library. Caveman also documents a PydanticAI
proxy route, but its engine-linked runtime uses BSL-1.1. Do not install both in
the same request path: they overlap, add latency, and make failures harder to
attribute.

Compression is intentionally not enabled by this package. For private notes,
verify that the chosen mode is local, disable telemetry where applicable, and
run retrieval-quality evaluations before enabling lossy transforms.

## Security boundary

- The model receives only Markdown beneath the resolved workspace root.
- Absolute paths, parent traversal, non-Markdown reads, and symlink escapes are
  rejected.
- Listing filenames does not claim them as sources. Reading a file or returning
  its matching lines does.
- Tool output, request count, tool-call count, output tokens, cost, and wall
  time are bounded independently.
- Provider data collection is an explicit host decision, not an implicit
  default.
